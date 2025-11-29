import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

INCIDENT_CSV_PATH = "GTPD_incidents.csv"
ROUTES_JSON_PATH = "routes.json"

RISK_RADIUS = 80.0
ALERT_RANGE = 100.0
STEP = 10.0
SAFE_REGION = 350.0
HIBERNATE_STEPS = 5

# Direction filter configs
DIRECTION_CONE_ANGLE = 45.0   # half-angle (degrees)
CONE_DIST = 120.0
# Detour config
SAFETY_BUFFER_AROUND_CIRCLE = 25.0

def load_routes(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    routes = []
    raw_routes = data.get("routes", [])
    for idx, route in enumerate(raw_routes, start=1):
        route_id = int(route.get("id", idx))
        route_name = route.get("name", f"route_{route_id}")

        points = []
        for x, y in route.get("points", []):
            points.append((float(x), float(y)))

        routes.append(
            {
                "id": route_id,
                "name": route_name,
                "points": points,
            }
        )

    return routes

ROUTES = load_routes(ROUTES_JSON_PATH)

# Load incidents
incident_file = pd.read_csv(INCIDENT_CSV_PATH)
incident_points = []
for lat, lon in incident_file[["LocationLatitude", "LocationLongitude"]].itertuples(index=False):
    incident_points.append((float(lat), float(lon)))

# Create risk circles
risk_circles = []
for lat, lon in incident_points:
    circle = Point(lat, lon).buffer(RISK_RADIUS)
    risk_circles.append(circle)

def steps_along_route_line(line: LineString, step: float) -> List[Point]:
    points = [line.interpolate(0.0)]
    dist = step
    while dist < line.length:
        points.append(line.interpolate(dist))
        dist += step
    points.append(line.interpolate(line.length))
    return points

def dist_to_boundary(pt: Point, poly: Polygon) -> float:
    return pt.distance(poly)

def get_heading_angle(p1: Point, p2: Point) -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    ang = math.degrees(math.atan2(dy, dx))
    return ang if ang >= 0 else ang + 360.0

def angle_difference(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180

# GPT helped
def is_alert_inside_direction_cone(user_pos: Point, user_direction: float, center: Tuple[float, float]) -> bool:
    cx, cy = center
    target = Point(cx, cy)

    if user_pos.distance(target) > CONE_DIST:
        return False

    heading_angle = get_heading_angle(user_pos, target)
    return abs(angle_difference(heading_angle, user_direction)) <= DIRECTION_CONE_ANGLE

# GPT helped - to visually show how direction cone works
def sector_polygon(center: Tuple[float, float], heading_angle: float, half_angle_deg: float, radius_m: float, n: int = 60) -> Polygon:
    cx, cy = center
    start = math.radians(heading_angle - half_angle_deg)
    end = math.radians(heading_angle + half_angle_deg)
    angles = [start + t * (end - start) / n for t in range(n + 1)]

    pts = [(cx, cy)]
    for a in angles:
        x = cx + radius_m * math.cos(a)
        y = cy + radius_m * math.sin(a)
        pts.append((x, y))
    pts.append((cx, cy))
    return Polygon(pts)

@dataclass
class AlertEvent:
    step_id: int
    latitude: float
    longitude: float
    risk_id: int
    IncidentFromDate: str
    IncidentToDate: str
    CaseStatus: str
    Description: str

def path_intersects_circles(points: List[Point], circles: List[Polygon]) -> bool:
    line = LineString([(p.x, p.y) for p in points])
    return any(line.intersects(c) for c in circles)

def build_detour_route(
    alert_point: Point,
    dest_point: Point,
    circles: List[Polygon],
) -> List[Point]:

    # Get straight line from alert to destination
    straight_line = LineString([(alert_point.x, alert_point.y), (dest_point.x, dest_point.y)])

    # Get circles that intersect the straight line
    blocking_circles = [c for c in circles if straight_line.intersects(c)]
    if not blocking_circles:
        return [alert_point, dest_point]

    # Pick the circle that blocks the straight path and is closest to the alert point
    main_circle = min(blocking_circles, key=lambda c: alert_point.distance(c))
    circle_center = main_circle.centroid

    # Find the current direction from alert to dest
    dx = dest_point.x - alert_point.x
    dy = dest_point.y - alert_point.y
    dist = math.hypot(dx, dy)
    if dist == 0:
        dist = 1.0
    ux = dx / dist
    uy = dy / dist

    # Get perpendicular (left or right) directions to current direction
    left_x, left_y = -uy, ux
    right_x, right_y = uy, -ux

    def path_is_safe(detour_point: Point) -> bool:
        path_points = [alert_point, detour_point, dest_point]
        return not path_intersects_circles(path_points, circles)

    # Place detour point away from risk circle
    safe_gap = RISK_RADIUS + SAFETY_BUFFER_AROUND_CIRCLE
    for tries in range(1, 9):
        offset = safe_gap * tries

        left_detour_point = Point(
            circle_center.x + left_x * offset,
            circle_center.y + left_y * offset,
        )
        if path_is_safe(left_detour_point):
            return [alert_point, left_detour_point, dest_point]

        right_detour_point = Point(
            circle_center.x + right_x * offset,
            circle_center.y + right_y * offset,
        )
        if path_is_safe(right_detour_point):
            return [alert_point, right_detour_point, dest_point]

    # fallback if no path exists
    return [alert_point, dest_point]

def run_refinement_for_route(route_points: List[Tuple[float, float]], route_id: int, route_name: str) -> None:
    route = LineString(route_points)
    steps = steps_along_route_line(route, STEP)
    alerts: List[AlertEvent] = []

    # track already alerted risk circles
    fired_for_circle: Dict[int, bool] = {idx: False for idx in range(len(risk_circles))}
    hibernate_counter = 0
    sleep_steps = 0
    checks_attempted = 0

    for i in range(len(steps) - 1):
        p = steps[i]
        p_next = steps[i + 1]
        heading_angle = get_heading_angle(p, p_next)

        elb_distance = min(
            p.distance(circle)
            for circle in risk_circles
        )

        asleep = False

        if elb_distance > SAFE_REGION and hibernate_counter == 0:
            hibernate_counter = HIBERNATE_STEPS
            asleep = True
        elif hibernate_counter > 0:
            asleep = True
            hibernate_counter -= 1

        if asleep:
            nearest_dist = min(dist_to_boundary(p, circle) for circle in risk_circles)
            if nearest_dist <= ALERT_RANGE:
                asleep = False
                hibernate_counter = 0

        if asleep:
            sleep_steps += 1
            continue

        # Direction filter logic
        risk_candidates = []
        for pid, (cx, cy) in enumerate(incident_points):
            if fired_for_circle[pid]:
                continue
            if is_alert_inside_direction_cone(p, heading_angle, (cx, cy)):
                risk_candidates.append((pid, risk_circles[pid]))

        for pid, circle in risk_candidates:
            d = dist_to_boundary(p, circle)
            checks_attempted += 1

            within_alert_range = d <= ALERT_RANGE
            inside_now = circle.contains(p)
            inside_next_step = circle.contains(p_next)

            if within_alert_range or ((not inside_now) and inside_next_step):
                if not fired_for_circle[pid]:
                    incident_row = incident_file.iloc[pid]
                    alerts.append(
                        AlertEvent(
                            step_id=i,
                            latitude=p.x,
                            longitude=p.y,
                            risk_id=pid + 1,
                            IncidentFromDate=str(incident_row["IncidentFromDate"]),
                            IncidentToDate=str(incident_row["IncidentToDate"]),
                            CaseStatus=str(incident_row["CaseStatus"]),
                            Description=str(incident_row["Offense Description"]),
                        )
                    )
                    fired_for_circle[pid] = True

    # CSV Outputs
    alerts_file = f"refinement_route{route_id}_alerts.csv"
    summary_file = f"refinement_route{route_id}_summary.csv"
    direction_filter_plot = f"refinement_route{route_id}_direction_filter_plot.png"
    detour_plot = f"refinement_route{route_id}_detour_plot.png"

    alerts_table = pd.DataFrame([a.__dict__ for a in alerts])
    alerts_table.to_csv(alerts_file, index=False)

    fig, ax = plt.subplots(figsize=(12, 5))

    for poly in risk_circles:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.25, color="red")

    for pid, (cx, cy) in enumerate(incident_points, start=1):
        ax.annotate(
            str(pid),
            (cx, cy),
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=5,
        )

    rx, ry = route.xy
    ax.plot(rx, ry, linewidth=2.5, color="black", label="Route")

    ax.scatter(
        [pt.x for pt in steps],
        [pt.y for pt in steps],
        s=10,
        alpha=0.55,
        color="black",
        label=f"Steps ({STEP:.0f} m)",
    )

    # visualize direction filter
    if len(steps) > 1:
        p0 = steps[0]
        hdg0 = get_heading_angle(p0, steps[1])
        cone_poly = sector_polygon((p0.x, p0.y), hdg0, DIRECTION_CONE_ANGLE, CONE_DIST, n=50)
        xs, ys = cone_poly.exterior.xy
        ax.fill(xs, ys, alpha=0.12, color="tab:blue")

    if not alerts_table.empty:
        ax.scatter(
            alerts_table["latitude"],
            alerts_table["longitude"],
            s=320,
            marker="*",
            color="#0A6CFF",
            edgecolor="black",
            linewidths=1.0,
            zorder=6,
            label="Alert",
        )
        for _, row in alerts_table.iterrows():
            ax.annotate(
                str(int(row["risk_id"])),
                (row["latitude"], row["longitude"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=6,
                color="black",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                zorder=7,
            )

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Refinement - Route {route_id} ({route_name})")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(direction_filter_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Route logic for safe detour
    detour_line = route
    detour_length = float(route.length)
    
    if not alerts_table.empty:
        first_alert = alerts_table.iloc[0]  # assuming user clicked safer route button on UI
        step_id = int(first_alert["step_id"])

        alert_point = steps[step_id]
        dest_x, dest_y = route_points[-1]  # assuming user entered the last route point as destination
        dest_point = Point(dest_x, dest_y)

        detour_route: List[Point] = build_detour_route(
            alert_point=alert_point,
            dest_point=dest_point,
            circles=risk_circles,
        )
        origin_to_alert_coordinates = [(pt.x, pt.y) for pt in steps[: step_id + 1]]
        alert_to_dest_coordinates = [(pt.x, pt.y) for pt in detour_route[1:]]

        detour_line = LineString(origin_to_alert_coordinates + alert_to_dest_coordinates)
        detour_length = float(detour_line.length)

    # Detour plot - GPT
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    for poly in risk_circles:
        xs, ys = poly.exterior.xy
        ax2.fill(xs, ys, alpha=0.25, color="red")

    for pid, (cx, cy) in enumerate(incident_points, start=1):
        ax2.annotate(
            str(pid),
            (cx, cy),
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=5,
        )

    rx, ry = route.xy
    ax2.plot(rx, ry, linewidth=2.5, color="black", label="Original Route")

    dx, dy = detour_line.xy
    ax2.plot(dx, dy, linewidth=3.0, color="tab:blue", label="Safe Detour")

    if not alerts_table.empty:
        ax2.scatter(
            alerts_table["latitude"],
            alerts_table["longitude"],
            s=320,
            marker="*",
            color="#0A6CFF",
            edgecolor="black",
            linewidths=1.0,
            zorder=6,
            label="Alert",
        )

    ax2.set_aspect("equal", "box")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(f"Refinement – Safe Route {route_id} ({route_name})")
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(detour_plot, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    summary_row = {
        "route_id": route_id,
        "route_name": route_name,
        "route_length_m": round(route.length, 1),
        "detour_length": round(detour_length, 1),
        "detour_extra_m": round(detour_length - float(route.length), 1),
        "total_steps": len(steps) - 1,
        "total_alerts": len(alerts_table),
        "unique_circles_alerted": len(set(alerts_table["risk_id"])) if not alerts_table.empty else 0,
        "num_incidents": len(incident_points),
        "checks_attempted": checks_attempted,
        "sleep_steps": sleep_steps,
        "ALERT_RANGE": ALERT_RANGE,
        "RISK_RADIUS": RISK_RADIUS,
        "SAFE_REGION": SAFE_REGION,
        "hibernate_steps": HIBERNATE_STEPS,
        "STEP": STEP,
        "DIRECTION_CONE_ANGLE": DIRECTION_CONE_ANGLE,
        "CONE_DIST": CONE_DIST,
    }
    pd.DataFrame([summary_row]).to_csv(summary_file, index=False)

    print(
        f"[route {route_id} / {route_name}] wrote "
        f"{alerts_file}, {summary_file}, {direction_filter_plot}, {detour_plot}"
    )


if __name__ == "__main__":
    for route in ROUTES:
        run_refinement_for_route(
            route_points=route["points"],
            route_id=route["id"],
            route_name=route["name"],
        )
