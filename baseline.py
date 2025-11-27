# ELB baseline prototype

# 1. Reads crime incidents from a CSV and draws a circle around each point with a fixed radius.
# 2. Loads a set of campus routes from routes.json
# 3. Simulate walks along each route in 10 m steps, with
#     a) sleep logic based on a safe region
#     b) one alert per risk circle when entering 100 m alert range.
# 4. For each route, output - baseline_routeX_alerts.csv, baseline_routeX_summary.csv, baseline_routeX_plot.png

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

# Inputs
INCIDENT_CSV_PATH = "GTPD_incidents.csv"
ROUTES_JSON_PATH = "routes.json"

# Configs
RISK_RADIUS = 80.0
ALERT_RANGE = 100.0   # alert when boundary is 100 m close to the current location
STEP = 10.0
SAFE_REGION = 350.0   # if everything is farther than this, app can go to hibernation
HIBERNATE_STEPS = 5

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

def run_baseline_for_route(route_points: List[Tuple[float, float]], route_id: int, route_name: str) -> None:
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

        elb_distance = min(
            p.distance(circle)
            for circle in risk_circles
        )

        asleep = False

        # hibernate logic
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
            continue   # skip elb checks in sleep mode

        for pid, circle in enumerate(risk_circles):
            if fired_for_circle[pid]:
                continue

            d = dist_to_boundary(p, circle)
            checks_attempted += 1

            within_alert_range = d <= ALERT_RANGE
            inside_now = circle.contains(p)
            inside_next_step = circle.contains(p_next)

            # trigger either when we enter 100 m range or actually cross into the circle
            if within_alert_range or ((not inside_now) and inside_next_step):
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
    alerts_file = f"baseline_route{route_id}_alerts.csv"
    summary_file = f"baseline_route{route_id}_summary.csv"
    plot_file = f"baseline_route{route_id}_plot.png"

    alerts_df = pd.DataFrame([a.__dict__ for a in alerts])
    alerts_df.to_csv(alerts_file, index=False)

    summary_row = {
        "route_id": route_id,
        "route_name": route_name,
        "route_length_m": round(route.length, 1),
        "total_steps": len(steps) - 1,
        "total_alerts": len(alerts),
        "unique_circles_alerted": len({a.risk_id for a in alerts}),
        "num_incidents": len(incident_points),
        "checks_attempted": checks_attempted,
        "sleep_steps": sleep_steps,
        "ALERT_RANGE": ALERT_RANGE,
        "RISK_RADIUS": RISK_RADIUS,
        "SAFE_REGION": SAFE_REGION,
        "hibernate_steps": HIBERNATE_STEPS,
        "STEP": STEP,
    }
    pd.DataFrame([summary_row]).to_csv(summary_file, index=False)

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

    for idx, pt in enumerate(steps):
        if idx % 10 == 0:
            ax.annotate(
                str(idx),
                (pt.x, pt.y),
                textcoords="offset points",
                xytext=(2, 2),
                fontsize=9,
                color="gray",
            )

    if not alerts_df.empty:
        ax.scatter(
            alerts_df["latitude"],
            alerts_df["longitude"],
            s=320,
            marker="*",
            color="#0A6CFF",
            edgecolor="black",
            linewidths=1.0,
            zorder=6,
            label="Alert",
        )
        for _, row in alerts_df.iterrows():
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
    ax.set_title(f"ELB Baseline – Route {route_id} ({route_name})")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(plot_file, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[route {route_id} / {route_name}] wrote "
        f"{alerts_file}, {summary_file}, {plot_file}"
    )

if __name__ == "__main__":
    for route in ROUTES:
        run_baseline_for_route(
            route_points=route["points"],
            route_id=route["id"],
            route_name=route["name"],
        )
