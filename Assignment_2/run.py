from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys


waypoints = [
    {"lat": 47.3977419, "lon": 8.5455939, "alt": 50},
    {"lat": 47.3978419, "lon": 8.5456939, "alt": 55},
    {"lat": 47.3979419, "lon": 8.5457939, "alt": 60},
    {"lat": 47.3980419, "lon": 8.5458939, "alt": 65},
    {"lat": 47.3981419, "lon": 8.5459939, "alt": 70},
    {"lat": 47.3982419, "lon": 8.5460939, "alt": 75},
    {"lat": 47.3983419, "lon": 8.5461939, "alt": 80},
    {"lat": 47.3984419, "lon": 8.5462939, "alt": 85},
    {"lat": 47.3985419, "lon": 8.5463939, "alt": 90},
    {"lat": 47.3986419, "lon": 8.5464939, "alt": 95},
    {"lat": 47.3987419, "lon": 8.5465939, "alt": 100},
    {"lat": 47.3988419, "lon": 8.5466939, "alt": 95},
    {"lat": 47.3989419, "lon": 8.5467939, "alt": 90},
    {"lat": 47.3990419, "lon": 8.5468939, "alt": 85},
    {"lat": 47.3991419, "lon": 8.5469939, "alt": 80},
]


def connect_vehicle():
    try:
        print("Attempting to connect to vehicle...")

        connection_string = "udp:127.0.0.1:14550"

        vehicle = connect(connection_string, wait_ready=True, timeout=60, heartbeat_timeout=30)

        print("Successfully connected to vehicle!")
        return vehicle

    except Exception as e:
        print(f"Error connecting to vehicle: {e}")
        print("Please ensure SITL is running properly")
        sys.exit(1)


def get_distance_between_points(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth's radius
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def calculate_perpendicular_point(wp1, wp2, distance=100):
    """Calculate a point 100m perpendicular to the direction of travel"""
    # Convert to radians
    lat1, lon1 = math.radians(wp1["lat"]), math.radians(wp1["lon"])
    lat2, lon2 = math.radians(wp2["lat"]), math.radians(wp2["lon"])

    # Calculate bearing
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(y, x)

    # Calculate perpendicular bearing
    perp_bearing = bearing + math.pi / 2

    # Calculate new point
    R = 6371000  # Earth's radius
    d = distance  # Distance

    new_lat = math.asin(math.sin(lat1) * math.cos(d / R) + math.cos(lat1) * math.sin(d / R) * math.cos(perp_bearing))
    new_lon = lon1 + math.atan2(
        math.sin(perp_bearing) * math.sin(d / R) * math.cos(lat1), math.cos(d / R) - math.sin(lat1) * math.sin(new_lat)
    )

    new_lat = math.degrees(new_lat)
    new_lon = math.degrees(new_lon)

    return {"lat": new_lat, "lon": new_lon, "alt": wp1["alt"]}


def main():
    print("Connecting to vehicle...")
    vehicle = connect_vehicle()

    cmds = vehicle.commands
    cmds.clear()
    cmds.upload()

    # Takeoff
    cmds.add(
        Command(
            0,
            0,
            0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            waypoints[0]["lat"],
            waypoints[0]["lon"],
            waypoints[0]["alt"],
        )
    )

    for i, wp in enumerate(waypoints):
        if i == 10:
            perp_wp = calculate_perpendicular_point(waypoints[i - 1], wp)
            cmds.add(
                Command(
                    0,
                    0,
                    0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    perp_wp["lat"],
                    perp_wp["lon"],
                    perp_wp["alt"],
                )
            )

        cmds.add(
            Command(
                0,
                0,
                0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0,
                0,
                0,
                0,
                0,
                0,
                wp["lat"],
                wp["lon"],
                wp["alt"],
            )
        )

    cmds.add(
        Command(
            0,
            0,
            0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0,
            0,
            0,
            0,
            0,
            waypoints[-1]["lat"],
            waypoints[-1]["lon"],
            0,
        )
    )

    cmds.upload()

    vehicle.mode = VehicleMode("AUTO")

    while True:
        nextwaypoint = vehicle.commands.next

        remaining_distance = 0
        for i in range(nextwaypoint, len(waypoints)):
            if i > 0:
                remaining_distance += get_distance_between_points(
                    waypoints[i - 1]["lat"], waypoints[i - 1]["lon"], waypoints[i]["lat"], waypoints[i]["lon"]
                )

        estimated_time = remaining_distance / 5

        print(f"Next waypoint: {nextwaypoint}")
        print(f"Remaining distance: {remaining_distance:.1f} meters")
        print(f"Estimated time: {timedelta(seconds=estimated_time)}")

        if nextwaypoint == len(waypoints):
            print("Mission complete!")
            break

        time.sleep(1)

    plt.figure(figsize=(10, 10))
    lats = [wp["lat"] for wp in waypoints]
    lons = [wp["lon"] for wp in waypoints]

    perp_wp = calculate_perpendicular_point(waypoints[9], waypoints[10])
    lats.insert(10, perp_wp["lat"])
    lons.insert(10, perp_wp["lon"])

    plt.plot(lons, lats, "b-", label="Flight path")
    plt.plot(lons[0], lats[0], "go", label="Start")
    plt.plot(lons[-1], lats[-1], "ro", label="End")
    plt.scatter(lons[1:-1], lats[1:-1], c="yellow", label="Waypoints")

    plt.title("Drone Mission Path")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    vehicle.close()


if __name__ == "__main__":
    main()
