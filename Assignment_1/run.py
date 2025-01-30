from utilities import Point3D, PathRequest
from rrt import plan_multiple_paths
from vizualize import visualize_paths

bounds = (100, 100, 100)
velocity = 1.0

requests = [
    PathRequest(Point3D(14, 23, 65), Point3D(67, 32, 73), start_time=0.0),
    PathRequest(Point3D(91, 79, 43), Point3D(18, 24, 91), start_time=0.0),
]

paths = plan_multiple_paths(requests, bounds, velocity)

visualize_paths(paths, bounds, save_animation=True)
