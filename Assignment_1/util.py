from rrt import RRT


def plan_paths(starts, ends, v, time):
    paths = []
    for start, end in zip(starts, ends):
        rrt = RRT(start, end, v, time)
        path = rrt.plan()
        if path is None:
            print("No path found for start:", start, "end:", end)
        else:
            paths.append(path)
    return paths
