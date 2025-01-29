from typing import List, Tuple, Dict
from collections import defaultdict
import random
from utilities import Point3D, PathRequest


class Node:
    def __init__(self, point: Point3D, parent=None, time=0.0):
        self.point = point
        self.parent = parent
        self.time = time

    def distance_to(self, other: "Node") -> float:
        return self.point.distance_to(other.point)


class TemporalRRTPlanner:
    def __init__(self, bounds: Tuple[int, int, int], velocity: float):
        self.bounds = bounds
        self.velocity = velocity
        self.step_size = 5.0
        self.max_iterations = 10000
        self.occupied_space_time = defaultdict(set)

    def _is_valid_point(self, point: Point3D) -> bool:
        return 0 <= point.x <= self.bounds[0] and 0 <= point.y <= self.bounds[1] and 0 <= point.z <= self.bounds[2]

    def _random_point(self) -> Point3D:
        return Point3D(
            random.uniform(0, self.bounds[0]), random.uniform(0, self.bounds[1]), random.uniform(0, self.bounds[2])
        )

    def _nearest_node(self, nodes: List[Node], point: Point3D) -> Node:
        return min(nodes, key=lambda n: n.point.distance_to(point))

    def _steer(self, from_point: Point3D, to_point: Point3D) -> Point3D:
        dist = from_point.distance_to(to_point)
        if dist <= self.step_size:
            return to_point

        ratio = self.step_size / dist
        new_x = from_point.x + (to_point.x - from_point.x) * ratio
        new_y = from_point.y + (to_point.y - from_point.y) * ratio
        new_z = from_point.z + (to_point.z - from_point.z) * ratio

        return Point3D(new_x, new_y, new_z)

    def _is_collision_free(self, from_node: Node, to_point: Point3D) -> bool:
        # Check if path segments intersect with occupied space-time
        travel_time = from_node.point.distance_to(to_point) / self.velocity
        new_time = from_node.time + travel_time

        # Discretize the path and check for collisions
        steps = int(travel_time * 10)  # Check every 0.1 seconds
        if steps < 1:
            steps = 1

        for i in range(steps + 1):
            t = from_node.time + (travel_time * i / steps)
            ratio = i / steps

            x = from_node.point.x + (to_point.x - from_node.point.x) * ratio
            y = from_node.point.y + (to_point.y - from_node.point.y) * ratio
            z = from_node.point.z + (to_point.z - from_node.point.z) * ratio

            # Check if point at time t is occupied
            point_key = (round(x), round(y), round(z))
            time_key = round(t * 10) / 10  # Round to nearest 0.1s

            if time_key in self.occupied_space_time[point_key]:
                return False

        return True

    def _reconstruct_path(self, final_node: Node) -> List[Tuple[Point3D, float]]:
        path = []
        current = final_node
        while current is not None:
            path.append((current.point, current.time))
            current = current.parent
        return list(reversed(path))

    def plan_path(self, request: PathRequest) -> List[Tuple[Point3D, float]]:
        nodes = [Node(request.start, None, request.start_time)]

        for _ in range(self.max_iterations):
            if random.random() < 0.1:
                random_point = request.end  # Bias towards goal
            else:
                random_point = self._random_point()

            nearest_node = self._nearest_node(nodes, random_point)
            new_point = self._steer(nearest_node.point, random_point)

            if self._is_valid_point(new_point) and self._is_collision_free(nearest_node, new_point):
                travel_time = nearest_node.point.distance_to(new_point) / self.velocity
                new_node = Node(new_point, nearest_node, nearest_node.time + travel_time)
                nodes.append(new_node)

                # Check if close to goal
                if new_point.distance_to(request.end) < self.step_size:
                    final_node = Node(
                        request.end, new_node, new_node.time + new_point.distance_to(request.end) / self.velocity
                    )

                    if self._is_collision_free(new_node, request.end):
                        path = self._reconstruct_path(final_node)

                        # Mark path in occupied space-time
                        for i in range(len(path) - 1):
                            p1, t1 = path[i]
                            p2, t2 = path[i + 1]

                            steps = int((t2 - t1) * 10)  # Check every 0.1 seconds
                            if steps < 1:
                                steps = 1

                            for step in range(steps + 1):
                                t = t1 + (t2 - t1) * step / steps
                                ratio = step / steps

                                x = p1.x + (p2.x - p1.x) * ratio
                                y = p1.y + (p2.y - p1.y) * ratio
                                z = p1.z + (p2.z - p1.z) * ratio

                                point_key = (round(x), round(y), round(z))
                                time_key = round(t * 10) / 10
                                self.occupied_space_time[point_key].add(time_key)

                        return path

        raise Exception("Failed to find path within iteration limit")


def plan_multiple_paths(
    requests: List[PathRequest], bounds: Tuple[int, int, int], velocity: float
) -> Dict[int, List[Tuple[Point3D, float]]]:
    planner = TemporalRRTPlanner(bounds, velocity)
    paths = {}

    for i, request in enumerate(requests):
        try:
            path = planner.plan_path(request)
            paths[i] = path
        except Exception as e:
            print(f"Failed to plan path {i}: {str(e)}")

    return paths
