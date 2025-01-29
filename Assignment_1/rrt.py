from typing import List, Tuple, Dict
from collections import defaultdict
from utilities import Point3D, PathRequest
import torch


class Node:
    def __init__(self, point: Point3D, parent=None, time=0.0):
        self.point = point
        self.parent = parent
        self.time = time

    def distance_to(self, other: "Node") -> float:
        return self.point.distance_to(other.point)


class TemporalRRTPlanner:
    def __init__(self, bounds: Tuple[int, int, int], velocity: float):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bounds = torch.tensor(bounds, dtype=torch.float32, device=self.device)
        self.velocity = velocity
        self.step_size = 5.0
        self.max_iterations = 10000
        self.occupied_space_time = defaultdict(set)

    def _is_valid_point(self, point: Point3D) -> bool:
        return 0 <= point.x <= self.bounds[0] and 0 <= point.y <= self.bounds[1] and 0 <= point.z <= self.bounds[2]

    def _random_point(self) -> Point3D:
        random_point = torch.rand(3, device=self.device) * self.bounds
        return Point3D(*random_point.tolist())

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
        start_tensor = request.start.to_tensor()
        end_tensor = request.end.to_tensor()

        nodes = [Node(request.start, None, request.start_time)]
        node_tensors = torch.stack([start_tensor])

        batch_size = 1000

        for _ in range(0, self.max_iterations, batch_size):
            random_tensors = torch.rand((batch_size, 3), device=self.device) * self.bounds.unsqueeze(0)

            # Apply goal bias
            goal_mask = torch.rand(batch_size, device=self.device) < 0.1
            random_tensors[goal_mask] = end_tensor

            # Find nearest nodes for all random points at once
            distances = torch.cdist(random_tensors, node_tensors)
            nearest_indices = torch.argmin(distances, dim=1)
            nearest_positions = node_tensors[nearest_indices]

            # Vectorized steering
            directions = random_tensors - nearest_positions
            distances = torch.norm(directions, dim=1, keepdim=True)
            mask = distances > self.step_size
            new_points = nearest_positions + directions * mask.float() * (
                self.step_size / torch.clamp(distances, min=1e-6)
            )

            # Validate points in batch
            valid_mask = torch.all((new_points >= 0) & (new_points <= self.bounds), dim=1)

            # Collision checking in batch
            nearest_times = torch.tensor([nodes[i].time for i in nearest_indices], device=self.device)
            travel_times = torch.norm(new_points - nearest_positions, dim=1) / self.velocity

            # Generate interpolated positions for collision checking
            steps = 10
            t = torch.linspace(0, 1, steps, device=self.device).unsqueeze(1).unsqueeze(0)
            interpolated_positions = nearest_positions.unsqueeze(1) + (new_points - nearest_positions).unsqueeze(1) * t
            interpolated_times = nearest_times.unsqueeze(1) + travel_times.unsqueeze(1) * t

            # Round positions and times for collision checking
            positions_rounded = torch.round(interpolated_positions)
            times_rounded = torch.round(interpolated_times * 10) / 10

            # Vectorized collision checking
            collision_free = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            # pos_time_pairs = torch.cat([positions_rounded.reshape(-1, 3), times_rounded.reshape(-1, 1)], dim=1)

            # Convert occupied space-time to tensor format for batch checking
            if len(self.occupied_space_time) > 0:
                # occupied_positions = torch.tensor(list(self.occupied_space_time.keys()), device=self.device)
                # occupied_times = torch.tensor(
                #     [list(times) for times in self.occupied_space_time.values()], device=self.device
                # )

                # Check collisions for all points and times
                for i in range(batch_size):
                    if valid_mask[i]:
                        pos_sequence = positions_rounded[i]
                        time_sequence = times_rounded[i]

                        for j in range(steps):
                            pos = tuple(pos_sequence[j].cpu().numpy())
                            time = time_sequence[j].item()

                            if pos in self.occupied_space_time and time in self.occupied_space_time[pos]:
                                collision_free[i] = False
                                break

            valid_indices = torch.where(valid_mask & collision_free)[0]
            for idx in valid_indices:
                new_point = Point3D.from_tensor(new_points[idx])
                parent_idx = nearest_indices[idx]
                travel_time = travel_times[idx].item()
                new_node = Node(new_point, nodes[parent_idx], nodes[parent_idx].time + travel_time)
                nodes.append(new_node)
                node_tensors = torch.cat([node_tensors, new_points[idx].unsqueeze(0)])

                # Check if goal reached
                if torch.norm(new_points[idx] - end_tensor) < self.step_size:
                    final_travel_time = torch.norm(end_tensor - new_points[idx]) / self.velocity
                    final_node = Node(request.end, new_node, new_node.time + final_travel_time.item())

                    # Verify final path for collisions
                    if self._is_collision_free(new_node, request.end):
                        path = self._reconstruct_path(final_node)

                        self._update_occupied_space_time(path)
                        return path

        raise Exception("Failed to find path within iteration limit")

    def _update_occupied_space_time(self, path: List[Tuple[Point3D, float]]):
        import torch

        """Efficiently update occupied space-time using tensor operations"""
        if len(path) < 2:
            return

        points = torch.stack([point.to_tensor() for point, _ in path])
        times = torch.tensor([time for _, time in path], device=points.device)

        # Generate interpolation points for all segments at once
        segments = torch.stack([points[:-1], points[1:]], dim=1)
        time_segments = torch.stack([times[:-1], times[1:]], dim=1)

        steps = 10
        t = torch.linspace(0, 1, steps, device=points.device).unsqueeze(1)

        # Interpolate all segments simultaneously
        interpolated_points = segments[:, 0].unsqueeze(1) + (segments[:, 1] - segments[:, 0]).unsqueeze(1) * t
        interpolated_times = (
            time_segments[:, 0].unsqueeze(1) + (time_segments[:, 1] - time_segments[:, 0]).unsqueeze(1) * t
        )

        # Round positions and times
        positions_rounded = torch.round(interpolated_points)
        times_rounded = torch.round(interpolated_times * 10) / 10

        for i in range(len(path) - 1):
            for j in range(steps):
                pos = tuple(positions_rounded[i, j].cpu().numpy())
                time = times_rounded[i, j].item()

                if pos not in self.occupied_space_time:
                    self.occupied_space_time[pos] = set()
                self.occupied_space_time[pos].add(time)


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
