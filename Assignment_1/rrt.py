import numpy as np


class Node:
    def __init__(self, position, time):
        self.position = np.array(position)
        self.time = time
        self.parent = None


class RRT:
    def __init__(self, start, goal, v, start_time):
        self.start = start
        self.goal = goal
        self.v = v 
        self.start_time = start_time
        self.nodes = [Node(start, start_time)]
        self.goal_node = None
        self.max_iter = 1000
        self.step_size = 10  

    def distance(self, node1, node2):
        return np.linalg.norm(node1.position - node2.position)

    def nearest_node(self, new_position):
        distances = [self.distance(node, Node(new_position, 0)) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_node, to_position):
        direction = np.array(to_position) - from_node.position
        length = np.linalg.norm(direction)
        if length == 0:
            return from_node  # Avoid division by zero
        travel_time = length / self.v
        direction = (direction / length) * min(self.step_size, length)
        new_position = from_node.position + direction
        new_time = from_node.time + travel_time
        new_node = Node(new_position, new_time)
        new_node.parent = from_node
        return new_node

    def plan(self):
        for _ in range(self.max_iter):
            rand_pos = np.random.randint(0, 101, size=3)  # Random position within 0-100 grid
            nearest = self.nearest_node(rand_pos)
            new_node = self.steer(nearest, rand_pos)

            # Check for collision in space-time
            if not self.is_collision_free(new_node):
                continue

            self.nodes.append(new_node)

            if self.distance(new_node, Node(self.goal, 0)) < self.step_size:
                self.goal_node = new_node
                break

        if self.goal_node:
            return self.extract_path()
        else:
            return None

    def is_collision_free(self, node):
        for n in self.nodes:
            # Check if new node is too close in space and time
            if np.linalg.norm(node.position - n.position) < self.step_size and abs(node.time - n.time) < (
                self.step_size / self.v
            ):
                return False
        return True

    def extract_path(self):
        path = []
        current = self.goal_node
        while current is not None:
            path.append([*current.position, current.time])
            current = current.parent
        return path[::-1]  # Reverse to get from start to goal
