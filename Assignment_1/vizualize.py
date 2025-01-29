import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict
from utilities import Point3D


class PathVisualizer:
    def __init__(self, bounds: Tuple[int, int, int]):
        self.bounds = bounds
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

    def setup_plot(self):
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xlim(0, self.bounds[0])
        self.ax.set_ylim(0, self.bounds[1])
        self.ax.set_zlim(0, self.bounds[2])
        self.ax.grid(True)
        self.ax.set_title("3D Path Planning with Temporal Constraints")

    def plot_paths(self, paths: Dict[int, List[Tuple[Point3D, float]]]):
        # Set up distinct colors for each path
        colors = list(mcolors.TABLEAU_COLORS.values())

        for path_id, path in paths.items():
            color = colors[path_id % len(colors)]

            # Extract coordinates
            xs = [point.x for point, _ in path]
            ys = [point.y for point, _ in path]
            zs = [point.z for point, _ in path]

            # Plot path
            self.ax.plot(xs, ys, zs, color=color, linewidth=2, label=f"Path {path_id}")

            # Plot start and end points
            self.ax.scatter(xs[0], ys[0], zs[0], color=color, marker="o", s=100, label=f"Start {path_id}")
            self.ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker="s", s=100, label=f"End {path_id}")

        self.ax.legend()

    def create_animation(self, paths: Dict[int, List[Tuple[Point3D, float]]]):
        colors = list(mcolors.TABLEAU_COLORS.values())

        max_time = max(max(time for _, time in path) for path in paths.values())

        scatters = {}
        for path_id in paths:
            scatter = self.ax.scatter([], [], [], color=colors[path_id % len(colors)], label=f"Agent {path_id}")
            scatters[path_id] = scatter

        def update(frame_time):
            for path_id, path in paths.items():
                # Find current position by interpolating between path points
                current_pos = None
                for i in range(len(path) - 1):
                    point1, time1 = path[i]
                    point2, time2 = path[i + 1]

                    if time1 <= frame_time <= time2:
                        ratio = (frame_time - time1) / (time2 - time1)
                        current_pos = Point3D(
                            point1.x + (point2.x - point1.x) * ratio,
                            point1.y + (point2.y - point1.y) * ratio,
                            point1.z + (point2.z - point1.z) * ratio,
                        )
                        break

                if current_pos:
                    scatters[path_id]._offsets3d = ([current_pos.x], [current_pos.y], [current_pos.z])

            return list(scatters.values())

        frames = np.linspace(0, max_time, int(max_time * 10))
        anim = FuncAnimation(self.fig, update, frames=frames, interval=100, blit=True)

        return anim


def visualize_paths(
    paths: Dict[int, List[Tuple[Point3D, float]]], bounds: Tuple[int, int, int], save_animation: bool = False
):
    visualizer = PathVisualizer(bounds)
    visualizer.setup_plot()
    visualizer.plot_paths(paths)
    anim = visualizer.create_animation(paths)

    if save_animation:
        anim.save("path_animation.gif", writer="pillow")

    plt.show()
