import matplotlib.pyplot as plt
import numpy as np
from util import plan_paths
from mpl_toolkits.mplot3d import Axes3D


starts = [[0, 0, 0], [2, 4, 6]]
ends = [[89, 21, 10], [89, 21, 10]]
v = 1
time = 0

paths = plan_paths(starts, ends, v, time)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for path in paths:
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
plt.show()
