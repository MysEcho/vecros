from dataclasses import dataclass
import math
import torch


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def distance_to(self, other: "Point3D") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def to_tensor(self):
        return torch.tensor([self.x, self.y, self.z], dtype=torch.float32, device="cuda")

    @classmethod
    def from_tensor(cls, tensor):
        return cls(*tensor.cpu().tolist())


@dataclass
class PathRequest:
    start: Point3D
    end: Point3D
    start_time: float
