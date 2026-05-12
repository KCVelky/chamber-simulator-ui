from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Basic geometry / helpers
# -------------------------

Vector3 = np.ndarray  # shape (3,)
ArrayNx3 = np.ndarray  # shape (N, 3)


def as_vec3(x: float, y: float, z: float) -> Vector3:
    v = np.array([x, y, z], dtype=float)
    if v.shape != (3,):
        raise ValueError("Vector3 must have shape (3,)")
    return v


def ensure_nx3(arr: np.ndarray, name: str) -> ArrayNx3:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    return arr


# -------------------------
# Boundary conditions
# -------------------------

class Wall(str, Enum):
    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    Z_MIN = "z_min"  # floor
    Z_MAX = "z_max"  # ceiling


class BoundaryType(str, Enum):
    ANECHOIC = "anechoic"
    RIGID = "rigid"
    IMPEDANCE = "impedance"  # future: freq-dependent impedance/absorption


@dataclass(frozen=True)
class BoundaryCondition:
    kind: BoundaryType
    # Placeholder for future models:
    # - impedance: complex Z(f)
    # - absorption: alpha(f)
    params: Dict[str, float] = field(default_factory=dict)


# -------------------------
# Room / scene definitions
# -------------------------

@dataclass(frozen=True)
class Room3D:
    """Axis-aligned rectangular room."""
    size: Vector3  # [Lx, Ly, Lz] in meters
    boundaries: Dict[Wall, BoundaryCondition]

    def __post_init__(self) -> None:
        if self.size.shape != (3,):
            raise ValueError("Room size must be Vector3 shape (3,)")
        if np.any(self.size <= 0):
            raise ValueError("Room dimensions must be strictly positive.")
        # Ensure all walls are defined (for robustness)
        missing = {w for w in Wall} - set(self.boundaries.keys())
        if missing:
            raise ValueError(f"Missing boundary conditions for walls: {missing}")

    @property
    def Lx(self) -> float:
        return float(self.size[0])

    @property
    def Ly(self) -> float:
        return float(self.size[1])

    @property
    def Lz(self) -> float:
        return float(self.size[2])

    def contains(self, points: ArrayNx3, margin: float = 0.0) -> np.ndarray:
        """Return boolean mask for points inside room with optional margin."""
        p = ensure_nx3(points, "points")
        return (
            (p[:, 0] >= margin) & (p[:, 0] <= self.Lx - margin) &
            (p[:, 1] >= margin) & (p[:, 1] <= self.Ly - margin) &
            (p[:, 2] >= margin) & (p[:, 2] <= self.Lz - margin)
        )

    def assert_inside(self, points: ArrayNx3, margin: float = 0.0, name: str = "points") -> None:
        mask = self.contains(points, margin=margin)
        if not np.all(mask):
            bad = np.where(~mask)[0].tolist()
            raise ValueError(f"{name} contains points outside the room (indices: {bad}).")


# -------------------------
# Microphones / arrays
# -------------------------

@dataclass(frozen=True)
class MicArray:
    positions: ArrayNx3  # (M, 3)
    name: str = "mic_array"

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", ensure_nx3(self.positions, "MicArray.positions"))

    @property
    def n_mics(self) -> int:
        return int(self.positions.shape[0])


# -------------------------
# Sources
# -------------------------

class SourceType(str, Enum):
    MONOPOLE = "monopole"
    DIPOLE = "dipole"  # future extension
    # etc.


@dataclass(frozen=True)
class Source:
    position: Vector3
    kind: SourceType = SourceType.MONOPOLE
    name: str = "source"
    orientation: Optional[Vector3] = None  # useful for dipole later

    def __post_init__(self) -> None:
        if self.position.shape != (3,):
            raise ValueError("Source.position must be Vector3 shape (3,)")
        if self.orientation is not None and self.orientation.shape != (3,):
            raise ValueError("Source.orientation must be Vector3 shape (3,)")


# -------------------------
# Scene (container)
# -------------------------

@dataclass(frozen=True)
class Scene:
    room: Room3D
    mic_array: MicArray
    sources: List[Source]

    def validate(self, margin: float = 0.05) -> None:
        """Validate that all objects are inside room with a safety margin."""
        self.room.assert_inside(self.mic_array.positions, margin=margin, name="mic_array.positions")
        src_pos = np.array([s.position for s in self.sources], dtype=float)
        self.room.assert_inside(src_pos, margin=margin, name="sources.positions")


# -------------------------
# Factory helpers
# -------------------------

def make_semi_anechoic_room(
    Lx: float, Ly: float, Lz: float,
    floor_rigid: bool = True
) -> Room3D:
    """Typical semi-anechoic: rigid floor, other walls anechoic (placeholder)."""
    bc_anechoic = BoundaryCondition(kind=BoundaryType.ANECHOIC)
    bc_rigid = BoundaryCondition(kind=BoundaryType.RIGID)

    boundaries = {
        Wall.X_MIN: bc_anechoic,
        Wall.X_MAX: bc_anechoic,
        Wall.Y_MIN: bc_anechoic,
        Wall.Y_MAX: bc_anechoic,
        Wall.Z_MAX: bc_anechoic,
        Wall.Z_MIN: bc_rigid if floor_rigid else bc_anechoic,
    }
    return Room3D(size=as_vec3(Lx, Ly, Lz), boundaries=boundaries)
