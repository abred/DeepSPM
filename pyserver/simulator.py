"""
Simulated STM backend for development and testing.

Generates synthetic scan images with Gaussian atom blobs.
No real hardware required — runs anywhere Python + NumPy are available.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanResult:
    """Result of an STM scan."""
    img_forward: np.ndarray
    img_backward: np.ndarray
    offset_nm: np.ndarray
    size_nm: np.ndarray


class STMBackend(abc.ABC):
    """Minimal interface that the server expects from any STM backend."""

    @abc.abstractmethod
    def scan_image(
        self, size_nm: float, offset_nm: np.ndarray,
        pixel: int, bias_mv: float,
    ) -> ScanResult:
        ...

    @abc.abstractmethod
    def tip_form(self, z_angstrom: float, x_nm: float, y_nm: float) -> None:
        ...

    @abc.abstractmethod
    def ramp_bias(self, target_mv: float) -> None:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...


class SimulatorBackend(STMBackend):
    """
    Simulated STM environment.

    Models a flat surface with adatoms rendered as 2D Gaussians.
    Suitable for testing the full DeepSPM agent loop without hardware.

    Parameters
    ----------
    noise_level : float
        Gaussian noise std added to scan images (default 0.05).
    seed : int, optional
        Random seed for reproducibility.
    initial_atoms : list of [x, y] pairs, optional
        Initial atom positions in nm.
    """

    def __init__(
        self,
        noise_level: float = 0.05,
        seed: int | None = None,
        initial_atoms: list[list[float]] | None = None,
    ) -> None:
        self._noise = noise_level
        self._rng = np.random.default_rng(seed)
        self._bias_mv = 100.0
        self._lattice = 0.288  # Ag(111) lattice constant in nm

        if initial_atoms:
            self._atoms = np.array(initial_atoms, dtype=float)
        else:
            self._atoms = np.array([[0.0, 0.0]])

        logger.info(
            "Simulator ready: %d atoms on surface", len(self._atoms)
        )

    def scan_image(
        self, size_nm: float, offset_nm: np.ndarray,
        pixel: int, bias_mv: float,
    ) -> ScanResult:
        self._bias_mv = bias_mv
        img = self._render(size_nm, offset_nm, pixel)
        img_bwd = img + self._rng.normal(
            0, self._noise * 0.3, img.shape
        )
        return ScanResult(
            img_forward=img,
            img_backward=img_bwd,
            offset_nm=np.array(offset_nm),
            size_nm=np.array([size_nm, size_nm]),
        )

    def tip_form(
        self, z_angstrom: float, x_nm: float, y_nm: float,
    ) -> None:
        logger.info(
            "Tip form at (%.2f, %.2f) nm, z=%.1f A",
            x_nm, y_nm, z_angstrom,
        )

    def ramp_bias(self, target_mv: float) -> None:
        self._bias_mv = target_mv

    @property
    def name(self) -> str:
        return "Simulator"

    def _render(
        self, size_nm: float, offset_nm: np.ndarray, pixel: int,
    ) -> np.ndarray:
        """Render synthetic STM image with atom blobs."""
        img = np.zeros((pixel, pixel), dtype=float)

        x_min = offset_nm[0] - size_nm / 2
        x_max = offset_nm[0] + size_nm / 2
        y_min = offset_nm[1] - size_nm / 2
        y_max = offset_nm[1] + size_nm / 2

        xs = np.linspace(x_min, x_max, pixel)
        ys = np.linspace(y_min, y_max, pixel)
        xx, yy = np.meshgrid(xs, ys)

        sigma = 0.065  # nm, typical adatom FWHM

        for atom in self._atoms:
            r2 = (xx - atom[0]) ** 2 + (yy - atom[1]) ** 2
            img += np.exp(-r2 / (2 * sigma ** 2))

        # Background lattice corrugation
        k = 2 * np.pi / self._lattice
        img += 0.03 * (np.cos(k * xx) + np.cos(k * yy))

        # Noise
        img += self._rng.normal(0, self._noise, img.shape)
        return img
