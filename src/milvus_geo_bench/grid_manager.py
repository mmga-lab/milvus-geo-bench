"""
Grid manager for spatial partitioning of large datasets.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any


class GridManager:
    """Manages spatial grid partitioning for large dataset generation."""

    def __init__(self, bbox: list[float], total_points: int, config: dict[str, Any]):
        """
        Initialize grid manager.

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            total_points: Total number of points to generate
            config: Grid configuration from dataset config
        """
        self.bbox = bbox
        self.total_points = total_points
        self.config = config

        # Calculate grid dimensions first
        self.lon_range = self.bbox[2] - self.bbox[0]  # max_lon - min_lon
        self.lat_range = self.bbox[3] - self.bbox[1]  # max_lat - min_lat

        # Calculate grid layout
        if config.get("auto_calculate", True):
            self.num_grids, self.grid_layout = self._auto_calculate_layout()
        else:
            # Manual configuration
            self.num_grids = config.get("num_grids", 1)
            self.grid_layout = self._calculate_manual_layout()

        self.points_per_grid = total_points // self.num_grids

        logging.info(
            f"GridManager initialized: {self.num_grids} grids ({self.grid_layout[0]}x{self.grid_layout[1]}), "
            f"{self.points_per_grid:,} points per grid"
        )

    def _auto_calculate_layout(self) -> tuple[int, tuple[int, int]]:
        """Auto-calculate optimal grid layout."""
        target_points_per_grid = self.config.get("target_points_per_grid", 1_000_000)
        min_grids = self.config.get("min_grids", 1)
        max_grids = self.config.get("max_grids", 1000)

        # Calculate number of grids needed
        num_grids = max(
            min_grids, min(max_grids, math.ceil(self.total_points / target_points_per_grid))
        )

        # Find best rectangular layout
        grid_layout = self._find_best_grid_layout(num_grids)
        actual_grids = grid_layout[0] * grid_layout[1]

        return actual_grids, grid_layout

    def _calculate_manual_layout(self) -> tuple[int, int]:
        """Calculate grid layout for manual configuration."""
        if "grid_layout" in self.config:
            rows, cols = self.config["grid_layout"]
            return (rows, cols)
        else:
            # Square layout
            return self._find_best_grid_layout(self.num_grids)

    def _find_best_grid_layout(self, target_grids: int) -> tuple[int, int]:
        """
        Find the best rectangular grid layout.
        Prefers layouts that match the aspect ratio of the bounding box.
        """
        bbox_aspect_ratio = self.lon_range / self.lat_range

        best_layout = (1, target_grids)
        best_score = float("inf")

        for rows in range(1, int(math.sqrt(target_grids)) + 1):
            if target_grids % rows == 0:
                cols = target_grids // rows

                # Calculate aspect ratio of this layout
                layout_aspect_ratio = cols / rows

                # Score based on how close it matches bbox aspect ratio
                score = abs(layout_aspect_ratio - bbox_aspect_ratio)

                if score < best_score:
                    best_score = score
                    best_layout = (rows, cols)

        return best_layout

    def get_grid_info(self, grid_id: int) -> dict[str, Any]:
        """Get information about a specific grid."""
        if grid_id < 0 or grid_id >= self.num_grids:
            raise ValueError(f"Grid ID {grid_id} out of range [0, {self.num_grids - 1}]")

        rows, cols = self.grid_layout
        row = grid_id // cols
        col = grid_id % cols

        # Calculate grid boundaries
        lon_step = self.lon_range / cols
        lat_step = self.lat_range / rows

        min_lon = self.bbox[0] + col * lon_step
        max_lon = min_lon + lon_step
        min_lat = self.bbox[1] + row * lat_step
        max_lat = min_lat + lat_step

        return {
            "grid_id": grid_id,
            "row": row,
            "col": col,
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "center_lon": (min_lon + max_lon) / 2,
            "center_lat": (min_lat + max_lat) / 2,
            "area_deg2": lon_step * lat_step,
        }

    def point_to_grid(self, lon: float, lat: float) -> int:
        """Determine which grid a point belongs to."""
        if not (self.bbox[0] <= lon <= self.bbox[2] and self.bbox[1] <= lat <= self.bbox[3]):
            raise ValueError(f"Point ({lon}, {lat}) is outside bounding box {self.bbox}")

        rows, cols = self.grid_layout

        # Handle edge case where point is exactly on the boundary
        col = min(int((lon - self.bbox[0]) / self.lon_range * cols), cols - 1)
        row = min(int((lat - self.bbox[1]) / self.lat_range * rows), rows - 1)

        grid_id = row * cols + col
        return grid_id

    def get_all_grid_info(self) -> list[dict[str, Any]]:
        """Get information for all grids."""
        return [self.get_grid_info(grid_id) for grid_id in range(self.num_grids)]

    def save_metadata(self, output_dir: Path) -> str:
        """Save grid metadata to JSON file."""
        metadata = {
            "total_points": self.total_points,
            "num_grids": self.num_grids,
            "grid_layout": self.grid_layout,
            "points_per_grid": self.points_per_grid,
            "bbox": self.bbox,
            "config": self.config,
            "grids": self.get_all_grid_info(),
        }

        metadata_file = output_dir / "grid_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Saved grid metadata to {metadata_file}")
        return str(metadata_file)

    def load_metadata(self, metadata_file: str) -> dict[str, Any]:
        """Load grid metadata from JSON file."""
        metadata_path = Path(metadata_file)
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

        # Validate compatibility
        if metadata["bbox"] != self.bbox:
            logging.warning("Loaded grid metadata has different bounding box")

        return metadata

    def get_statistics(self) -> dict[str, Any]:
        """Get grid statistics."""
        grid_area_deg2 = (self.lon_range / self.grid_layout[1]) * (
            self.lat_range / self.grid_layout[0]
        )

        return {
            "num_grids": self.num_grids,
            "grid_layout": f"{self.grid_layout[0]}x{self.grid_layout[1]}",
            "points_per_grid": self.points_per_grid,
            "grid_area_deg2": grid_area_deg2,
            "total_area_deg2": self.lon_range * self.lat_range,
            "point_density_per_deg2": self.total_points / (self.lon_range * self.lat_range),
        }

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"GridManager(grids={stats['num_grids']}, layout={stats['grid_layout']}, "
            f"points_per_grid={stats['points_per_grid']:,})"
        )
