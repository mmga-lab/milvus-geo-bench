"""
Dataset generation module for Milvus geo benchmark tool.
"""

import logging
import math
import random
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from .utils import ensure_dir, save_parquet


class DatasetGenerator:
    """Generate datasets for geo search benchmarks."""

    def __init__(self, config: dict[str, Any]):
        self.config = config["dataset"]
        self.bbox = self.config["bbox"]  # [min_lon, min_lat, max_lon, max_lat]
        self.min_points_per_query = self.config["min_points_per_query"]

        # Calculate maximum possible search radius based on bbox
        bbox_width = self.bbox[2] - self.bbox[0]  # max_lon - min_lon
        bbox_height = self.bbox[3] - self.bbox[1]  # max_lat - min_lat
        self.max_search_radius = min(bbox_width, bbox_height) / 2

        # Setup coordinate transformation for distance calculations
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.utm = pyproj.CRS("EPSG:3857")  # Web Mercator
        self.project = pyproj.Transformer.from_crs(self.wgs84, self.utm, always_xy=True).transform
        self.unproject = pyproj.Transformer.from_crs(self.utm, self.wgs84, always_xy=True).transform

        logging.info(f"DatasetGenerator initialized with bbox: {self.bbox}")

    def generate_train_data(self, num_points: int) -> pd.DataFrame:
        """Generate training data with points and vectors."""
        logging.info(f"Generating {num_points} training points...")

        # Generate random points within bounding box
        points = []
        vectors = []

        for _ in tqdm(range(num_points), desc="Generating training data"):
            # Random point in bbox
            lon = random.uniform(self.bbox[0], self.bbox[2])
            lat = random.uniform(self.bbox[1], self.bbox[3])
            wkt = f"POINT({lon} {lat})"
            points.append(wkt)

            # Random 8D vector normalized to unit sphere
            vec = np.random.normal(0, 1, 8)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec.tolist())

        df = pd.DataFrame({"id": range(1, num_points + 1), "wkt": points, "vec": vectors})

        logging.info(f"Generated {len(df)} training points")
        return df

    def generate_test_queries(self, train_df: pd.DataFrame, num_queries: int) -> pd.DataFrame:
        """Generate test queries that guarantee minimum number of results."""
        logging.info(f"Generating {num_queries} test queries...")

        # Parse training points and cache coordinates for faster distance calculation
        train_points = []
        train_coords = []
        for wkt in train_df["wkt"]:
            coords = self._parse_point_wkt(wkt)
            point = Point(coords[0], coords[1])
            train_points.append(point)
            train_coords.append(coords)
        # Convert to numpy array for vectorized operations
        train_coords_array = np.array(train_coords)

        queries = []
        query_id = 1

        with tqdm(total=num_queries, desc="Generating test queries") as pbar:
            attempts = 0
            max_attempts = num_queries * 10  # Prevent infinite loop

            while len(queries) < num_queries and attempts < max_attempts:
                attempts += 1

                # Randomly select a center point from training data
                center_idx = random.randint(0, len(train_points) - 1)
                center_point = train_points[center_idx]
                center_lon, center_lat = center_point.x, center_point.y

                # Find optimal radius using binary search
                radius = self._find_optimal_radius(
                    center_point, train_coords_array, self.min_points_per_query
                )

                if radius is not None:
                    # Create hexagonal polygon around center
                    polygon = self._create_hexagon(center_lon, center_lat, radius)
                    polygon_wkt = polygon.wkt

                    # Create ST_WITHIN expression
                    expr = f"ST_WITHIN(location, '{polygon_wkt}')"

                    queries.append(
                        {
                            "query_id": query_id,
                            "expr": expr,
                            "polygon_wkt": polygon_wkt,
                            "center_lon": center_lon,
                            "center_lat": center_lat,
                            "radius": radius,
                        }
                    )

                    query_id += 1
                    pbar.update(1)

        if len(queries) < num_queries:
            logging.warning(f"Only generated {len(queries)} queries out of {num_queries} requested")

        df = pd.DataFrame(queries)
        logging.info(f"Generated {len(df)} test queries")
        return df

    def calculate_ground_truth(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ground truth results using Shapely vectorized operations."""
        logging.info("Calculating ground truth...")

        # Parse all training points into a numpy array of Points
        train_points = []
        train_ids = train_df["id"].values
        for wkt in train_df["wkt"]:
            coords = self._parse_point_wkt(wkt)
            train_points.append(Point(coords[0], coords[1]))

        # Convert to numpy array for vectorized operations
        train_points_array = np.array(train_points, dtype=object)

        ground_truth = []

        for _, query_row in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Calculating ground truth"
        ):
            query_id = query_row["query_id"]
            polygon_wkt = query_row["polygon_wkt"]

            # Parse polygon
            polygon = self._parse_polygon_wkt(polygon_wkt)

            # Use vectorized within operation to match ST_WITHIN semantics
            within_mask = shapely.within(train_points_array, polygon)
            matching_ids = train_ids[within_mask].tolist()

            ground_truth.append(
                {
                    "query_id": query_id,
                    "result_ids": matching_ids,
                    "result_count": len(matching_ids),
                }
            )

        df = pd.DataFrame(ground_truth)
        logging.info(f"Calculated ground truth for {len(df)} queries")

        # Log statistics only if we have data
        if len(df) > 0:
            result_counts = df["result_count"].values
            logging.info(
                f"Result count stats - Min: {result_counts.min()}, "
                f"Max: {result_counts.max()}, "
                f"Mean: {result_counts.mean():.2f}, "
                f"Median: {np.median(result_counts):.2f}"
            )

        return df

    def generate_full_dataset(self, output_dir: str) -> dict[str, str]:
        """Generate complete dataset and save to parquet files."""
        output_path = ensure_dir(output_dir)

        # Generate train data
        train_df = self.generate_train_data(self.config["num_points"])
        train_file = output_path / "train.parquet"
        save_parquet(train_df, train_file)

        # Generate test queries
        test_df = self.generate_test_queries(train_df, self.config["num_queries"])
        test_file = output_path / "test.parquet"
        save_parquet(test_df, test_file)

        # Calculate ground truth
        ground_truth_df = self.calculate_ground_truth(train_df, test_df)
        ground_truth_file = output_path / "ground_truth.parquet"
        save_parquet(ground_truth_df, ground_truth_file)

        return {
            "train": str(train_file),
            "test": str(test_file),
            "ground_truth": str(ground_truth_file),
        }

    def _parse_point_wkt(self, wkt: str) -> tuple[float, float]:
        """Parse POINT WKT to coordinates."""
        # Extract coordinates from POINT(lon lat)
        coords_str = wkt.replace("POINT(", "").replace(")", "")
        lon, lat = map(float, coords_str.split())
        return lon, lat

    def _parse_polygon_wkt(self, wkt: str) -> Polygon:
        """Parse POLYGON WKT to Shapely Polygon."""
        # Extract coordinates from POLYGON ((x1 y1, x2 y2, ...)) or POLYGON((x1 y1, x2 y2, ...))
        # Handle both formats with and without space after POLYGON
        coords_str = (
            wkt.replace("POLYGON ((", "")
            .replace("POLYGON((", "")
            .replace("))", "")
            .replace(")", "")
        )
        coords = []
        for coord_pair in coords_str.split(", "):
            if coord_pair.strip():  # Skip empty strings
                lon, lat = map(float, coord_pair.split())
                coords.append((lon, lat))
        return Polygon(coords)

    def _find_optimal_radius(
        self, center: Point, points_coords: np.ndarray, min_count: int
    ) -> float:
        """Find optimal radius using binary search to contain exactly min_count points."""
        center_coords = np.array([center.x, center.y])

        # Calculate all distances once
        distances = np.sqrt(np.sum((points_coords - center_coords) ** 2, axis=1))

        if len(distances) < min_count:
            return None

        # Sort distances to enable binary search
        sorted_distances = np.sort(distances)

        # The optimal radius is the distance to the min_count-th nearest point
        # Add small margin to ensure we include the boundary point
        optimal_radius = sorted_distances[min_count - 1] * 1.001

        # Check if radius exceeds bbox bounds
        if optimal_radius > self.max_search_radius:
            return None

        return float(optimal_radius)

    def _find_suitable_radius(self, center: Point, points: list[Point], min_count: int) -> float:
        """Legacy method - kept for compatibility."""
        # Convert to numpy array and use fast method
        points_coords = np.array([[p.x, p.y] for p in points])
        return self._find_suitable_radius_fast(center, points_coords, min_count)

    def _calculate_distance(self, point1: Point, point2: Point) -> float:
        """Calculate distance between two points in degrees."""
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def _create_hexagon(self, center_lon: float, center_lat: float, radius: float) -> Polygon:
        """Create regular hexagon around center point."""
        vertices = []
        for i in range(6):
            angle = i * math.pi / 3  # 60 degrees
            x = center_lon + radius * math.cos(angle)
            y = center_lat + radius * math.sin(angle)
            vertices.append((x, y))

        # Close the polygon
        vertices.append(vertices[0])

        return Polygon(vertices)
