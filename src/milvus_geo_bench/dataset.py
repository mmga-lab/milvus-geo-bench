"""
Dataset generation module for Milvus geo benchmark tool.
"""

import logging
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from .grid_manager import GridManager
from .utils import ensure_dir, load_parquet, save_parquet


class DatasetGenerator:
    """Generate datasets for geo search benchmarks."""

    def __init__(self, config: dict[str, Any]):
        self.config = config["dataset"]
        self.bbox = self.config["bbox"]  # [min_lon, min_lat, max_lon, max_lat]
        self.min_points_per_query = self.config["min_points_per_query"]

        # Grid configuration
        self.grid_config = self.config.get("grid", {})
        self.grid_enabled = self.grid_config.get("enabled", False)
        self.grid_manager = None

        # Calculate maximum possible search radius based on bbox
        bbox_width = self.bbox[2] - self.bbox[0]  # max_lon - min_lon
        bbox_height = self.bbox[3] - self.bbox[1]  # max_lat - min_lat
        self.max_search_radius = min(bbox_width, bbox_height) / 2

        # Setup coordinate transformation for distance calculations
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.utm = pyproj.CRS("EPSG:3857")  # Web Mercator
        self.project = pyproj.Transformer.from_crs(self.wgs84, self.utm, always_xy=True).transform
        self.unproject = pyproj.Transformer.from_crs(self.utm, self.wgs84, always_xy=True).transform

        logging.info(
            f"DatasetGenerator initialized with bbox: {self.bbox}, grid_enabled: {self.grid_enabled}"
        )

    def generate_train_data(self, num_points: int) -> pd.DataFrame:
        """Generate training data with points and vectors."""
        if self.grid_enabled:
            return self._generate_train_data_with_grid(num_points)
        else:
            return self._generate_train_data_legacy(num_points)

    def _generate_train_data_legacy(self, num_points: int) -> pd.DataFrame:
        """Original training data generation method."""
        logging.info(f"Generating {num_points} training points (legacy mode)...")

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

    def _generate_train_data_with_grid(self, num_points: int) -> pd.DataFrame:
        """Generate training data using grid partitioning."""
        logging.info(f"Generating {num_points} training points with grid partitioning...")

        # Initialize grid manager
        self.grid_manager = GridManager(self.bbox, num_points, self.grid_config)

        # This method returns an empty DataFrame as actual data is saved per grid
        # The real work is done in generate_full_dataset
        return pd.DataFrame()

    def generate_test_queries(self, train_df: pd.DataFrame, num_queries: int) -> pd.DataFrame:
        """Generate test queries that guarantee minimum number of results."""
        if self.grid_enabled:
            # For grid mode, this method shouldn't be called with train_df
            raise ValueError("Use _generate_test_queries_with_grid for grid mode")
        return self._generate_test_queries_legacy(train_df, num_queries)

    def _generate_test_queries_legacy(
        self, train_df: pd.DataFrame, num_queries: int
    ) -> pd.DataFrame:
        """Original test query generation method."""
        logging.info(f"Generating {num_queries} test queries (legacy mode)...")

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

                # Find optimal rectangle size
                half_width, half_height = self._find_optimal_rectangle_size(
                    center_point, train_coords_array, self.min_points_per_query
                )

                if half_width is not None and half_height is not None:
                    # Create rectangular polygon around center
                    polygon = self._create_rectangle(
                        center_lon, center_lat, half_width, half_height
                    )
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
                            "half_width": half_width,
                            "half_height": half_height,
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

    def _generate_train_data_by_grid(self, train_dir: Path) -> list[str]:
        """Generate training data split by grids."""
        logging.info(f"Generating training data for {self.grid_manager.num_grids} grids...")

        train_files = []
        base_id = 1

        for grid_id in tqdm(range(self.grid_manager.num_grids), desc="Generating grid data"):
            # Get grid info
            grid_info = self.grid_manager.get_grid_info(grid_id)
            grid_bbox = grid_info["bbox"]

            # Generate points within this grid
            points_in_grid = self.grid_manager.points_per_grid
            grid_data = []

            for i in range(points_in_grid):
                # Random point within grid bbox
                lon = random.uniform(grid_bbox[0], grid_bbox[2])
                lat = random.uniform(grid_bbox[1], grid_bbox[3])
                wkt = f"POINT({lon} {lat})"

                # Random 8D vector normalized to unit sphere
                vec = np.random.normal(0, 1, 8)
                vec = vec / np.linalg.norm(vec)

                grid_data.append(
                    {
                        "id": base_id + i,
                        "wkt": wkt,
                        "vec": vec.tolist(),
                        "grid_id": grid_id,
                    }
                )

            # Save grid data
            grid_file = train_dir / f"train_grid_{grid_id}.parquet"
            grid_df = pd.DataFrame(grid_data)
            save_parquet(grid_df, grid_file)
            train_files.append(str(grid_file))

            # Update base_id for next grid
            base_id += points_in_grid

            # Clean up memory
            del grid_data
            del grid_df

        logging.info(f"Generated {len(train_files)} grid files")
        return train_files

    def generate_full_dataset(self, output_dir: str) -> dict[str, str]:
        """Generate complete dataset and save to parquet files."""
        if self.grid_enabled:
            return self._generate_full_dataset_with_grid(output_dir)
        else:
            return self._generate_full_dataset_legacy(output_dir)

    def _generate_full_dataset_legacy(self, output_dir: str) -> dict[str, str]:
        """Original dataset generation method."""
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

    def _generate_full_dataset_with_grid(self, output_dir: str) -> dict[str, str]:
        """Generate dataset using grid partitioning for memory efficiency."""
        output_path = ensure_dir(output_dir)

        # Initialize grid manager
        num_points = self.config["num_points"]
        self.grid_manager = GridManager(self.bbox, num_points, self.grid_config)

        # Create train subdirectory for grid files
        train_dir = ensure_dir(output_path / "train")

        # Generate training data by grid
        train_files = self._generate_train_data_by_grid(train_dir)

        # Generate test queries (before shuffling train files)
        test_df = self._generate_test_queries_with_grid(output_path)
        test_file = output_path / "test.parquet"

        # Calculate ground truth (before shuffling anything)
        ground_truth_df = self._calculate_ground_truth_with_grid(test_df, output_path)

        # Now shuffle training data using pairwise approach
        shuffled_train_files = self._shuffle_grid_train_data_pairwise(train_dir, train_files)

        # Save and shuffle test queries
        save_parquet(test_df, test_file)
        self._shuffle_test_data(test_file)
        ground_truth_file = output_path / "ground_truth.parquet"
        save_parquet(ground_truth_df, ground_truth_file)

        # Save grid metadata
        metadata_file = self.grid_manager.save_metadata(output_path)

        return {
            "train": str(train_dir),  # Directory containing grid files
            "train_files": shuffled_train_files,  # List of shuffled grid files
            "test": str(test_file),
            "ground_truth": str(ground_truth_file),
            "metadata": metadata_file,
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

    def _find_optimal_rectangle_size(
        self,
        center: Point,
        points_coords: np.ndarray,
        min_count: int,
        grid_bbox: list | None = None,
    ) -> tuple[float, float]:
        """
        Find optimal rectangle size to contain at least min_count points.

        Args:
            center: Center point
            points_coords: Training point coordinates array
            min_count: Minimum point count requirement
            grid_bbox: Grid boundary [min_lon, min_lat, max_lon, max_lat] (optional)

        Returns:
            (half_width, half_height) tuple, or (None, None) if invalid
        """
        center_lon, center_lat = center.x, center.y

        # Calculate distances to center (separately for x and y directions)
        dx = np.abs(points_coords[:, 0] - center_lon)
        dy = np.abs(points_coords[:, 1] - center_lat)

        # Find minimum rectangle containing at least min_count points
        # Use Chebyshev distance (max(dx, dy)) for rectangular containment
        chebyshev_distances = np.maximum(dx, dy)
        sorted_indices = np.argsort(chebyshev_distances)

        if len(sorted_indices) < min_count:
            return None, None

        # Get the position of the min_count-th point
        target_idx = sorted_indices[min_count - 1]

        # Calculate required half-width and half-height (add small margin for boundary points)
        half_width = dx[target_idx] * 1.001
        half_height = dy[target_idx] * 1.001

        # Apply grid boundary constraints if provided
        if grid_bbox is not None:
            min_lon, min_lat, max_lon, max_lat = grid_bbox

            # Calculate maximum allowed half-width and half-height
            max_half_width = min(center_lon - min_lon, max_lon - center_lon)
            max_half_height = min(center_lat - min_lat, max_lat - center_lat)

            # Apply boundary constraints (with safety margin to avoid floating point errors)
            half_width = min(half_width, max_half_width * 0.999)
            half_height = min(half_height, max_half_height * 0.999)

        # Check minimum size threshold
        min_size_threshold = 0.01
        if half_width < min_size_threshold or half_height < min_size_threshold:
            return None, None

        return float(half_width), float(half_height)

    def _create_rectangle(
        self,
        center_lon: float,
        center_lat: float,
        half_width: float,
        half_height: float | None = None,
    ) -> Polygon:
        """
        Create rectangle centered at the given point.

        Args:
            center_lon: Center longitude
            center_lat: Center latitude
            half_width: Half width of rectangle (longitude direction)
            half_height: Half height of rectangle (latitude direction), uses half_width if None

        Returns:
            Rectangle Polygon object
        """
        if half_height is None:
            half_height = half_width

        # Create rectangle vertices (counter-clockwise)
        vertices = [
            (center_lon - half_width, center_lat - half_height),  # bottom-left
            (center_lon + half_width, center_lat - half_height),  # bottom-right
            (center_lon + half_width, center_lat + half_height),  # top-right
            (center_lon - half_width, center_lat + half_height),  # top-left
            (center_lon - half_width, center_lat - half_height),  # close polygon
        ]

        return Polygon(vertices)

    def _generate_test_queries_with_grid(self, output_path: Path) -> pd.DataFrame:
        """Generate test queries distributed across grids."""
        num_queries = self.config["num_queries"]
        logging.info(f"Generating {num_queries} test queries with grid distribution...")

        queries_per_grid = num_queries // self.grid_manager.num_grids
        remainder_queries = num_queries % self.grid_manager.num_grids

        all_queries = []
        query_id = 1

        for grid_id in tqdm(range(self.grid_manager.num_grids), desc="Generating grid queries"):
            # Determine number of queries for this grid
            grid_queries = queries_per_grid
            if grid_id < remainder_queries:
                grid_queries += 1  # Distribute remainder queries

            if grid_queries == 0:
                continue

            # Get grid boundary information
            grid_info = self.grid_manager.get_grid_info(grid_id)
            grid_bbox = grid_info["bbox"]  # [min_lon, min_lat, max_lon, max_lat]

            # Load training data for this grid
            train_file = output_path / "train" / f"train_grid_{grid_id}.parquet"
            try:
                grid_train_df = load_parquet(train_file)
            except FileNotFoundError:
                logging.warning(
                    f"Grid file {train_file} not found, skipping queries for grid {grid_id}"
                )
                continue

            # Parse training points for this grid
            train_points = []
            train_coords = []
            for wkt in grid_train_df["wkt"]:
                coords = self._parse_point_wkt(wkt)
                point = Point(coords[0], coords[1])
                train_points.append(point)
                train_coords.append(coords)

            train_coords_array = np.array(train_coords)

            # Generate queries within this grid
            grid_query_list = []
            attempts = 0
            max_attempts = grid_queries * 10

            while len(grid_query_list) < grid_queries and attempts < max_attempts:
                attempts += 1

                # Randomly select a center point from this grid's training data
                if len(train_points) == 0:
                    break

                center_idx = random.randint(0, len(train_points) - 1)
                center_point = train_points[center_idx]
                center_lon, center_lat = center_point.x, center_point.y

                # Find optimal rectangle size using only this grid's points with boundary constraints
                half_width, half_height = self._find_optimal_rectangle_size(
                    center_point, train_coords_array, self.min_points_per_query, grid_bbox
                )

                if half_width is not None and half_height is not None:
                    # Create rectangular polygon around center
                    polygon = self._create_rectangle(
                        center_lon, center_lat, half_width, half_height
                    )
                    polygon_wkt = polygon.wkt

                    # Create ST_WITHIN expression
                    expr = f"ST_WITHIN(location, '{polygon_wkt}')"

                    grid_query_list.append(
                        {
                            "query_id": query_id,
                            "expr": expr,
                            "polygon_wkt": polygon_wkt,
                            "center_lon": center_lon,
                            "center_lat": center_lat,
                            "half_width": half_width,
                            "half_height": half_height,
                            "grid_id": grid_id,
                        }
                    )

                    query_id += 1

            all_queries.extend(grid_query_list)

            # Clean up memory
            del grid_train_df
            del train_points
            del train_coords

            logging.debug(f"Generated {len(grid_query_list)} queries for grid {grid_id}")

        df = pd.DataFrame(all_queries)
        logging.info(f"Generated {len(df)} test queries across {self.grid_manager.num_grids} grids")
        return df

    def _calculate_ground_truth_with_grid(
        self, test_df: pd.DataFrame, output_path: Path
    ) -> pd.DataFrame:
        """Calculate ground truth using grid-optimized approach."""
        logging.info("Calculating ground truth with grid optimization...")

        ground_truth = []

        # Group queries by grid_id for efficient processing
        grouped_queries = test_df.groupby("grid_id")

        for grid_id, grid_queries in tqdm(grouped_queries, desc="Calculating ground truth by grid"):
            # Load training data for this grid only
            train_file = output_path / "train" / f"train_grid_{grid_id}.parquet"
            try:
                grid_train_df = load_parquet(train_file)
            except FileNotFoundError:
                logging.warning(f"Grid file {train_file} not found, skipping GT for grid {grid_id}")
                continue

            # Parse training points into Shapely objects
            train_points = []
            train_ids = grid_train_df["id"].values

            for wkt in grid_train_df["wkt"]:
                coords = self._parse_point_wkt(wkt)
                train_points.append(Point(coords[0], coords[1]))

            # Convert to numpy array for vectorized operations
            train_points_array = np.array(train_points, dtype=object)

            # Process all queries for this grid
            for _, query_row in grid_queries.iterrows():
                query_id = query_row["query_id"]
                polygon_wkt = query_row["polygon_wkt"]

                # Parse polygon
                polygon = self._parse_polygon_wkt(polygon_wkt)

                # Use vectorized within operation
                within_mask = shapely.within(train_points_array, polygon)
                matching_ids = train_ids[within_mask].tolist()

                ground_truth.append(
                    {
                        "query_id": query_id,
                        "result_ids": matching_ids,
                        "result_count": len(matching_ids),
                        "grid_id": grid_id,
                    }
                )

            # Clean up memory
            del grid_train_df
            del train_points
            del train_points_array

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

            # Log grid distribution
            grid_counts = df.groupby("grid_id").size()
            logging.info(f"Queries per grid - Min: {grid_counts.min()}, Max: {grid_counts.max()}")

        return df

    def _shuffle_pair(self, file1: str, file2: str) -> None:
        """Shuffle data between two files"""
        logging.debug(f"Shuffling pair: {Path(file1).name} and {Path(file2).name}")

        # Load both files
        df1 = load_parquet(file1)
        df2 = load_parquet(file2)

        # Combine and shuffle
        combined = pd.concat([df1, df2], ignore_index=True)
        shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split back into two files maintaining original sizes
        split_point = len(df1)
        df1_new = shuffled.iloc[:split_point]
        df2_new = shuffled.iloc[split_point:]

        # Save back to original files
        save_parquet(df1_new, file1)
        save_parquet(df2_new, file2)

        # Clean up memory
        del df1, df2, combined, shuffled, df1_new, df2_new

    def _shuffle_grid_train_data_pairwise(
        self, train_dir: Path, train_files: list[str]
    ) -> list[str]:
        """Shuffle training data using pairwise approach for memory efficiency"""
        logging.info("Shuffling training data using pairwise approach...")

        current_files = train_files.copy()

        # Single pass shuffling
        logging.info("Performing pairwise shuffling...")

        # Shuffle pairs: (0,1), (2,3), (4,5)...
        for i in range(0, len(current_files) - 1, 2):
            self._shuffle_pair(current_files[i], current_files[i + 1])

        # Shuffle offset pairs: (1,2), (3,4), (5,6)... if we have enough files
        if len(current_files) > 2:
            for i in range(1, len(current_files) - 1, 2):
                self._shuffle_pair(current_files[i], current_files[i + 1])

        # Circular shuffle: pair the last file with the first file
        if len(current_files) > 1:
            self._shuffle_pair(current_files[-1], current_files[0])
            logging.debug("Performed circular shuffle between last and first file")

        # Rename files to new convention
        new_files = []
        for i, old_file in enumerate(current_files):
            old_path = Path(old_file)
            new_path = train_dir / f"train_{i:02d}.parquet"

            # Rename file
            old_path.rename(new_path)
            new_files.append(str(new_path))
            logging.debug(f"Renamed {old_path.name} to {new_path.name}")

        logging.info(f"Completed shuffling and renamed {len(new_files)} files")
        return new_files

    def _shuffle_test_data(self, test_file: Path) -> None:
        """Shuffle test queries in place"""
        logging.info("Shuffling test queries...")
        test_df = load_parquet(test_file)
        shuffled_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        save_parquet(shuffled_df, test_file)
        logging.info(f"Shuffled {len(shuffled_df)} test queries")
