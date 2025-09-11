"""
Milvus client wrapper for geo benchmark tool.
"""

import logging
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from pymilvus import DataType, MilvusClient
    from pymilvus.milvus_client import IndexParams
except ImportError as e:
    raise ImportError("pymilvus is required. Install it with: uv add pymilvus") from e


class MilvusGeoClient:
    """Milvus client for geo search operations."""

    def __init__(self, uri: str, token: str):
        """Initialize Milvus client with URI and token."""
        self.uri = uri
        self.token = token
        self.client = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Milvus server."""
        try:
            self.client = MilvusClient(uri=self.uri, token=self.token)
            logging.info(f"Connected to Milvus at {self.uri}")
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self, collection_name: str, recreate: bool = True) -> None:
        """Create collection with geo and vector fields."""
        try:
            # Drop collection if exists and recreate is True
            if recreate and self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                logging.info(f"Dropped existing collection: {collection_name}")

            # Define schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
                description="Collection for geo benchmark",
            )

            # Add fields
            schema.add_field(
                field_name="id", datatype=DataType.INT64, is_primary=True, description="Primary key"
            )

            schema.add_field(
                field_name="location",
                datatype=DataType.GEOMETRY,
                description="Geometry field for spatial data",
            )

            schema.add_field(
                field_name="embedding",
                datatype=DataType.FLOAT_VECTOR,
                dim=8,
                description="8-dimensional vector embedding",
            )

            # Create collection
            self.client.create_collection(
                collection_name=collection_name, schema=schema, consistency_level="Strong"
            )

            logging.info(f"Created collection: {collection_name}")

            # Create indexes
            self._create_indexes(collection_name)

        except Exception as e:
            logging.error(f"Failed to create collection {collection_name}: {e}")
            raise

    def _create_indexes(self, collection_name: str) -> None:
        """Create indexes for collection."""
        try:
            # Create vector index using IndexParams
            index_params = IndexParams()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 128},
            )

            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params,
            )
            logging.info(f"Created vector index for {collection_name}")

            # Create geometry index (RTREE for spatial data)
            try:
                geo_index_params = IndexParams()
                geo_index_params.add_index(
                    field_name="location",
                    index_type="RTREE",  # Use RTREE instead of GEOMETRY
                )

                self.client.create_index(
                    collection_name=collection_name,
                    index_params=geo_index_params,
                )
                logging.info(f"Created RTREE geometry index for {collection_name}")
            except Exception as e:
                logging.warning(f"Failed to create geometry index (may not be supported): {e}")

        except Exception as e:
            logging.error(f"Failed to create indexes for {collection_name}: {e}")
            raise

    def insert_data(
        self, collection_name: str, data_df: pd.DataFrame, batch_size: int = 1000
    ) -> None:
        """Insert data from DataFrame into collection."""
        logging.info(f"Inserting {len(data_df)} records into {collection_name}")

        try:
            # Prepare data
            total_rows = len(data_df)
            inserted_count = 0

            with tqdm(total=total_rows, desc="Inserting data") as pbar:
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = data_df.iloc[start_idx:end_idx]

                    # Prepare batch data
                    batch_data = []
                    for _, row in batch_df.iterrows():
                        record = {
                            "id": int(row["id"]),
                            "location": row["wkt"],  # WKT format
                            "embedding": row["vec"],
                        }
                        batch_data.append(record)

                    # Insert batch
                    self.client.insert(collection_name=collection_name, data=batch_data)

                    inserted_count += len(batch_data)
                    pbar.update(len(batch_data))

            logging.info(f"Successfully inserted {inserted_count} records")
            # flush data
            self.client.flush(collection_name)
            # Load collection to memory
            self.client.load_collection(collection_name)
            logging.info(f"Loaded collection {collection_name} to memory")

            # Wait for indexes to be ready
            logging.info("Waiting for indexes to be ready...")
            if self.wait_for_indexes_ready(collection_name, timeout=600):  # 10 minutes timeout
                logging.info("All indexes are ready. Data loading completed successfully.")
            else:
                logging.warning("Timeout waiting for indexes to be ready. Proceeding anyway.")

        except Exception as e:
            logging.error(f"Failed to insert data: {e}")
            raise

    def search_geo(
        self, collection_name: str, expr: str, timeout: int = 30
    ) -> tuple[list[int], float]:
        """Execute geo search query and return results with timing."""
        try:
            start_time = time.time()

            # Execute query
            results = self.client.query(
                collection_name=collection_name, filter=expr, output_fields=["id"], timeout=timeout
            )

            end_time = time.time()
            query_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Extract IDs from results
            result_ids = [result["id"] for result in results]

            return result_ids, query_time

        except Exception as e:
            logging.error(f"Failed to execute geo search: {e}")
            raise

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            stats = self.client.get_collection_stats(collection_name)
            return stats
        except Exception as e:
            logging.error(f"Failed to get collection stats: {e}")
            return {}

    def get_index_status(self, collection_name: str, field_name: str) -> dict[str, Any]:
        """Get index status for a specific field."""
        try:
            # First list all indexes to find the one for our field
            index_names = self.client.list_indexes(collection_name=collection_name)
            logging.debug(f"Available indexes in '{collection_name}': {index_names}")

            # Try to find an index that corresponds to our field
            # Check multiple possible naming patterns
            possible_names = [
                field_name,  # Direct field name
                f"{field_name}_index",  # Field name with suffix
                f"_{field_name}",  # Field name with prefix
            ]

            index_name = None
            # First try direct name matches
            for name in possible_names:
                if name in index_names:
                    index_name = name
                    break

            # If no direct match, examine each index to find the one for our field
            if not index_name:
                for idx_name in index_names:
                    try:
                        idx_info = self.client.describe_index(
                            collection_name=collection_name, index_name=idx_name
                        )
                        if idx_info.get("field_name") == field_name:
                            index_name = idx_name
                            logging.debug(f"Found index '{idx_name}' for field '{field_name}'")
                            break
                    except Exception as e:
                        logging.debug(f"Failed to describe index '{idx_name}': {e}")
                        continue

            if index_name:
                # Get detailed index information
                index_info = self.client.describe_index(
                    collection_name=collection_name, index_name=index_name
                )
                logging.debug(f"Index info for '{field_name}': {index_info}")
                return index_info
            else:
                logging.debug(
                    f"No index found for field '{field_name}' in collection '{collection_name}'. "
                    f"Available indexes: {index_names}"
                )
                return {}

        except Exception as e:
            logging.warning(f"Failed to get index status for field '{field_name}': {e}")
            return {}

    def is_index_ready(self, collection_name: str, field_name: str) -> bool:
        """Check if index is ready for a specific field."""
        try:
            index_info = self.get_index_status(collection_name, field_name)
            if not index_info:
                return False

            # According to the documentation, check these fields:
            total_rows = index_info.get("total_rows", 0)
            indexed_rows = index_info.get("indexed_rows", 0)
            pending_index_rows = index_info.get("pending_index_rows", 0)
            state = index_info.get("state", 0)

            # Index is ready if:
            # 1. All rows are indexed (indexed_rows == total_rows)
            # 2. No pending rows (pending_index_rows == 0)
            # 3. Total rows > 0 (collection has data)
            if total_rows > 0:
                is_ready = (indexed_rows == total_rows) and (pending_index_rows == 0)
            else:
                # If no data yet, consider index ready for now
                is_ready = True

            logging.debug(
                f"Index for field '{field_name}': total={total_rows}, indexed={indexed_rows}, "
                f"pending={pending_index_rows}, state={state}, ready={is_ready}"
            )
            return is_ready

        except Exception as e:
            logging.warning(f"Failed to check index readiness for field '{field_name}': {e}")
            return False

    def wait_for_indexes_ready(
        self, collection_name: str, timeout: int = 300, check_interval: int = 5
    ) -> bool:
        """Wait for all indexes to be ready."""
        logging.info(f"Waiting for indexes to be ready for collection '{collection_name}'...")

        # Enable debug logging temporarily for index checking
        original_level = logging.getLogger().level
        if original_level > logging.DEBUG:
            logging.getLogger().setLevel(logging.DEBUG)

        start_time = time.time()
        fields_to_check = ["embedding", "location"]  # Vector and geo fields

        while time.time() - start_time < timeout:
            all_ready = True
            status_messages = []

            for field_name in fields_to_check:
                is_ready = self.is_index_ready(collection_name, field_name)
                index_info = self.get_index_status(collection_name, field_name)

                if index_info:
                    total_rows = index_info.get("total_rows", 0)
                    indexed_rows = index_info.get("indexed_rows", 0)
                    pending_rows = index_info.get("pending_index_rows", 0)
                    progress_pct = (indexed_rows / total_rows * 100) if total_rows > 0 else 100

                    status = f"{field_name}: {progress_pct:.1f}% ({indexed_rows}/{total_rows})"
                    if pending_rows > 0:
                        status += f", pending: {pending_rows}"
                else:
                    status = f"{field_name}: No index"

                status_messages.append(status)

                if not is_ready:
                    all_ready = False

            logging.info(f"Index build progress - {', '.join(status_messages)}")

            if all_ready:
                logging.info(f"All indexes are ready for collection '{collection_name}'")
                # Restore original logging level
                logging.getLogger().setLevel(original_level)
                return True

            logging.debug(f"Waiting for indexes... ({time.time() - start_time:.1f}s elapsed)")
            time.sleep(check_interval)

        # Timeout reached
        elapsed = time.time() - start_time
        logging.warning(f"Timeout waiting for indexes after {elapsed:.1f}s")

        # Restore original logging level
        logging.getLogger().setLevel(original_level)
        return False

    def health_check(self) -> bool:
        """Check if Milvus server is healthy."""
        try:
            # Try to list collections as a health check
            collections = self.client.list_collections()
            logging.info(f"Health check passed. Found {len(collections)} collections.")
            return True
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False

    def close(self) -> None:
        """Close connection to Milvus."""
        if self.client:
            try:
                self.client.close()
                logging.info("Closed Milvus connection")
            except Exception as e:
                logging.warning(f"Error closing Milvus connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
