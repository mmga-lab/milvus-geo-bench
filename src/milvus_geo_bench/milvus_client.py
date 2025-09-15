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
        """Create collection with geo and vector fields (without indexes)."""
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

            logging.info(f"Created collection: {collection_name} (without indexes)")

        except Exception as e:
            logging.error(f"Failed to create collection {collection_name}: {e}")
            raise

    def create_vector_index(self, collection_name: str) -> float:
        """Create vector index and return creation time in seconds."""
        try:
            start_time = time.time()

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

            end_time = time.time()
            creation_time = end_time - start_time

            logging.info(f"Created vector index for {collection_name} in {creation_time:.2f}s")
            return creation_time

        except Exception as e:
            logging.error(f"Failed to create vector index for {collection_name}: {e}")
            raise

    def drop_vector_index(self, collection_name: str) -> float:
        """Drop vector index and return operation time in seconds."""
        try:
            start_time = time.time()

            # First check if vector index exists
            if not self._has_vector_index(collection_name):
                logging.info(f"No vector index found for {collection_name}, nothing to drop")
                return 0.0

            # Find the vector index name
            vector_index_name = self._find_vector_index_name(collection_name)
            if not vector_index_name:
                logging.warning(f"Could not find vector index name for {collection_name}")
                return 0.0

            # Release collection before dropping index
            logging.info(f"Releasing collection {collection_name} before dropping vector index")
            self.client.release_collection(collection_name)

            # Drop the vector index
            self.client.drop_index(
                collection_name=collection_name,
                index_name=vector_index_name
            )

            end_time = time.time()
            drop_time = end_time - start_time

            logging.info(f"Dropped vector index '{vector_index_name}' for {collection_name} in {drop_time:.2f}s")
            return drop_time

        except Exception as e:
            logging.error(f"Failed to drop vector index for {collection_name}: {e}")
            raise

    def _has_vector_index(self, collection_name: str) -> bool:
        """Check if collection has a vector index."""
        try:
            index_info = self.get_index_status(collection_name, "embedding")
            return bool(index_info)
        except Exception:
            return False

    def _find_vector_index_name(self, collection_name: str) -> str | None:
        """Find the name of the vector index for the embedding field."""
        try:
            index_names = self.client.list_indexes(collection_name=collection_name)

            # Try to find the index for the embedding field
            for idx_name in index_names:
                try:
                    idx_info = self.client.describe_index(
                        collection_name=collection_name, index_name=idx_name
                    )
                    if idx_info.get("field_name") == "embedding":
                        return idx_name
                except Exception:
                    continue

            return None
        except Exception:
            return None

    def create_geo_index(self, collection_name: str) -> float:
        """Create geometry index and return creation time in seconds."""
        try:
            start_time = time.time()

            # Create geometry index (RTREE for spatial data)
            geo_index_params = IndexParams()
            geo_index_params.add_index(
                field_name="location",
                index_type="RTREE",
            )

            self.client.create_index(
                collection_name=collection_name,
                index_params=geo_index_params,
            )

            end_time = time.time()
            creation_time = end_time - start_time

            logging.info(f"Created RTREE geometry index for {collection_name} in {creation_time:.2f}s")
            return creation_time

        except Exception as e:
            logging.error(f"Failed to create geometry index for {collection_name}: {e}")
            raise

    def drop_geo_index(self, collection_name: str) -> float:
        """Drop geometry index and return operation time in seconds."""
        try:
            start_time = time.time()

            # First check if geo index exists
            if not self._has_geo_index(collection_name):
                logging.info(f"No geo index found for {collection_name}, nothing to drop")
                return 0.0

            # Find the geo index name
            geo_index_name = self._find_geo_index_name(collection_name)
            if not geo_index_name:
                logging.warning(f"Could not find geo index name for {collection_name}")
                return 0.0

            # Release collection before dropping index
            logging.info(f"Releasing collection {collection_name} before dropping geo index")
            self.client.release_collection(collection_name)

            # Drop the geo index
            self.client.drop_index(
                collection_name=collection_name,
                index_name=geo_index_name
            )

            end_time = time.time()
            drop_time = end_time - start_time

            logging.info(f"Dropped geo index '{geo_index_name}' for {collection_name} in {drop_time:.2f}s")
            return drop_time

        except Exception as e:
            logging.error(f"Failed to drop geo index for {collection_name}: {e}")
            raise

    def _has_geo_index(self, collection_name: str) -> bool:
        """Check if collection has a geo index."""
        try:
            index_info = self.get_index_status(collection_name, "location")
            return bool(index_info)
        except Exception:
            return False

    def _find_geo_index_name(self, collection_name: str) -> str | None:
        """Find the name of the geo index for the location field."""
        try:
            index_names = self.client.list_indexes(collection_name=collection_name)

            # Try to find the index for the location field
            for idx_name in index_names:
                try:
                    idx_info = self.client.describe_index(
                        collection_name=collection_name, index_name=idx_name
                    )
                    if idx_info.get("field_name") == "location":
                        return idx_name
                except Exception:
                    continue

            return None
        except Exception:
            return None

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
            logging.info(f"Data insertion and flush completed for {collection_name}")

        except Exception as e:
            logging.error(f"Failed to insert data: {e}")
            raise

    def load_collection(self, collection_name: str) -> float:
        """Load collection to memory and return loading time in seconds."""
        try:
            start_time = time.time()

            self.client.load_collection(collection_name)

            end_time = time.time()
            loading_time = end_time - start_time

            logging.info(f"Loaded collection {collection_name} to memory in {loading_time:.2f}s")
            return loading_time

        except Exception as e:
            logging.error(f"Failed to load collection {collection_name}: {e}")
            raise

    def release_collection(self, collection_name: str) -> float:
        """Release collection from memory and return operation time in seconds."""
        try:
            start_time = time.time()

            self.client.release_collection(collection_name)

            end_time = time.time()
            release_time = end_time - start_time

            logging.info(f"Released collection {collection_name} from memory in {release_time:.2f}s")
            return release_time

        except Exception as e:
            logging.error(f"Failed to release collection {collection_name}: {e}")
            raise

    def is_collection_loaded(self, collection_name: str) -> bool:
        """Check if collection is currently loaded in memory."""
        try:
            # Try to get collection load state
            # Note: This is a simple check - in practice, you might need to use
            # client.describe_collection or check collection stats
            self.client.get_collection_stats(collection_name)
            # If we can get stats, assume it's loaded (this is a simplification)
            return True
        except Exception:
            return False

    def release_other_collections(self, target_collection: str) -> dict[str, float]:
        """Release all collections except the target collection and return operation times."""
        release_times = {}

        try:
            # Get list of all collections
            all_collections = self.client.list_collections()
            logging.debug(f"Found {len(all_collections)} collections: {all_collections}")

            # Filter out the target collection
            other_collections = [col for col in all_collections if col != target_collection]

            if not other_collections:
                logging.info(f"No other collections to release (only '{target_collection}' exists)")
                return release_times

            logging.info(f"Releasing {len(other_collections)} other collections to free up memory")

            for collection_name in other_collections:
                try:
                    # Check if collection is loaded before trying to release
                    if self.is_collection_loaded(collection_name):
                        start_time = time.time()
                        self.client.release_collection(collection_name)
                        end_time = time.time()
                        release_time = end_time - start_time
                        release_times[collection_name] = release_time
                        logging.debug(f"Released collection '{collection_name}' in {release_time:.2f}s")
                    else:
                        logging.debug(f"Collection '{collection_name}' is not loaded, skipping release")
                        release_times[collection_name] = 0.0
                except Exception as e:
                    logging.warning(f"Failed to release collection '{collection_name}': {e}")
                    release_times[collection_name] = -1.0  # Indicate failure

            total_released = len([t for t in release_times.values() if t > 0])
            total_time = sum(t for t in release_times.values() if t > 0)
            logging.info(f"Successfully released {total_released} collections in {total_time:.2f}s total")

            return release_times

        except Exception as e:
            logging.error(f"Failed to release other collections: {e}")
            raise

    def get_loaded_collections(self) -> list[str]:
        """Get list of currently loaded collections."""
        try:
            all_collections = self.client.list_collections()
            loaded_collections = []

            for collection_name in all_collections:
                if self.is_collection_loaded(collection_name):
                    loaded_collections.append(collection_name)

            return loaded_collections
        except Exception as e:
            logging.warning(f"Failed to get loaded collections: {e}")
            return []

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
        self, collection_name: str, timeout: int = 300, check_interval: int = 5, fields: list[str] | None = None
    ) -> bool:
        """Wait for specified indexes to be ready."""
        if fields is None:
            fields = ["embedding", "location"]  # Default to all fields

        logging.info(f"Waiting for indexes to be ready for collection '{collection_name}' (fields: {fields})...")

        # Enable debug logging temporarily for index checking
        original_level = logging.getLogger().level
        if original_level > logging.DEBUG:
            logging.getLogger().setLevel(logging.DEBUG)

        start_time = time.time()
        fields_to_check = fields  # Use specified fields

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
