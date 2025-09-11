"""
Milvus client wrapper for geo benchmark tool.
"""

import logging
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
    from pymilvus.exceptions import MilvusException
    from pymilvus.milvus_client import IndexParams
except ImportError:
    raise ImportError("pymilvus is required. Install it with: uv add pymilvus")


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

            # Load collection to memory
            self.client.load_collection(collection_name)
            logging.info(f"Loaded collection {collection_name} to memory")

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
