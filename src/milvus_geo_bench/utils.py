"""
Utility functions for milvus geo benchmark tool.
"""

import logging
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
import yaml


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution."""
    load_dotenv()  # Load environment variables from .env file

    if config_path and Path(config_path).exists():
        config_file_path = Path(config_path)
        with config_file_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Replace environment variables in config
        config = _substitute_env_vars(config)
        return config

    # Return default config if no file specified
    return get_default_config()


def get_default_config() -> dict[str, Any]:
    """Get default configuration."""
    return {
        "dataset": {
            "num_points": 100000,
            "num_queries": 1000,
            "output_dir": "./data",
            "bbox": [-180, -90, 180, 90],
            "min_points_per_query": 100,
            "max_radius": 1.0,
            "grid": {
                "enabled": False,
                "auto_calculate": True,
                "target_points_per_grid": 1000000,
                "min_grids": 1,
                "max_grids": 1000,
            },
        },
        "milvus": {
            "uri": os.getenv("MILVUS_URI", ""),
            "token": os.getenv("MILVUS_TOKEN", ""),
            "collection": "geo_bench",
            "batch_size": 1000,
            "timeout": 30,
        },
        "benchmark": {"timeout": 30, "warmup": 10, "concurrency": 1},
        "output": {"results": "./data/results.parquet", "report": "./reports/report.md"},
    }


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute environment variables in configuration."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Replace ${VAR_NAME} pattern with environment variable
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, obj)
        for var_name in matches:
            var_value = os.getenv(var_name, "")
            obj = obj.replace(f"${{{var_name}}}", var_value)
        return obj
    else:
        return obj


def save_parquet(df: pd.DataFrame, file_path: str | Path) -> None:
    """Save DataFrame to Parquet file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)
    logging.info(f"Saved {len(df)} rows to {file_path}")


def load_parquet(file_path: str | Path) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_parquet(file_path)
    logging.info(f"Loaded {len(df)} rows from {file_path}")
    return df


def load_train_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load training data from either a single parquet file or a directory of parquet files.

    Args:
        data_path: Path to a parquet file or directory containing train parquet files

    Returns:
        Combined DataFrame with all training data
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if data_path.is_file():
        # Single parquet file
        logging.info(f"Loading single training file: {data_path}")
        return load_parquet(data_path)

    elif data_path.is_dir():
        # Directory containing multiple parquet files
        # Look for train_*.parquet files (shuffled naming) or train.parquet
        train_files = sorted(data_path.glob("train_*.parquet"))

        if not train_files:
            # Fallback to train.parquet or train_grid_*.parquet
            train_files = list(data_path.glob("train.parquet"))
            if not train_files:
                train_files = sorted(data_path.glob("train_grid_*.parquet"))

        if not train_files:
            raise FileNotFoundError(f"No training parquet files found in directory: {data_path}")

        logging.info(f"Loading {len(train_files)} training files from directory: {data_path}")

        # Load and combine all files
        dfs = []
        for train_file in train_files:
            logging.debug(f"Loading file: {train_file.name}")
            df = load_parquet(train_file)
            dfs.append(df)

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        logging.info(
            f"Loaded {len(combined_df)} total training records from {len(train_files)} files"
        )

        return combined_df

    else:
        raise ValueError(f"Data path is neither file nor directory: {data_path}")


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_wkt_point(wkt: str) -> bool:
    """Validate WKT Point format."""
    pattern = r"^POINT\s*\(\s*-?\d+\.?\d*\s+-?\d+\.?\d*\s*\)$"
    return bool(re.match(pattern, wkt.strip(), re.IGNORECASE))


def validate_wkt_polygon(wkt: str) -> bool:
    """Validate WKT Polygon format."""
    pattern = r"^POLYGON\s*\(\s*\([^)]+\)\s*\)$"
    return bool(re.match(pattern, wkt.strip(), re.IGNORECASE))


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def print_table(data: list, headers: list) -> None:
    """Print data as formatted table."""
    print(tabulate(data, headers=headers, tablefmt="grid"))
