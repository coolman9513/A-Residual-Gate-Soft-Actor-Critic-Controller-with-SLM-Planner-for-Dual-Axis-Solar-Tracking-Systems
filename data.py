"""Dataset utilities for solar-tracker simulation assets in ``st/data``."""

import os
from pathlib import Path
import shutil
from typing import List, Union

from utilities import read_json


class DataSet:
    """Utilities to discover and copy dataset folders under ``st/data``."""

    __ROOT_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")

    @staticmethod
    def get_names() -> List[str]:
        return sorted(
            [
                d
                for d in os.listdir(DataSet.__ROOT_DIRECTORY)
                if os.path.isdir(os.path.join(DataSet.__ROOT_DIRECTORY, d))
            ]
        )

    @staticmethod
    def copy(name: str, destination_directory: Union[Path, str] = None):
        source_directory = os.path.join(DataSet.__ROOT_DIRECTORY, name)
        destination_directory = "" if destination_directory is None else str(destination_directory)
        destination_directory = os.path.join(destination_directory, name)
        os.makedirs(destination_directory, exist_ok=True)

        for filename in os.listdir(source_directory):
            if not (filename.endswith(".csv") or filename.endswith(".json")):
                continue

            source_filepath = os.path.join(source_directory, filename)
            destination_filepath = os.path.join(destination_directory, filename)
            shutil.copy(source_filepath, destination_filepath)

    @staticmethod
    def get_schema(name: str):
        root_directory = os.path.join(DataSet.__ROOT_DIRECTORY, name)
        filepath = os.path.join(root_directory, "schema.json")
        schema = read_json(filepath)
        schema["root_directory"] = root_directory
        return schema
