"""Schema loading and normalization utilities for the solar tracker environment."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from data import DataSet
from utilities import read_json


class UnknownSchemaError(Exception):
    """Raised when schema is not a known dataset name, mapping, or JSON file path."""


def load_schema(schema: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    """Return normalized schema as a dictionary with an absolute ``root_directory``."""

    schema_path = None

    if isinstance(schema, (str, Path)):
        candidate = Path(schema)

        if candidate.is_file():
            schema_path = candidate.resolve()
        else:
            # Common notebook case: relative path is provided from a different CWD.
            package_root = Path(__file__).resolve().parent
            candidate_from_project_root = (package_root / candidate).resolve()

            if candidate_from_project_root.is_file():
                schema_path = candidate_from_project_root

    if schema_path is not None:
        data = read_json(str(schema_path))
        data["root_directory"] = _resolve_root_directory(data, schema_path)
        return data

    if isinstance(schema, str) and schema in DataSet.get_names():
        return DataSet.get_schema(schema)

    if isinstance(schema, Mapping):
        data = deepcopy(dict(schema))
        data["root_directory"] = "" if data.get("root_directory") is None else data["root_directory"]
        return data

    raise UnknownSchemaError(
        "Unknown schema. Provide a dataset name in st/data, a schema mapping, or a JSON schema filepath."
    )


def _resolve_root_directory(data: Dict[str, Any], schema_path: Path) -> str:
    """Resolve the CSV directory for a schema file after the project is moved."""

    data_filename = data.get("data", {}).get("filename")
    configured_root = data.get("root_directory")
    candidates = []

    if configured_root not in (None, ""):
        configured_path = Path(configured_root)
        candidates.append(configured_path if configured_path.is_absolute() else schema_path.parent / configured_path)

    candidates.extend(
        [
            schema_path.parent,
            Path(__file__).resolve().parent / "data",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()

        if data_filename is None or (candidate / data_filename).is_file():
            return str(candidate)

    return str(schema_path.parent.resolve())
