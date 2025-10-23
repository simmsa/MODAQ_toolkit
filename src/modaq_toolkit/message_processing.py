import logging
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_array_column(series):
    """Check if a pandas series contains numpy arrays"""
    if series.dtype == "object":
        first_val = series.dropna().iloc[0] if not series.isna().all() else None
        return isinstance(first_val, (np.ndarray, list))
    return False


def _normalize_to_array(value):
    """Normalize a value to an array, handling scalars and empty arrays.

    Returns None for values that should be skipped (empty arrays, None, NaN).
    Returns a list/array for valid values.
    """
    # Handle None/NaN
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # Handle numpy scalars (from type conversion) - extract Python value
    if isinstance(value, np.generic) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    ):
        return [value.item()]

    # Already an array or list
    if isinstance(value, (np.ndarray, list)):
        # Skip empty arrays
        if len(value) == 0:
            return None
        return value

    # Scalar value - wrap in list
    return [value]


def expand_array_columns_vertically(df):
    """Expands all columns containing arrays vertically, creating new rows for each array element.

    Preservation strategy:
    - Converts scalars to [scalar]
    - Keeps non-empty arrays as-is
    - Skips empty arrays []
    - Concatenates all values vertically across all rows
    - One observation per cell (no nested arrays)
    """
    # Early exit for empty DataFrame
    if df.empty:
        return df

    array_columns = []
    non_array_columns = []
    for col in df.columns:
        if is_array_column(df[col]):
            array_columns.append(col)
        else:
            non_array_columns.append(col)

    logger.debug(f"Found array columns: {array_columns}")
    if not array_columns:
        return df

    # Collect all array values across all rows for each array column
    # This implements the concatenation strategy: [a, b], [], [c] -> [a, b, c]
    all_arrays = {col: [] for col in array_columns}
    non_array_values = {col: [] for col in non_array_columns}

    for idx, row in df.iterrows():
        # Normalize each array column value
        normalized_arrays = {}
        for col in array_columns:
            normalized = _normalize_to_array(row[col])
            normalized_arrays[col] = normalized

        # Check if all array columns are None (should skip this row entirely)
        if all(arr is None for arr in normalized_arrays.values()):
            logger.debug(f"Skipping row {idx}: all array columns are empty")
            continue

        # Get the maximum length among non-None arrays in this row
        lengths = [len(arr) for arr in normalized_arrays.values() if arr is not None]
        if not lengths:
            continue

        max_length = max(lengths)

        # Validate: all non-None arrays should have the same length
        # This is a data integrity check - inconsistent array lengths indicate
        # corrupted data or transmission errors that must be investigated
        if not all(length == max_length for length in lengths):
            raise ValueError(
                f"Data integrity error at row {idx}: Array columns have inconsistent lengths {lengths}. "
                f"All nested arrays in a ROS message must have the same length. "
            )

        # Concatenate this row's arrays to the master list
        for col in array_columns:
            if normalized_arrays[col] is not None:
                all_arrays[col].extend(normalized_arrays[col])
            else:
                # If one column is None but others aren't, pad with NaN
                all_arrays[col].extend([np.nan] * max_length)

        # Repeat non-array values to match array length
        for col in non_array_columns:
            non_array_values[col].extend([row[col]] * max_length)

    # Build the result DataFrame
    result_data = {}
    for col in non_array_columns:
        result_data[col] = non_array_values[col]
    for col in array_columns:
        result_data[col] = all_arrays[col]

    result_df = pd.DataFrame(result_data)
    logger.info(f"Expanded shape from {df.shape} to {result_df.shape}")
    return result_df


def parse_ros_message_definition(
    definition: str | bytes, _all_sections: list[str] | None = None
) -> dict[str, Any]:
    """Parse a ROS message definition into a dictionary describing the message structure.

    Args:
        definition: The ROS message definition string or bytes
        _all_sections: Internal parameter - full list of all schema sections for recursive lookups

    Returns:
        Dictionary mapping field names to their specifications
    """
    if isinstance(definition, bytes):
        try:
            definition_str = definition.decode("utf-8")
        except UnicodeDecodeError:
            try:
                definition_str = definition.decode("ascii")
            except UnicodeDecodeError:
                definition_str = definition.decode("latin-1")
    else:
        definition_str = definition

    # ROS message constant definitions to skip (not actual message fields)
    ROS_CONSTANTS = {
        "DEBUG=10",
        "INFO=20",
        "WARN=30",
        "ERROR=40",
        "FATAL=50",
    }

    message_spec: dict[str, dict] = {}

    # Split into sections on first call, or use existing sections
    if _all_sections is None:
        sections = definition_str.split(
            "================================================================================"
        )
        _all_sections = sections  # Preserve all sections for recursive calls
    else:
        sections = [definition_str]  # For recursive calls, only parse main section

    main_section = sections[0].strip()

    for line in main_section.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 2:
            field_type, field_name = parts[0], parts[1]

            # Skip known ROS logging level constants
            if field_name in ROS_CONSTANTS:
                logger.debug(f"Skipping ROS constant: {field_name}")
                continue

            is_array = field_type.endswith("[]")
            if is_array:
                field_type = field_type[:-2]

            default_value = None
            if len(parts) >= 3:
                try:
                    raw_default = parts[2].strip('"')
                    if field_type == "bool":
                        default_value = raw_default.lower() == "true"
                    elif field_type == "string":
                        default_value = raw_default
                    elif field_type.startswith("float"):
                        default_value = float(raw_default)
                    elif field_type.startswith("int"):
                        default_value = int(raw_default)
                except:
                    pass

            message_spec[field_name] = {
                "type": field_type,
                "is_array": is_array,
                "default": default_value,
            }

            # For nested types (e.g., std_msgs/Header, builtin_interfaces/Time),
            # search ALL available sections to find the type definition
            if "/" in field_type:
                type_name = field_type.split("/")[-1]
                for section in _all_sections[
                    1:
                ]:  # Use all sections, not just local ones
                    if f"MSG: {field_type}" in section:
                        # Pass all sections to recursive call so deeply nested types can be found
                        nested_fields = parse_ros_message_definition(
                            section, _all_sections
                        )
                        message_spec[field_name]["fields"] = nested_fields
                        break

    return message_spec


class MessageProcessor:
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.messages: list[dict[str, Any]] = []

        self.ros_type_to_numpy_type_map = {
            # "float32": np.float32,
            "float64": np.float64,
            # "int8": np.int8,
            # "int16": np.int16,
            # "int32": np.int32,
            # "int64": np.int64,
            # "uint8": np.uint8,
            # "uint16": np.uint16,
            # "uint32": np.uint32,
            "uint64": np.uint64,
        }

    def process_message(self, msg: Any) -> None:
        message_dict = {}
        # print(self.schema)
        # Typical entry
        # data = {
        #     "header": {
        #         "type": "std_msgs/Header",
        #         "is_array": False,
        #         "default": None,
        #         "fields": {
        #             "std_msgs/Header": {
        #                 "type": "MSG:",
        #                 "is_array": False,
        #                 "default": None,
        #             },
        #             "stamp": {
        #                 "type": "builtin_interfaces/Time",
        #                 "is_array": False,
        #                 "default": None,
        #             },
        #             "frame_id": {
        #                 "type": "string",
        #                 "is_array": False,
        #                 "default": None,
        #             },
        #         },
        #     },
        #     "ain0": {"type": "float64", "is_array": True, "default": None},
        #     "ain1": {"type": "float64", "is_array": True, "default": None},
        #     "ain2": {"type": "float64", "is_array": True, "default": None},
        #     "ain3": {"type": "float64", "is_array": True, "default": None},
        #     "ain4": {"type": "float64", "is_array": True, "default": None},
        #     "ain5": {"type": "float64", "is_array": True, "default": None},
        #     "ain6": {"type": "float64", "is_array": True, "default": None},
        #     "ain7": {"type": "float64", "is_array": True, "default": None},
        #     "core_timer": {"type": "uint64", "is_array": True, "default": None},
        #     "system_time": {"type": "uint64", "is_array": True, "default": None},
        # }
        if "header" in self.schema:
            message_dict["sec"] = msg.header.stamp.sec
            message_dict["nanosec"] = msg.header.stamp.nanosec
            message_dict["frame_id"] = msg.header.frame_id

        for field_name, field_spec in self.schema.items():
            if field_name == "header":
                continue
            try:
                value = getattr(msg, field_name)

                # Get the type from field_spec
                field_type = field_spec.get("type")

                # Try to convert to numpy type if applicable
                if field_type in self.ros_type_to_numpy_type_map:
                    numpy_dtype = self.ros_type_to_numpy_type_map[field_type]
                    print(
                        "Converting field:", field_name, "to numpy dtype:", numpy_dtype
                    )
                    message_dict[field_name] = np.array(value, dtype=numpy_dtype)
                else:
                    message_dict[field_name] = value

            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Failed to get field '{field_name}': {e}")

        self.messages.append(message_dict)

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.messages)
        if "sec" in df.columns and "nanosec" in df.columns:
            df["timestamp"] = df["sec"] + df["nanosec"] * 1e-9

        if "stamp" in df.columns:
            df["sec"] = df["stamp"].apply(lambda t: t.sec)
            df["nanosec"] = df["stamp"].apply(lambda t: t.nanosec)

            df = df.drop(columns=["stamp"])

            df["timestamp"] = df["sec"] + df["nanosec"] * 1e-9

        return df
