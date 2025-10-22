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

    # Already an array or list
    if isinstance(value, (np.ndarray, list)):
        # Skip empty arrays
        if len(value) == 0:
            return None
        return value

    # Scalar value - wrap in list
    return [value]


def expand_array_columns_vertically(df):
    """Expands all columns containing arrays vertically, creating new rows for each array element."""
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

    expanded_data = []
    for idx, row in df.iterrows():
        array_length = len(row[array_columns[0]])
        for i in range(array_length):
            new_row = {}
            for col in non_array_columns:
                new_row[col] = row[col]
            for col in array_columns:
                new_row[col] = row[col][i]
            expanded_data.append(new_row)

    result_df = pd.DataFrame(expanded_data)
    logger.debug(f"Expanded shape from {df.shape} to {result_df.shape}")
    return result_df


def parse_ros_message_definition(definition: str | bytes) -> dict[str, Any]:
    """Parse a ROS message definition into a dictionary describing the message structure."""
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

    message_spec: dict[str, dict] = {}
    sections = definition_str.split(
        "================================================================================"
    )
    main_section = sections[0].strip()

    for line in main_section.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 2:
            field_type, field_name = parts[0], parts[1]
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

            if "/" in field_type:
                type_name = field_type.split("/")[-1]
                for section in sections[1:]:
                    if f"MSG: {field_type}" in section:
                        nested_fields = parse_ros_message_definition(section)
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
