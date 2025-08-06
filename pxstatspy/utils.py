from typing import Dict, List, Optional, Union

import pandas as pd
from .pxapi import OutputFormatParam, PxVariables


def print_table_variables(
    metadata: Dict,
    max_values: Union[int, str] = 10,
    variable_id: Optional[str] = None,
):
    """
    Print available variables and their values for a specific table

    Args:
        metadata: Metadata dictionary (json-stat2 format)
        max_values: Maximum number of values to show for each variable.
                Use "*" to show all values. (default: 10)
        variable_id: Optional variable ID to show values for only that variable.
                If None, shows all variables. (default: None)
    """
    # metadata = self.get_table_metadata(table_id, output_format="json-stat2")
    print(f"\nTable: {metadata['label']}")

    show_all = max_values == "*"
    limit = None if show_all else int(max_values)

    # Get variables from JSON-stat2 format
    variables = []
    for var_id in metadata["id"]:
        dimension = metadata["dimension"][var_id]

        # Create a variable entry with the same structure expected by the function
        var = {"id": var_id, "label": dimension.get("label", var_id), "values": []}

        # Extract values from category
        if "category" in dimension and "index" in dimension["category"]:
            index = dimension["category"]["index"]
            labels = dimension["category"].get("label", {})

            for code in index.keys():
                var["values"].append({"code": code, "label": labels.get(code, code)})

        variables.append(var)

    if variable_id:
        # Find the specified variable
        var = next((v for v in variables if v["id"] == variable_id), None)
        if not var:
            raise ValueError(f"Variable '{variable_id}' not found in table")

        print(f"\nValues for {var['label']} ({var['id']}):")
        if show_all:
            print("All values:")
        else:
            print(f"First {limit} values:")

        values_to_show = var["values"] if show_all else var["values"][:limit]
        for val in values_to_show:
            print(f"  - {val['label']} (Code: {val['code']})")

        total_values = len(var["values"])
        if not show_all and total_values > limit:  # type: ignore
            print(f"\n... and {total_values - limit} more values")  # type: ignore
    else:
        # Show all variables
        print("\nAvailable variables:")
        for var in variables:
            print(f"\n{var['label']} ({var['id']}):")

            if show_all:
                print("All values:")
            else:
                print(f"First {limit} values:")

            values_to_show = var["values"] if show_all else var["values"][:limit]
            for val in values_to_show:
                print(f"  - {val['label']} (Code: {val['code']})")

            total_values = len(var["values"])
            if not show_all and total_values > limit:  # type: ignore
                print(f"... and {total_values - limit} more values")  # type: ignore


def process_jsonstat_to_df(
    data: Dict, output_format_param: OutputFormatParam, clean_colnames: bool
) -> pd.DataFrame:
    """Process JSON-stat data into DataFrame

    Args:
        data: JSON-stat 2 data dictionary
        output_format_param: Output format parameter for how to handle texts/codes
        clean_colnames: Whether to clean column names (lowercase, underscores, etc.)
    Returns:
        DataFrame with processed data
    """

    dimensions = data["dimension"]
    values = data["value"]

    # Find the ContentsVariable to get its values
    contents_var_id = next(
        dim_id for dim_id in data["id"] if PxVariables.CONTENTS in dim_id
    )
    contents_dim = dimensions[contents_var_id]
    contents_mapping = {
        code: label for code, label in contents_dim["category"]["label"].items()
    }

    # Create a list to store rows
    rows = []

    # Get dimension sizes and names
    sizes = data["size"]
    dim_names = data["id"]

    # Create a helper function to convert flat index to dimensional indices
    def flat_to_indices(flat_idx, sizes):
        indices = []
        for size in reversed(sizes):
            indices.append(flat_idx % size)
            flat_idx //= size
        return list(reversed(indices))

    # Find the content dimension index
    content_dim_id = next(
        i for i, dim in enumerate(dim_names) if PxVariables.CONTENTS in dim
    )

    # Create a dictionary to temporarily store values
    temp_data = {}

    # Process each value
    for i, value in enumerate(values):
        # Get indices for each dimension
        dim_indices = flat_to_indices(i, sizes)

        # Create key for the row (excluding content dimension)
        row_key = []
        content_code = None

        for dim_name, idx in zip(dim_names, dim_indices):
            dim = dimensions[dim_name]
            code = list(dim["category"]["index"].keys())[idx]

            if PxVariables.CONTENTS in dim_name:
                content_code = code
                continue

            if dim_name == PxVariables.REGION:
                label = dim["category"]["label"][code]
                row_key.extend([code, label])
            else:
                if dim_name == PxVariables.TIME:
                    # For time dimension, just use the code/label without combining
                    label = (
                        dim["category"]["label"][code]
                        if output_format_param == OutputFormatParam.USE_TEXTS
                        else code
                    )
                    row_key.append(label)
                else:
                    # For other dimensions
                    if output_format_param == OutputFormatParam.USE_CODES_AND_TEXTS:
                        label = dim["category"]["label"][code]
                        row_key.append(f"{code} - {label}")
                    else:
                        label = (
                            dim["category"]["label"][code]
                            if output_format_param == OutputFormatParam.USE_TEXTS
                            else code
                        )
                        row_key.append(label)

        row_key = tuple(row_key)

        # Initialize row if not exists
        if row_key not in temp_data:
            temp_data[row_key] = {}

        # Add value to appropriate content column
        temp_data[row_key][content_code] = value

    # Convert to DataFrame
    rows = []
    for row_key, row_values in temp_data.items():
        if PxVariables.REGION in dim_names:
            # Assuming region is always first when present
            region_code, region_label = row_key[:2]
            other_dims = row_key[2:]
            row_dict = {"region_code": region_code, "region": region_label}
        else:
            other_dims = row_key
            row_dict = {}

        # Add other dimensions
        other_dim_names = [
            dim
            for dim in dim_names
            if dim != PxVariables.REGION and PxVariables.CONTENTS not in dim
        ]
        for dim_name, value in zip(other_dim_names, other_dims):
            clean_name = dimensions[dim_name]["label"]
            row_dict[clean_name] = value

        # Add content values
        row_dict.update(row_values)
        rows.append(row_dict)

    df = pd.DataFrame(rows)

    # Rename content columns to their labels if using texts
    if output_format_param in [
        OutputFormatParam.USE_TEXTS,
        OutputFormatParam.USE_CODES_AND_TEXTS,
    ]:
        df = df.rename(columns=contents_mapping)

    # Clean column names if requested
    if clean_colnames:

        def clean_column_name(col):
            # Convert to lowercase
            col = col.lower()
            # Replace spaces with underscores
            col = col.replace(" ", "_")
            # Replace Swedish characters
            col = col.replace("å", "a").replace("ä", "a").replace("ö", "o")
            # Remove special characters (keeping underscores)
            col = "".join(c for c in col if c.isalnum() or c == "_")
            # Remove multiple consecutive underscores
            col = "_".join(filter(None, col.split("_")))
            return col

        df.columns = [clean_column_name(col) for col in df.columns]

    return df
