import polars as pl
from typing import Dict, Any, Optional


class MarginaleffectsDataFrame(pl.DataFrame):
    def __init__(
        self,
        data=None,
        by=None,
        conf_level=0.95,
        jacobian=None,
        newdata=None,
        mapping=None,
        print_head="",
    ):
        if isinstance(data, pl.DataFrame):
            self._df = data._df
            self.by = by
            self.conf_level = conf_level
            self.jacobian = jacobian
            if hasattr(newdata, "datagrid_explicit"):
                self.datagrid_explicit = newdata.datagrid_explicit
            else:
                self.datagrid_explicit = []

            self.print_head = print_head

            # Split the dictionary into two parts and combine them into default_mapping.
            # The first part only includes "term" and any column from `data` that start with "contrast".
            # Any contrast key that starts with contrast_ should have a value in the form: "C: v", where v is the part of the key after the underscore.
            de = dict(zip(self.datagrid_explicit, self.datagrid_explicit))
            contrast_columns = {
                col: f"C: {col.split('_', 1)[1]}"
                for col in data.columns
                if col.startswith("contrast_")
            }
            default_mapping = {
                "term": "Term",
                "group": "Group",
                **de,
                **contrast_columns,
                "contrast": "Contrast",
                "estimate": "Estimate",
                "std_error": "Std.Error",
                "statistic": "z",
                "p_value": "P(>|z|)",
                "s_value": "S",
                "p_value_noninf": "p (NonInf)",
                "p_value_nonsup": "p (NonSup)",
                "p_value_equiv": "p (Equiv)",
                "pred_low": "Pred low",
                "pred_high": "Pred high",
            }
            if mapping is None:
                self.mapping = default_mapping
            else:
                for key, val in default_mapping.items():
                    if key not in mapping.keys():
                        mapping[key] = val
                self.mapping = mapping

            return
        super().__init__(data)

    def __str__(self):
        if hasattr(self, "conf_level"):
            self.mapping["conf_low"] = f"{(1 - self.conf_level) / 2 * 100:.1f}%"
            self.mapping["conf_high"] = f"{(1 - (1 - self.conf_level) / 2) * 100:.1f}%"
        else:
            self.mapping["conf_low"] = "["
            self.mapping["conf_high"] = "]"

        if hasattr(self, "by"):
            if self.by is None:
                valid = list(self.mapping.keys())
            elif self.by is True:
                valid = list(self.mapping.keys())
            elif self.by is False:
                valid = list(self.mapping.keys())
            elif isinstance(self.by, list):
                valid = self.by + list(self.mapping.keys())
            elif isinstance(self.by, str):
                valid = [self.by] + list(self.mapping.keys())
            else:
                raise ValueError("by must be None or a string or a list of strings")
        else:
            valid = list(self.mapping.keys())

        valid = self.datagrid_explicit + valid
        valid = [x for x in valid if x in self.columns]

        # sometimes we get duplicates when there is `by` and `datagrid()`
        valid = dict.fromkeys(valid)
        valid = list(valid.keys())

        out = self.print_head
        self.mapping = {key: self.mapping[key] for key in self.mapping if key in valid}
        tmp = self.select(valid).rename(self.mapping)
        for col in tmp.columns:
            if tmp[col].dtype.is_numeric():

                def fmt(x):
                    out = pl.Series([f"{i:.3g}" for i in x])
                    return out

                tmp.with_columns(
                    pl.col(col).map_batches(fmt, return_dtype=pl.Utf8).alias(col)
                )

        if "Term" in tmp.columns and len(tmp["Term"].unique()) == 1:
            term_str = tmp["Term"].unique()
            tmp = tmp.drop("Term")
        else:
            term_str = None

        if "Contrast" in tmp.columns and len(tmp["Contrast"].unique()) == 1:
            contrast_str = tmp["Contrast"].unique()
            tmp = tmp.drop("Contrast")
        else:
            contrast_str = None

        out += tmp.__str__()
        if term_str is not None:
            out += f"\nTerm: {term_str[0]}"
        if contrast_str is not None:
            out += f"\nContrast: {contrast_str[0]}"

        ## we no longer print the column names
        # out = out + f"\n\nColumns: {', '.join(self.columns)}\n"
        return out


def _detect_variable_type(
    data: pl.DataFrame, model: Optional[Any] = None
) -> Dict[str, str]:
    """
    Detect variable types in a DataFrame similar to R's detect_variable_class.

    Parameters:
    -----------
    data : pl.DataFrame
        The DataFrame to analyze
    model : Optional[Any]
        Optional model object (for compatibility)

    Returns:
    --------
    Dict[str, str]
        Dictionary mapping column names to variable types
    """
    variable_type = {}

    for col in data.columns:
        dtype = data[col].dtype

        # Skip columns with complex dtypes that don't support unique()
        try:
            unique_vals = data[col].unique()
            n_unique = len(unique_vals.drop_nulls())
        except pl.exceptions.InvalidOperationError:
            # If unique() is not supported, treat as "other"
            variable_type[col] = "other"
            continue

        # Check if binary (only 2 unique values, excluding nulls)
        if n_unique == 2:
            variable_type[col] = "binary"
        # Check if integer type
        elif (
            dtype == pl.Int64
            or dtype == pl.Int32
            or dtype == pl.Int16
            or dtype == pl.Int8
        ):
            variable_type[col] = "integer"
        # Check if numeric (float)
        elif dtype == pl.Float64 or dtype == pl.Float32:
            variable_type[col] = "numeric"
        # Check if boolean/logical
        elif dtype == pl.Boolean:
            variable_type[col] = "logical"
        # Check if categorical (explicit polars categorical) - check this first
        elif dtype == pl.Categorical:
            variable_type[col] = "categorical"
        # Check if string/character - strings are character by default
        elif dtype == pl.Utf8 or dtype == pl.String:
            variable_type[col] = "character"
        # Everything else
        else:
            variable_type[col] = "other"

    return variable_type


def _check_variable_type(
    variable_type: Dict[str, str], variable_name: str, expected_type: str
) -> bool:
    """
    Check if a variable has the expected type.

    Parameters:
    -----------
    variable_type : Dict[str, str]
        Dictionary mapping variable names to types
    variable_name : str
        Name of the variable to check
    expected_type : str
        Expected variable type

    Returns:
    --------
    bool
        True if variable has expected type, False otherwise
    """
    if variable_name not in variable_type:
        return False

    actual_type = variable_type[variable_name]

    # Handle some aliases/mappings
    if expected_type == "factor" and actual_type == "categorical":
        return True
    elif expected_type == "categorical" and actual_type == "factor":
        return True
    else:
        return actual_type == expected_type
