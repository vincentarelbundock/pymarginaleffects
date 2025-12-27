import numpy as np
import polars as pl
import warnings
from .utils import get_type_dictionary
from . import formulaic_utils as fml


class ModelValidation:
    def validation(self):
        self.validate_coef()
        self.validate_response_name()
        self.validate_formula()
        self.validate_modeldata()
        self.vault.update(
            variables_type=get_type_dictionary(self.get_formula(), self.get_modeldata())
        )

    def validate_coef(self):
        coef = self.get_coef()
        if not isinstance(coef, np.ndarray) and coef is not None:
            raise ValueError("coef must be a numpy array")
        self.coef = coef

    def validate_response_name(self):
        response_name = self.find_response()
        if not isinstance(response_name, str) and response_name is not None:
            raise ValueError("response_name must be a string")
        self.response_name = response_name

    def validate_formula(self):
        formula = self.get_formula()

        if not formula:
            return

        if not callable(formula) and not isinstance(formula, str):
            raise ValueError(
                "formula must be a string or a pre-processing function that returns `y` and `X` matrices."
            )

        if callable(formula):
            return

        if "~" not in formula:
            raise ValueError(
                "Model formula must contain '~' to separate dependent and independent variables"
            )

        if "scale(" in formula or "center(" in formula:
            raise ValueError(
                "The formula cannot include scale( or center(. Please center your variables before fitting the model."
            )

        # TODO: deduplicate once we only use the vault
        self.vault.update(formula=formula)
        self.formula = formula

    def validate_modeldata(self):
        modeldata = self.get_modeldata()
        if not isinstance(modeldata, pl.DataFrame):
            raise ValueError("data attribute must be a Polars DataFrame")

        formula = self.get_formula()
        if callable(formula):
            return

        # there can be no missing values in the formula variables
        original_row_count = modeldata.shape[0]
        modeldata = fml.listwise_deletion(self.get_formula(), modeldata)
        if modeldata.shape[0] != original_row_count:
            warnings.warn("Dropping rows with missing observations.", UserWarning)

        # Check for String columns in formula variables ONLY
        # Get all variables used in the formula
        formula_vars = fml.parse_variables(self.get_formula())

        for c in formula_vars:
            # Skip metadata columns
            if c in ["index", "rownames"]:
                continue

            # Skip if column not in modeldata (e.g., response variable)
            if c not in modeldata.columns:
                continue

            if modeldata[c].dtype in [pl.Utf8, pl.String]:
                msg = (
                    f"Column '{c}' has String type and is used in the model formula. "
                    f"String columns are not allowed in formulas. "
                    f"Please convert to Categorical or Enum before fitting the model.\n\n"
                    f"For Polars DataFrames:\n"
                    f"  # Option 1: Cast to Categorical\n"
                    f"  df = df.with_columns(pl.col('{c}').cast(pl.Categorical))\n\n"
                    f"  # Option 2: Cast to Enum with explicit categories\n"
                    f"  categories = df['{c}'].unique().sort()\n"
                    f"  df = df.with_columns(pl.col('{c}').cast(pl.Enum(categories)))\n\n"
                    f"For pandas DataFrames:\n"
                    f"  df['{c}'] = df['{c}'].astype('category')"
                )
                raise TypeError(msg)

        # Check C() formula variables are already categorical
        catvars = fml.parse_variables_categorical(self.get_formula())
        for c in catvars:
            if modeldata[c].dtype not in [pl.Enum, pl.Categorical]:
                if modeldata[c].dtype.is_numeric():
                    msg = (
                        f"Variable '{c}' is wrapped in C() but has numeric type. "
                        f"C() should only be used with categorical variables."
                    )
                    raise TypeError(msg)
                else:
                    msg = (
                        f"Variable '{c}' is wrapped in C() but has unsupported type {modeldata[c].dtype}. "
                        f"Please convert to Categorical or Enum first."
                    )
                    raise TypeError(msg)

        # Convert Categorical → Enum for internal consistency (ONLY conversion that remains)
        # IMPORTANT: Use unique() not cat.get_categories() to avoid Polars global string cache
        # which would include categories from all datasets loaded in the session

        # Get category orders from model metadata to preserve them
        # For sklearn models with formulaic
        model_spec = self.vault.get("model_spec")
        formulaic_categories = {}
        if model_spec and hasattr(model_spec, "encoder_state"):
            for var_name, (kind, state) in model_spec.encoder_state.items():
                if "categories" in state:
                    # Extract the actual variable name (remove C() wrapper if present)
                    actual_var = var_name.replace("C(", "").replace(")", "")
                    formulaic_categories[actual_var] = state["categories"]

        # For statsmodels/patsy models, use pandas categorical orders
        pandas_categorical_orders = self.vault.get("pandas_categorical_orders", {})

        # Convert Categorical to Enum with appropriate ordering
        for c in modeldata.columns:
            if modeldata[c].dtype == pl.Categorical:
                # Priority 1: Use formulaic categories if available (sklearn models)
                if c in formulaic_categories:
                    catvals = formulaic_categories[c]
                # Priority 2: Use pandas categorical orders if available (statsmodels)
                elif c in pandas_categorical_orders:
                    catvals = pandas_categorical_orders[c]
                else:
                    # Default: sort lexically for consistency with R behavior
                    catvals = (
                        modeldata[c]
                        .unique()
                        .cast(pl.String)
                        .sort()
                        .drop_nulls()
                        .to_list()
                    )

                modeldata = modeldata.with_columns(pl.col(c).cast(pl.Enum(catvals)))

        self.vault.update(modeldata=modeldata)
