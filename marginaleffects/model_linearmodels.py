from marginaleffects.utils import get_type_dictionary
import re
import numpy as np
import pandas as pd
import patsy
import polars as pl
import warnings
from .model_abstract import ModelAbstract


class ModelLinearmodels(ModelAbstract):
    """
    A class for handling linearmodels' models in the marginaleffects framework.

    This class provides methods for interfacing between linearmodels,
    which expects pandas DataFrames with a MultiIndex, and marginaleffects,
    which uses polars DataFrames that don't have indexes.

    Parameters
    ----------
    model : linearmodels.panel.results.PanelResults
        The fitted linearmodels model.
    dataframe : pandas.DataFrame
        The original dataframe used to fit the model.

    Attributes
    ----------
    model : linearmodels.model
        The fitted linearmodels model.
    _dataframe : pandas.DataFrame
        The original dataframe used to fit the model.
    variables_type : dict
        A dictionary of variable types in the model.
    """

    def __init__(self, model, dataframe):
        self.model = model
        self._dataframe = dataframe
        self.validate_coef()
        self.validate_modeldata()
        self.validate_response_name()
        self.validate_formula()
        self.variables_type = get_type_dictionary(self.modeldata)

    @property
    def multiindex(self):
        """
        Get the MultiIndex of the original dataframe.

        Returns
        -------
        pandas.MultiIndex or None
            The MultiIndex if the original dataframe has one, else None.
        """
        if isinstance(self._dataframe.index, pd.MultiIndex):
            return self._dataframe.index

    @property
    def multiindex_names(self):
        """
        Get the names of the MultiIndex levels.

        Returns
        -------
        list
            The names of the MultiIndex levels.
        """
        return list(self.multiindex.names)

    def _pl_to_pd(self, df):
        """
        Convert a polars DataFrame with the original index as columns to a pandas DataFrame with these columns as MultiIndex.

        Parameters
        ----------
        df : polars.DataFrame or pandas.DataFrame
            The DataFrame to convert.

        Returns
        -------
        pandas.DataFrame
            The converted DataFrame with MultiIndex.
        """
        if isinstance(df, pl.DataFrame):
            return df.to_pandas().set_index(self.multiindex_names)
        return df

    def _pd_to_pl(self, df):
        """
        Convert a pandas DataFrame with MultiIndex to a polars DataFrame
        with the MultiIndex as columns.

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            The DataFrame to convert.

        Returns
        -------
        polars.DataFrame
            The converted DataFrame.
        """
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df.reset_index())
        return df

    def get_coef(self):
        return np.array(self.model.params)

    def get_coef_names(self):
        return np.array(self.model.params.index.to_numpy())

    def get_modeldata(self):
        return self._pd_to_pl(self._dataframe)

    def get_response_name(self):
        return self.model.model.dependent.vars[0]

    def get_vcov(self, vcov=True):
        if isinstance(vcov, bool):
            if vcov is True:
                V = self.model.cov
            else:
                V = None
        elif isinstance(vcov, str):
            raise ValueError(f"Linearmodels currently does not support {vcov} vcov.")
            lab = f"cov_{vcov}"
            if hasattr(self.model, lab):
                V = getattr(self.model, lab)
            else:
                raise ValueError(f"The model object has no {lab} attribute.")
        else:
            raise ValueError(
                '`vcov` must be a boolean or a string like "HC3", which corresponds to an attribute of the model object such as "vcov_HC3".'
            )

        if V is not None:
            V = np.array(V)
            if V.shape != (len(self.coef), len(self.coef)):
                raise ValueError(
                    "vcov must be a square numpy array with dimensions equal to the length of self.coef"
                )

        return V

    def get_variables_names(self, variables=None, newdata=None):
        if variables is None:
            formula = self.formula
            columns = self.modeldata.columns
            order = {}
            for var in columns:
                match = re.search(rf"\b{re.escape(var)}\b", formula.split("~")[1])
                if match:
                    order[var] = match.start()
            variables = sorted(order, key=lambda i: order[i])

        if isinstance(variables, (str, dict)):
            variables = [variables] if isinstance(variables, str) else variables
        elif isinstance(variables, list) and all(
            isinstance(var, str) for var in variables
        ):
            pass
        else:
            raise ValueError(
                "`variables` must be None, a dict, string, or list of strings"
            )

        if newdata is not None:
            good = [x for x in variables if x in newdata.columns]
            bad = [x for x in variables if x not in newdata.columns]
            if len(bad) > 0:
                bad = ", ".join(bad)
                warnings.warn(f"Variable(s) not in newdata: {bad}")
            if len(good) == 0:
                raise ValueError("There is no valid column name in `variables`.")
        return variables

    def get_predict(self, params, newdata: pl.DataFrame):
        if isinstance(newdata, np.ndarray):
            exog = newdata
        else:
            y, exog = patsy.dmatrices(
                self.formula, self._pl_to_pd(newdata), return_type="dataframe"
            )

        p = self.model.model.predict(params=params, exog=exog).predictions.values
        if p.ndim == 1:
            p = pl.DataFrame({"rowid": range(newdata.shape[0]), "estimate": p})
        elif p.ndim == 2:
            colnames = {f"column_{i}": str(i) for i in range(p.shape[1])}
            p = (
                pl.DataFrame(p)
                .rename(colnames)
                .with_columns(
                    pl.Series(range(p.shape[0]), dtype=pl.Int32).alias("rowid")
                )
                .melt(id_vars="rowid", variable_name="group", value_name="estimate")
            )
        else:
            raise ValueError(
                "The `predict()` method must return an array with 1 or 2 dimensions."
            )
        p = p.with_columns(pl.col("rowid").cast(pl.Int32))
        return p

    def get_formula(self):
        return self.model.model.formula

    def get_df(self):
        return self.model.df_resid
