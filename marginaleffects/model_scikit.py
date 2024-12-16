import numpy as np
import warnings
import polars as pl
from . import formulaic as fml
from .model_abstract import ModelAbstract


def is_sklearn_model(model):
    """Check if an object is a scikit-learn model.

    Parameters
    ----------
    model : object
        The model object to check.

    Returns
    -------
    bool
        True if the object is a scikit-learn model, False otherwise.

    Notes
    -----
    Checks both isinstance(BaseEstimator) and module name starting with 'sklearn'.
    Returns False if scikit-learn is not installed.
    """
    try:
        from sklearn.base import BaseEstimator
        return isinstance(model, BaseEstimator) or model.__module__.startswith("sklearn")
    except (AttributeError, ImportError):
        return False


class ModelScikit(ModelAbstract):
    """Wrapper class for scikit-learn models.
    
    Parameters
    ----------
    model : object
        A fitted scikit-learn model object with formula and data attributes.

    Attributes
    ----------
    formula : str
        Model formula with response ~ predictors format.
    data : pl.DataFrame
        Training data used to fit the model.

    Raises
    ------
    ValueError
        If model lacks a data attribute.
    TypeError
        If model.data is not a polars DataFrame.
    """
    def __init__(self, model):
        """Initialize a scikit-learn model wrapper.

        Parameters
        ----------
        model : object
            A fitted scikit-learn model that must have:
                - formula attribute (str): Model formula with response ~ predictors format
                - data attribute (pl.DataFrame): Training data used to fit the model

        Raises
        ------
        ValueError
            If model lacks a data attribute
        TypeError
            If model.data is not a polars DataFrame

        Notes
        -----
        The model must be already fitted and contain both formula and data attributes.
        The data must be a polars DataFrame for compatibility with the package.
        """
        super().__init__(model)
        self.formula = model.formula

        # Validate data attribute
        if not hasattr(model, "data"):
            raise ValueError("Model must have a 'data' attribute")
        if not isinstance(model.data, pl.DataFrame):
            raise TypeError("Model data must be a polars DataFrame")
        self.data = model.data

    def get_variables_names(self, variables=None, newdata=None):
        """Get names of predictor variables from the model formula.

        Parameters
        ----------
        variables : None
            Not used, kept for interface consistency.
        newdata : None
            Not used, kept for interface consistency.

        Returns
        -------
        list
            Names of predictor variables, excluding the response variable.
        """
        out = fml.variables(self.formula)[1:]
        return out

    def get_predict(self, params, newdata: pl.DataFrame):
        """Get model predictions for new data.

        Parameters
        ----------
        params : object
            Not used for scikit-learn models.
        newdata : pl.DataFrame
            New data to generate predictions for.

        Returns
        -------
        pl.DataFrame
            DataFrame containing:
                - rowid: Row identifier
                - estimate: Predicted values
                - group: (For multiclass) Class labels
            
        Raises
        ------
        ImportError
            If formulaic package is not installed.
        ValueError
            If predict() returns array with more than 2 dimensions.

        Notes
        -----
        - Tries predict_proba() first, falls back to predict()
        - For binary classification, returns probabilities for positive class only
        - For multiclass, returns probabilities for all classes
        """
        if isinstance(newdata, np.ndarray):
            exog = newdata
        else:
            try:
                import formulaic

                if isinstance(newdata, formulaic.ModelMatrix):
                    exog = newdata.to_numpy()
                else:
                    if isinstance(newdata, pl.DataFrame):
                        nd = newdata.to_pandas()
                    else:
                        nd = newdata
                    y, exog = formulaic.model_matrix(self.model.formula, nd)
                    exog = exog.to_numpy()
            except ImportError:
                raise ImportError(
                    "The formulaic package is required to use this feature."
                )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                p = self.model.predict_proba(exog)
                # only keep the second column for binary classification since it is redundant info
                if p.shape[1] == 2:
                    p = p[:, 1]
        except (AttributeError, NotImplementedError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                p = self.model.predict(exog)

        if p.ndim == 1:
            p = pl.DataFrame({"rowid": range(newdata.shape[0]), "estimate": p})
        elif p.ndim == 2:
            colnames = {f"column_{i}": v for i, v in enumerate(self.model.classes_)}
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
