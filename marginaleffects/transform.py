import numpy as np
import polars as pl


def get_transform(x, transform=None):
    if transform is not None:
        transform = lambda x: np.exp(x)
        for col in ["estimate", "conf_low", "conf_high"]:
            if col in x.columns:
                x = x.with_columns(pl.col(col).apply(transform))
        return x.drop(["std_error", "statistic"])
    else:
        return x
