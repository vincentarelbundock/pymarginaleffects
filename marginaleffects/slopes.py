from .comparisons import comparisons


def slopes(
    model,
    variables=None,
    newdata=None,
    slope="dydx",
    vcov=True,
    conf_int=0.95,
    by=False,
    hypothesis=None,
    eps=1e-4,
):
    if slope not in ["dydx", "eyex", "eydx", "dyex"]:
        raise ValueError("slope must be one of 'dydx', 'eyex', 'eydx', 'dyex'")

    out = comparisons(
        model=model,
        variables=variables,
        newdata=newdata,
        comparison=slope,
        vcov=vcov,
        conf_int=conf_int,
        by=by,
        hypothesis=hypothesis,
        eps=eps,
    )
    return out
