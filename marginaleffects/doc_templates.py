_TEMPLATE_ORDER_OF_OPERATIONS = """
    Order of operations:

    Behind the scenes, the arguments of `marginaleffects` functions are evaluated in this order:

    1. `newdata`
    2. `variables`
    3. `comparison` and `slope`
    4. `by`
    5. `vcov`
    6. `hypothesis`
    7. `transform`
    """

_TEMPLATE_COMPARISONS_NOTE = """
    The `equivalence` argument specifies the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. The first element specifies the lower bound, and the second element specifies the upper bound. If `None`, equivalence tests are not performed.
    """
