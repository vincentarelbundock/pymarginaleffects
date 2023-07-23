def get_printout(x, by = None, conf_int = 0.95):
    mapping = {
        "term": "Term",
        "contrast": "Contrast",
        "estimate": "Estimate",
        "std_error": "Std.Error",
        "statistic": "z",
        "p_value": "P(>|z|)",
        "s_value": "S",
        "conf_low": f"{(1 - conf_int) / 2 * 100:.1f}%",
        "conf_high": f"{(1 - (1 - conf_int) / 2) * 100:.1f}%",
    }
    if isinstance(by, list):
        valid = by + list(mapping.keys())
    elif isinstance(by, str):
        valid = [by] + list(mapping.keys())
    else:
        valid = list(mapping.keys())
    valid = [column for column in valid if column in x.columns]
    mapping = {key: value for key, value in mapping.items() if key in valid}
    out = x \
        .select(valid) \
        .rename(mapping)
    out = out.__str__()
    out = out + "\nColumns: " + ", ".join(x.columns)
    def fn():
        return out
    return fn