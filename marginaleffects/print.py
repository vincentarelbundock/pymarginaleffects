def get_printout(x, by = None, conf_int = 0.95):
    column_mapping = {
        "term": "Term",
        "contrast": "Contrast",
    }
    
    if by is not None:
        new_mapping = {key: key for key in by}
    else:
        new_mapping = {}
    column_mapping.update(new_mapping)
    new_mapping = {
        "term": "Term",
        "contrast": "Contrast",
        "estimate": "Estimate",
        "std_error": "Std.Error",
        "statistic": "z",
        "p_value": "P(>|z|)",
        "s_value": "S",
        "conf_low": f"{(1 - conf_int) / 2 * 100:.1f}",
        "conf_high": f"{(1 - (1 - conf_int) / 2) * 100:.1f}",
    }
    column_mapping.update(new_mapping)
    existing_columns = [column for column in column_mapping.keys() if column in x.columns]
    new_names = {column: column_mapping[column] for column in existing_columns}
    out = x.select(existing_columns).rename(new_names)
    out = out.__str__()
    out = out + "\n\nColumns: " + ", ".join(x.columns)
    return out