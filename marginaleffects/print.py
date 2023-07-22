def get_printout(x, conf_level = 0.95):
    column_mappings = {
        "estimate": "Estimate",
        "std_error": "Std.Error",
        "statistic": "z",
        "p_value": "P(>|z|)",
        "s_value": "S",
        "conf_low": f"{(1 - conf_level) / 2 * 100:.1f}",
        "conf_high": f"{1 - (1 - conf_level) / 2 * 100:.1f}",
    }
    existing_columns = [column for column in column_mappings.keys() if column in x.columns]
    new_names = {column: column_mappings[column] for column in existing_columns}
    out = x.select(existing_columns).rename(new_names)
    out = out.__str__()
    out = out + "\n\nColumns: " + ", ".join(x.columns)
    return out