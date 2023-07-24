import polars as pl

class MarginaleffectsDataFrame(pl.DataFrame):
    def __init__(self, data=None, by=None, conf_level=0.95):
        if isinstance(data, pl.DataFrame):
            self._df = data._df
            self.by = by
            self.conf_level = conf_level
            return
        super().__init__(data)

    def __str__(self):
        mapping = {
            "term": "Term",
            "contrast": "Contrast",
            "estimate": "Estimate",
            "std_error": "Std.Error",
            "statistic": "z",
            "p_value": "P(>|z|)",
            "s_value": "S"
        }

        if hasattr(self, "conf_level"):
            mapping["conf_low"] = f"{(1 - self.conf_level) / 2 * 100:.1f}%"
            mapping["conf_high"] = f"{(1 - (1 - self.conf_level) / 2) * 100:.1f}%"
        else:
            mapping["conf_low"] = "["
            mapping["conf_high"] = "]"

        if hasattr(self, "by"):
            if self.by is None:
                valid = list(mapping.keys())
            elif self.by is True:
                valid = list(mapping.keys())
            elif self.by is False:
                valid = list(mapping.keys())
            elif isinstance(self.by, list):
                valid = self.by + list(mapping.keys())
            elif isinstance(self.by, str):
                valid = [self.by] + list(mapping.keys())
            else:
                raise ValueError("by must be None or a string or a list of strings")
        else:
            valid = list(mapping.keys())
        valid = [x for x in valid if x in self.columns]
        mapping = {key: mapping[key] for key in mapping if key in valid}
        tmp = self.select(valid).rename(mapping)
        for col in tmp.columns:
            if tmp[col].is_numeric():
                tmp = tmp.with_columns(pl.col(col).apply(lambda x: f"{x:.3g}"))
        out = tmp.__str__()
        out = out + f"\n\nColumns: {', '.join(self.columns)}\n"
        return out
