import polars as pl


class MarginaleffectsDataFrame(pl.DataFrame):
    def __init__(
        self,
        data=None,
        by=None,
        conf_level=0.95,
        jacobian=None,
        newdata=None,
        mapping=None,
        print_head="",
    ):
        if isinstance(data, pl.DataFrame):
            self._df = data._df
            self.by = by
            self.conf_level = conf_level
            self.jacobian = jacobian
            if hasattr(newdata, "datagrid_explicit"):
                self.datagrid_explicit = newdata.datagrid_explicit
            else:
                self.datagrid_explicit = []

            self.print_head = print_head

            default_mapping = {
                "term": "Term",
                "contrast": "Contrast",
                "estimate": "Estimate",
                "std_error": "Std.Error",
                "statistic": "z",
                "p_value": "P(>|z|)",
                "s_value": "S",
            }
            if mapping is None:
                self.mapping = default_mapping
            else:
                for key, val in default_mapping.items():
                    if key not in mapping.keys():
                        mapping[key] = val
                self.mapping = mapping

            return
        super().__init__(data)

    def __str__(self):
        if hasattr(self, "conf_level"):
            self.mapping["conf_low"] = f"{(1 - self.conf_level) / 2 * 100:.1f}%"
            self.mapping["conf_high"] = f"{(1 - (1 - self.conf_level) / 2) * 100:.1f}%"
        else:
            self.mapping["conf_low"] = "["
            self.mapping["conf_high"] = "]"

        if hasattr(self, "by"):
            if self.by is None:
                valid = list(self.mapping.keys())
            elif self.by is True:
                valid = list(self.mapping.keys())
            elif self.by is False:
                valid = list(self.mapping.keys())
            elif isinstance(self.by, list):
                valid = self.by + list(self.mapping.keys())
            elif isinstance(self.by, str):
                valid = [self.by] + list(self.mapping.keys())
            else:
                raise ValueError("by must be None or a string or a list of strings")
        else:
            valid = list(self.mapping.keys())

        valid = self.datagrid_explicit + valid
        valid = [x for x in valid if x in self.columns]

        # sometimes we get duplicates when there is `by` and `datagrid()`
        valid = dict.fromkeys(valid)
        valid = list(valid.keys())

        out = self.print_head
        self.mapping = {key: self.mapping[key] for key in self.mapping if key in valid}
        tmp = self.select(valid).rename(self.mapping)
        for col in tmp.columns:
            if tmp[col].dtype.is_numeric():
                tmp = tmp.with_columns(
                    pl.col(col).map_elements(lambda x: f"{x:.3g}", return_dtype=pl.Utf8)
                )
        out += tmp.__str__()
        out = out + f"\n\nColumns: {', '.join(self.columns)}\n"
        return out
