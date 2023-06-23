
``` python
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import polars as pl
from marginaleffects import comparisons
pl.Config.set_tbl_formatting("ASCII_FULL")
df = sm.datasets.get_rdataset("Guerry", "HistData").data
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
fit = mod.fit()

comparisons(fit, "Pop1831", value = 1, comparison = "differenceavg")
```

<small>shape: (1, 5)</small>

| estimate | std_error | statistic | conf_low  | conf_high |
|----------|-----------|-----------|-----------|-----------|
| f64      | f64       | f64       | f64       | f64       |
| 0.003717 | 0.011597  | 0.320485  | -0.019353 | 0.026786  |


``` python
comparisons(fit, "Pop1831", value = 1, comparison = "difference").head()
```

<small>shape: (5, 28)</small>

| estimate  | std_error | statistic | conf_low  | conf_high | dept | Region | Department     | Crime_pers | Crime_prop | Literacy | Donations | Infants | Suicides | MainCity | Wealth | Commerce | Clergy | Crime_parents | Infanticide | Donation_clergy | Lottery | Desertion | Instruction | Prostitutes | Distance | Area | Pop1831 |
|-----------|-----------|-----------|-----------|-----------|------|--------|----------------|------------|------------|----------|-----------|---------|----------|----------|--------|----------|--------|---------------|-------------|-----------------|---------|-----------|-------------|-------------|----------|------|---------|
| f64       | f64       | f64       | f64       | f64       | i64  | str    | str            | i64        | i64        | i64      | i64       | i64     | i64      | str      | i64    | i64      | i64    | i64           | i64         | i64             | i64     | i64       | i64         | i64         | f64      | i64  | f64     |
| -0.009946 | 0.013663  | -0.727936 | -0.037126 | 0.017234  | 1    | "E"    | "Ain"          | 28870      | 15890      | 37       | 5098      | 33120   | 35039    | "2:Med"  | 73     | 58       | 11     | 71            | 60          | 69              | 41      | 55        | 46          | 13          | 218.372  | 5762 | 346.03  |
| -0.042023 | 0.025143  | -1.671333 | -0.092041 | 0.007995  | 2    | "N"    | "Aisne"        | 26226      | 5521       | 51       | 8901      | 14572   | 12831    | "2:Med"  | 22     | 10       | 82     | 4             | 82          | 36              | 38      | 82        | 24          | 327         | 65.945   | 7369 | 513.0   |
| 0.036388  | 0.018517  | 1.965047  | -0.000449 | 0.073225  | 3    | "C"    | "Allier"       | 26747      | 7925       | 13       | 10973     | 17044   | 114121   | "2:Med"  | 61     | 66       | 68     | 46            | 42          | 76              | 66      | 16        | 85          | 34          | 161.927  | 7340 | 298.26  |
| 0.017379  | 0.012825  | 1.355129  | -0.008133 | 0.042891  | 4    | "E"    | "Basses-Alpes" | 12935      | 7289       | 46       | 2733      | 23018   | 14238    | "1:Sm"   | 76     | 49       | 5      | 70            | 12          | 37              | 80      | 32        | 29          | 2           | 351.399  | 6925 | 155.9   |
| 0.013815  | 0.012195  | 1.132804  | -0.010445 | 0.038075  | 5    | "E"    | "Hautes-Alpes" | 17488      | 8174       | 69       | 6962      | 23076   | 16171    | "1:Sm"   | 83     | 65       | 10     | 22            | 23          | 64              | 79      | 35        | 7           | 1           | 320.28   | 5549 | 129.1   |

