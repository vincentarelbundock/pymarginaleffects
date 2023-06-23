# Linear model

``` python
import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from marginaleffects import comparisons, predictions

df = sm.datasets.get_rdataset("Guerry", "HistData").data
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
fit = mod.fit()
```

# `comparisons()`

## Difference

``` python
comparisons(fit, variables = "Pop1831", value = 1, comparison = "differenceavg")
```

<small>shape: (1, 5)</small>

| estimate | std_error | statistic | conf_low  | conf_high |
|----------|-----------|-----------|-----------|-----------|
| f64      | f64       | f64       | f64       | f64       |
| 0.003717 | 0.011597  | 0.320485  | -0.019353 | 0.026786  |


``` python
comparisons(fit, variables = "Pop1831", value = 1, comparison = "difference").head()
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


## Ratio

``` python
comparisons(fit, variables = "Pop1831", value = 1, comparison = "ratioavg")
```

<small>shape: (1, 5)</small>

| estimate | std_error | statistic   | conf_low | conf_high |
|----------|-----------|-------------|----------|-----------|
| f64      | f64       | f64         | f64      | f64       |
| 1.000255 | 0.00033   | 3029.838358 | 0.999598 | 1.000911  |


``` python
comparisons(fit, variables = "Pop1831", value = 1, comparison = "ratio").head()
```

<small>shape: (5, 28)</small>

| estimate | std_error | statistic   | conf_low | conf_high | dept | Region | Department     | Crime_pers | Crime_prop | Literacy | Donations | Infants | Suicides | MainCity | Wealth | Commerce | Clergy | Crime_parents | Infanticide | Donation_clergy | Lottery | Desertion | Instruction | Prostitutes | Distance | Area | Pop1831 |
|----------|-----------|-------------|----------|-----------|------|--------|----------------|------------|------------|----------|-----------|---------|----------|----------|--------|----------|--------|---------------|-------------|-----------------|---------|-----------|-------------|-------------|----------|------|---------|
| f64      | f64       | f64         | f64      | f64       | i64  | str    | str            | i64        | i64        | i64      | i64       | i64     | i64      | str      | i64    | i64      | i64    | i64           | i64         | i64             | i64     | i64       | i64         | i64         | f64      | i64  | f64     |
| 0.999769 | 0.000315  | 3176.733532 | 0.999143 | 1.000395  | 1    | "E"    | "Ain"          | 28870      | 15890      | 37       | 5098      | 33120   | 35039    | "2:Med"  | 73     | 58       | 11     | 71            | 60          | 69              | 41      | 55        | 46          | 13          | 218.372  | 5762 | 346.03  |
| 0.999044 | 0.000647  | 1544.357883 | 0.997758 | 1.000331  | 2    | "N"    | "Aisne"        | 26226      | 5521       | 51       | 8901      | 14572   | 12831    | "2:Med"  | 22     | 10       | 82     | 4             | 82          | 36              | 38      | 82        | 24          | 327         | 65.945   | 7369 | 513.0   |
| 1.001225 | 0.000675  | 1484.291758 | 0.999883 | 1.002567  | 3    | "C"    | "Allier"       | 26747      | 7925       | 13       | 10973     | 17044   | 114121   | "2:Med"  | 61     | 66       | 68     | 46            | 42          | 76              | 66      | 16        | 85          | 34          | 161.927  | 7340 | 298.26  |
| 1.000529 | 0.000434  | 2306.440576 | 0.999666 | 1.001391  | 4    | "E"    | "Basses-Alpes" | 12935      | 7289       | 46       | 2733      | 23018   | 14238    | "1:Sm"   | 76     | 49       | 5      | 70            | 12          | 37              | 80      | 32        | 29          | 2           | 351.399  | 6925 | 155.9   |
| 1.000405 | 0.000393  | 2545.67391  | 0.999624 | 1.001187  | 5    | "E"    | "Hautes-Alpes" | 17488      | 8174       | 69       | 6962      | 23076   | 16171    | "1:Sm"   | 83     | 65       | 10     | 22            | 23          | 64              | 79      | 35        | 7           | 1           | 320.28   | 5549 | 129.1   |


## Group averages (SEs are broken)

``` python
comparisons(fit, variables = "Pop1831", value = 1, comparison = "difference", by = "Region")
```

<small>shape: (6, 6)</small>

| Region | estimate  | std_error | statistic | conf_low  | conf_high |
|--------|-----------|-----------|-----------|-----------|-----------|
| str    | f64       | f64       | f64       | f64       | f64       |
| "E"    | -0.011623 | 0.014098  | -0.824456 | -0.039668 | 0.016422  |
| "N"    | -0.004984 | 0.012579  | -0.396231 | -0.030007 | 0.020039  |
| "C"    | 0.017379  | 0.012825  | 1.355131  | -0.008133 | 0.042891  |
| "S"    | 0.021782  | 0.013844  | 1.573426  | -0.005757 | 0.049321  |
| "W"    | -0.006382 | 0.012851  | -0.496605 | -0.031946 | 0.019182  |
| null   | 0.044704  | 0.021702  | 2.059896  | 0.001532  | 0.087876  |


# `predictions()`

``` python
predictions(fit).head()
```

<small>shape: (5, 24)</small>

| estimate  | dept | Region | Department     | Crime_pers | Crime_prop | Literacy | Donations | Infants | Suicides | MainCity | Wealth | Commerce | Clergy | Crime_parents | Infanticide | Donation_clergy | Lottery | Desertion | Instruction | Prostitutes | Distance | Area | Pop1831 |
|-----------|------|--------|----------------|------------|------------|----------|-----------|---------|----------|----------|--------|----------|--------|---------------|-------------|-----------------|---------|-----------|-------------|-------------|----------|------|---------|
| f64       | i64  | str    | str            | i64        | i64        | i64      | i64       | i64     | i64      | str      | i64    | i64      | i64    | i64           | i64         | i64             | i64     | i64       | i64         | i64         | f64      | i64  | f64     |
| 42.992617 | 1    | "E"    | "Ain"          | 28870      | 15890      | 37       | 5098      | 33120   | 35039    | "2:Med"  | 73     | 58       | 11     | 71            | 60          | 69              | 41      | 55        | 46          | 13          | 218.372  | 5762 | 346.03  |
| 43.954782 | 2    | "N"    | "Aisne"        | 26226      | 5521       | 51       | 8901      | 14572   | 12831    | "2:Med"  | 22     | 10       | 82     | 4             | 82          | 36              | 38      | 82        | 24          | 327         | 65.945   | 7369 | 513.0   |
| 29.729568 | 3    | "C"    | "Allier"       | 26747      | 7925       | 13       | 10973     | 17044   | 114121   | "2:Med"  | 61     | 66       | 68     | 46            | 42          | 76              | 66      | 16        | 85          | 34          | 161.927  | 7340 | 298.26  |
| 32.891659 | 4    | "E"    | "Basses-Alpes" | 12935      | 7289       | 46       | 2733      | 23018   | 14238    | "1:Sm"   | 76     | 49       | 5      | 70            | 12          | 37              | 80      | 32        | 29          | 2           | 351.399  | 6925 | 155.9   |
| 34.085588 | 5    | "E"    | "Hautes-Alpes" | 17488      | 8174       | 69       | 6962      | 23076   | 16171    | "1:Sm"   | 83     | 65       | 10     | 22            | 23          | 64              | 79      | 35        | 7           | 1           | 320.28   | 5549 | 129.1   |


## Group averages (SEs are broken)

``` python
predictions(fit, by = "Region")
```

<small>shape: (6, 2)</small>

| Region | estimate  |
|--------|-----------|
| str    | f64       |
| "E"    | 43.892679 |
| "N"    | 41.974184 |
| "C"    | 35.693435 |
| "S"    | 33.82303  |
| "W"    | 41.871616 |
| null   | 22.66595  |


# `hypothesis` argument

``` python
hyp = np.array([1, 0, -1, 0, 0, 0])
predictions(fit, by = "Region", hypothesis = hyp)

hyp = np.vstack([
    [1, 0, -1, 0, 0, 0],
    [1, 0, 0, -1, 0, 0]
]).T
predictions(fit, by = "Region", hypothesis = hyp)
```

<small>shape: (2, 1)</small>

| estimate  |
|-----------|
| f64       |
| 8.199244  |
| 10.069649 |


Which corresponds to:

``` python
p = predictions(fit, by = "Region")
print(p["estimate"][0] - p["estimate"][2])
print(p["estimate"][0] - p["estimate"][3])
```

    8.199243639851169
    10.069648968961637
