import re
from itertools import compress
import numpy as np
import polars as pl


def eval_string_hypothesis(x: pl.DataFrame, hypothesis: str, lab: str) -> pl.DataFrame:
    hypothesis = re.sub("=", "-(", hypothesis) + ")"
    if re.search(r"\bb\d+\b", hypothesis):
        bmax = max(
            [
                int(re.sub("b", "", match.group()))
                for match in re.finditer(r"\bb\d+\b", lab)
            ],
            default=0,
        )
        if bmax > x.shape[0]:
            msg = f"b{bmax} cannot be used in `hypothesis` because the call produced just {x.shape[0]} estimate(s). Try executing the exact same command without the `hypothesis` argument to see which estimates are available for hypothesis testing."
            raise ValueError(msg)

        for i in range(x.shape[0]):
            tmp = f"marginaleffects__{i}"
            hypothesis = re.sub(f"b{i}", tmp, hypothesis)

        rowlabels = [f"marginaleffects__{i}" for i in range(x.shape[0])]
    else:
        if "term" not in x.columns or len(x["term"]) != len(set(x["term"])):
            msg = 'To use term names in a `hypothesis` string, the same function call without `hypothesis` argument must produce a `term` column with unique row identifiers. You can use `b1`, `b2`, etc. indices instead of term names in the `hypotheses` string Ex: "b1 + b2 = 0" Alternatively, you can use the `newdata`, `variables`, or `by` arguments:'
            raise ValueError(msg)

        rowlabels = x["term"].to_list()

    def eval_string_function(vec, hypothesis, rowlabels):
        env = {rowlabel: vec[i] for i, rowlabel in enumerate(rowlabels)}
        hypothesis = hypothesis.replace("=", "==")
        out = eval(hypothesis, env)
        return out

    out = eval_string_function(
        x["estimate"].to_numpy(), hypothesis=hypothesis, rowlabels=rowlabels
    )

    out = pl.DataFrame({"term": [re.sub(r"\s+", "", lab)], "estimate": [out]})

    return out


# function extracts the estimate column from a data frame and sets it to x. If `hypothesis` argument is a numpy array, it feeds it directly to lincome_multiply. If lincome is a string, it checks if the string is valid, and then calls the corresponding function.
def get_hypothesis(x, hypothesis, by=None):
    msg = f"Invalid hypothesis argument: {hypothesis}. Valid arguments are: 'reference', 'revreference', 'sequential', 'revsequential', 'pairwise', 'revpairwise' or a numpy array or a float."

    if hypothesis is None or isinstance(hypothesis, (int, float)):
        return x
    if isinstance(hypothesis, np.ndarray):
        out = lincom_multiply(x, hypothesis)
        lab = [f"H{i + 1}" for i in range(out.shape[0])]
        out = out.with_columns(pl.Series(lab).alias("term"))
    elif isinstance(hypothesis, str) and re.search("=", hypothesis) is not None:
        out = eval_string_hypothesis(x, hypothesis, lab=hypothesis)
    elif isinstance(hypothesis, str):
        if hypothesis == "reference":
            hypmat = lincom_reference(x, by)
        elif hypothesis == "revreference":
            hypmat = lincom_revreference(x, by)
        elif hypothesis == "sequential":
            hypmat = lincom_sequential(x, by)
        elif hypothesis == "revsequential":
            hypmat = lincom_revsequential(x, by)
        elif hypothesis == "pairwise":
            hypmat = lincom_pairwise(x, by)
        elif hypothesis == "revpairwise":
            hypmat = lincom_revpairwise(x, by)
        else:
            raise ValueError(msg)
        out = lincom_multiply(x, hypmat.to_numpy())
        out = out.with_columns(pl.Series(hypmat.columns).alias("term"))
    else:
        raise ValueError(msg)
    return out


def lincom_multiply(x, lincom):
    estimates = x["estimate"]
    multiplied_results = np.dot(estimates, lincom)
    out = pl.DataFrame({"estimate": multiplied_results})
    return out


def get_hypothesis_row_labels(x, by=None):
    pattern = re.compile(r"^(term|by|group|value|contrast|contrast_)$")
    lab = [col for col in x.columns if pattern.match(col)]

    # Step 2: Filter columns with more than one unique value
    lab = [col for col in lab if len(x[col].unique()) > 1]

    # Step 3: Include additional columns from "by" if provided
    if by is not None:
        if isinstance(by, str):
            by = [by]
        lab = [e for e in list(set(lab) | set(by)) if e != "group"]

    # Step 4: If no columns left, return default
    if len(lab) == 0:
        return [f"{i}" for i in range(len(x))]

    # Step 5: Create a sub-dataframe with selected columns
    lab_df = x[lab]
    idx = [x[col].n_unique() > 1 for col in lab_df.columns]

    # Step 6: Create row labels by concatenating values
    if any(idx):
        lab_df = lab_df.select(list(compress(lab_df.columns, idx)))
    lab = lab_df.select(
        pl.concat_str(lab_df.columns, separator=", ").alias("concatenated")
    )["concatenated"].to_list()

    # Step 7: Wrap labels containing "-" in parentheses
    lab = [f"({label})" if "-" in label else label for label in lab]

    return lab


def lincom_revreference(x, by):
    lincom = -1 * np.identity(len(x))
    lincom[0] = 1
    lab = get_hypothesis_row_labels(x, by)
    if len(lab) == 0 or len(set(lab)) != len(lab):
        lab = [f"Row 1 - Row {i+1}" for i in range(len(lincom))]
    else:
        lab = [f"{lab[0]} - {la}" for la in lab]
    lincom = pl.DataFrame(lincom, schema=lab)
    lincom = lincom.select(lab[1:])
    return lincom


def lincom_reference(x, by):
    lincom = np.identity(len(x))
    lincom[0, :] = -1
    lab = get_hypothesis_row_labels(x, by)
    if len(lab) == 0 or len(set(lab)) != len(lab):
        lab = [f"Row {i+1} - Row 1" for i in range(len(lincom))]
    else:
        lab = [f"{la} - {lab[0]}" for la in lab]
    if lincom.shape[1] == 1:
        lincom = pl.DataFrame(lincom, schema=lab)
    else:
        lincom = pl.DataFrame(lincom.T, schema=lab)
    lincom = lincom.select(lab[1:])
    return lincom


def lincom_revsequential(x, by):
    lincom = np.zeros((len(x), len(x) - 1))
    lab = get_hypothesis_row_labels(x, by)
    if len(lab) == 0 or len(set(lab)) != len(lab):
        lab = [f"Row {i+1} - Row {i+2}" for i in range(lincom.shape[1])]
    else:
        lab = [f"{lab[i]} - {lab[i+1]}" for i in range(lincom.shape[1])]
    for i in range(lincom.shape[1]):
        lincom[i : i + 2, i] = [1, -1]
    if lincom.shape[1] == 1:
        lincom = pl.DataFrame(lincom, schema=lab)
    else:
        lincom = pl.DataFrame(lincom.T, schema=lab)
    return lincom


def lincom_sequential(x, by):
    lincom = np.zeros((len(x), len(x) - 1))
    lab = get_hypothesis_row_labels(x, by)
    if len(lab) == 0 or len(set(lab)) != len(lab):
        lab = [f"Row {i+2} - Row {i+1}" for i in range(lincom.shape[1])]
    else:
        lab = [f"{lab[i+1]} - {lab[i]}" for i in range(lincom.shape[1])]
    for i in range(lincom.shape[1]):
        lincom[i : i + 2, i] = [-1, 1]
    if lincom.shape[1] == 1:
        lincom = pl.DataFrame(lincom, schema=lab)
    else:
        lincom = pl.DataFrame(lincom.T, schema=lab)
    return lincom


def lincom_revpairwise(x, by):
    lab_row = get_hypothesis_row_labels(x, by)
    lab_col = []
    flag = len(lab_row) == 0 or len(set(lab_row)) != len(lab_row)
    mat = []
    for i in range(len(x)):
        for j in range(1, len(x)):
            if i < j:
                tmp = np.zeros((len(x), 1))
                tmp[i] = -1
                tmp[j] = 1
                mat.append(tmp)
                if flag:
                    lab_col.append(f"Row {j+1} - Row {i+1}")
                else:
                    lab_col.append(f"{lab_row[j]} - {lab_row[i]}")
    if len(mat) == 1:
        lincom = pl.DataFrame(mat[0], schema=lab_col)
    else:
        lincom = pl.DataFrame(np.hstack(mat).T, schema=lab_col)
    return lincom


def lincom_pairwise(x, by):
    lab_row = get_hypothesis_row_labels(x, by)
    lab_col = []
    flag = len(lab_row) == 0 or len(set(lab_row)) != len(lab_row)
    mat = []
    for i in range(len(x)):
        for j in range(1, len(x)):
            if i < j:
                tmp = np.zeros((len(x), 1))
                tmp[j] = -1
                tmp[i] = 1
                mat.append(tmp)
                if flag:
                    lab_col.append(f"Row {i+1} - Row {j+1}")
                else:
                    lab_col.append(f"{lab_row[i]} - {lab_row[j]}")
    if len(mat) == 1:
        lincom = pl.DataFrame(mat[0], schema=lab_col)
    else:
        lincom = pl.DataFrame(np.hstack(mat).T, schema=lab_col)
    return lincom
