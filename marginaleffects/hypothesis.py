import numpy as np
import polars as pl


# function extracts the estimate column from a data frame and sets it to x. If `hypothesis` argument is a numpy array, it feeds it directly to lincome_multiply. If lincome is a string, it checks if the string is valid, and then calls the corresponding function.
def get_hypothesis(x, hypothesis):
    msg = f"Invalid hypothesis argument: {hypothesis}. Valid arguments are: 'reference', 'revreference', 'sequential', 'revsequential', 'pairwise', 'revpairwise' or a numpy array."
    if hypothesis is None:
        return(x)
    if isinstance(hypothesis, str):
        if hypothesis == "reference":
            hypothesis = lincom_reference(x)
        elif hypothesis == "revreference":
            hypothesis = lincom_revreference(x)
        elif hypothesis == "sequential":
            hypothesis = lincom_sequential(x)
        elif hypothesis == "revsequential":
            hypothesis = lincom_revsequential(x)
        elif hypothesis == "pairwise":
            hypothesis = lincom_pairwise(x)
        elif hypothesis == "revpairwise":
            hypothesis = lincom_revpairwise(x)
        else:
            raise ValueError(msg)
    elif isinstance(hypothesis, np.ndarray) is False:
        raise ValueError(msg)
    out = lincom_multiply(x, hypothesis)
    return out


def lincom_multiply(x, lincom):
    estimates = x['estimate']
    multiplied_results = np.dot(estimates, lincom)
    out = pl.DataFrame({
        'estimate': multiplied_results
    })
    return out


# TODO: improve labeling
def get_hypothesis_row_labels(x):
    return [f"i" for i in range(len(x))]

def lincom_revreference(x):
    lincom = -1 * np.identity(len(x))
    lincom[0] = 1
    lab = get_hypothesis_row_labels(x)
    if len(lab) == 0 or len(set(lab)) != len(lab):
        lab = [f"Row 1 - Row {i+1}" for i in range(len(lincom))]
    else:
        lab = [f"{lab[0]} - {l}" for l in lab]
    lincom = pl.DataFrame(lincom, schema=lab)
    lincom = lincom.select(lab[1:])
    return lincom


def lincom_reference(x):
    lincom = np.identity(len(x))
    lincom[0] = -1
    lab = get_hypothesis_row_labels(x)
    if len(lab) == 0 or len(set(lab)) != len(lab):
        lab = [f"Row {i+1} - Row 1" for i in range(len(lincom))]
    else:
        lab = [f"{l} - {lab[0]}" for l in lab]
    lincom = pl.DataFrame(lincom, schema=lab)
    lincom = lincom.select(lab[1:])
    return lincom


def lincom_revsequential(x):
    lincom = np.zeros((len(x), len(x) - 1))
    lab = get_hypothesis_row_labels(x)
    if len(lab) == 0 or len(set(lab)) != len(lab):
        lab = [f"Row {i+1} - Row {i+2}" for i in range(lincom.shape[1])]
    else:
        lab = [f"{lab[i]} - {lab[i+1]}" for i in range(lincom.shape[1])]
    for i in range(lincom.shape[1]):
        lincom[i:i+2, i] = [1, -1]
    lincom = pl.DataFrame(lincom, schema=lab)
    return lincom


def lincom_sequential(x):
    lincom = np.zeros((len(x), len(x) - 1))
    lab = get_hypothesis_row_labels(x)
    if len(lab) == 0 or len(set(lab)) != len(lab):
        lab = [f"Row {i+2} - Row {i+1}" for i in range(lincom.shape[1])]
    else:
        lab = [f"{lab[i+1]} - {lab[i]}" for i in range(lincom.shape[1])]
    for i in range(lincom.shape[1]):
        lincom[i:i+2, i] = [-1, 1]
    lincom = pl.DataFrame(lincom, schema=lab)
    return lincom


def lincom_revpairwise(x):
    lab_row = get_hypothesis_row_labels(x)
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
    lincom = np.hstack(mat)
    lincom = pl.DataFrame(lincom, schema=lab_col)
    return lincom


def lincom_pairwise(x):
    lab_row = get_hypothesis_row_labels(x)
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
    lincom = np.hstack(mat)
    lincom = pl.DataFrame(lincom, schema=lab_col)
    return lincom