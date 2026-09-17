import numpy as np
import pandas as pd
from scipy import stats


def build_rsm_design(df):
    A = df["A"].values
    B = df["B"].values
    C = df["C"].values
    D = df["D"].values

    X = np.column_stack([
        np.ones_like(A),
        A, B, C, D,
        A * B, A * C, A * D,
        B * C, B * D, C * D,
        A ** 2, B ** 2, C ** 2, D ** 2
    ])

    terms = [
        "Intercept",
        "A", "B", "C", "D",
        "AB", "AC", "AD", "BC", "BD", "CD",
        "A²", "B²", "C²", "D²"
    ]

    return X, terms


def fit_rsm(df, response="R1"):
    X, terms = build_rsm_design(df)
    y = df[response].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    residuals = y - y_hat

    n, p = X.shape
    df_res = n - p

    ss_res = np.sum(residuals ** 2)
    ss_total = np.sum((y - y.mean()) ** 2)

    ms_res = ss_res / df_res

    ss_terms = {}

    for i, term in enumerate(terms):
        if i == 0:
            ss_terms[term] = np.sum((y_hat - y.mean()) ** 2)
            continue

        X_red = np.delete(X, i, axis=1)
        beta_red, *_ = np.linalg.lstsq(X_red, y, rcond=None)

        ss_red = np.sum((y - X_red @ beta_red) ** 2)
        ss_terms[term] = ss_red - ss_res

    anova = []

    for i, term in enumerate(terms):
        if i == 0:
            continue

        F = ss_terms[term] / ms_res if ms_res > 0 else np.inf
        pval = 1 - stats.f.cdf(F, 1, df_res)

        anova.append({
            "Term": term,
            "SS": ss_terms[term],
            "df": 1,
            "MS": ss_terms[term],
            "F": F,
            "p": pval
        })

    r2 = 1 - ss_res / ss_total
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_res

    h = np.sum(
        X * (X @ np.linalg.pinv(X.T @ X)),
        axis=1
    )

    press = np.sum((residuals / (1 - h)) ** 2)
    pred_r2 = 1 - press / ss_total

    return {
        "coefficients": dict(zip(terms, beta)),
        "anova": pd.DataFrame(anova),
        "r2": r2,
        "adj_r2": adj_r2,
        "pred_r2": pred_r2,
        "y_hat": y_hat,
        "residuals": residuals,
        "std_dev": np.sqrt(ms_res),
        "cv": np.sqrt(ms_res) / y.mean() * 100,
        "mean": y.mean()
    }