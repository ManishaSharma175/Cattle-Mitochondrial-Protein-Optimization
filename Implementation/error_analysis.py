import numpy as np
import pandas as pd


def error_analysis(
    y_true,
    y_pred,
    label="Response"
):
    resid = y_true - y_pred

    rmse = np.sqrt(
        np.mean(resid ** 2)
    )

    mae = np.mean(
        np.abs(resid)
    )

    mape = np.mean(
        np.abs(
            resid
            / (y_true + 1e-12)
        )
    ) * 100

    r2 = (
        1
        - np.sum(resid ** 2)
        / np.sum(
            (y_true - y_true.mean()) ** 2
        )
    )

    print(
        f"  RMSE={rmse:.5f} "
        f"MAE={mae:.5f} "
        f"MAPE={mape:.3f}% "
        f"R²={r2:.5f}"
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }


def summarize_errors(err_summary):
    return pd.DataFrame(
        err_summary
    ).T.round(5)