import warnings

import numpy as np
import matplotlib.pyplot as plt

from data import get_dataframe
from rsm import fit_rsm
from anfis import ANFIS
from shap_analysis import (
    shapley_analysis,
    plot_shap_summary
)
from mso import MSO
from error_analysis import (
    error_analysis,
    summarize_errors
)

warnings.filterwarnings("ignore")


RESPONSES = [
    "R1",
    "R2",
    "R3",
    "R4"
]

FEATURES = [
    "A",
    "B",
    "C",
    "D"
]


def main():

    print("=" * 60)
    print(
        "MITOCHONDRIAL PROTEIN "
        "CO-EVOLUTION ANALYSIS PIPELINE"
    )
    print(
        "Embedded data — no external file needed"
    )
    print("=" * 60)

    # ========================================================
    # DATA
    # ========================================================

    df = get_dataframe()

    print(
        f"\nLoaded {len(df)} runs "
        f"× {df.shape[1]} columns"
    )

    print(
        df.head(3).to_string(
            index=False
        )
    )

    # ========================================================
    # RSM
    # ========================================================

    rsm_models = {}

    for resp in RESPONSES:

        print(
            f"\n{'=' * 60}\n"
            f"RSM Fitting: {resp}\n"
            f"{'=' * 60}"
        )

        model = fit_rsm(
            df,
            resp
        )

        rsm_models[resp] = model

        print(
            model["anova"].to_string(
                index=False
            )
        )

        print(
            f"R²={model['r2']:.4f} "
            f"Adj-R²={model['adj_r2']:.4f} "
            f"Pred-R²={model['pred_r2']:.4f} "
            f"CV%={model['cv']:.4f}"
        )

    # ========================================================
    # ANFIS
    # ========================================================

    print(
        f"\n{'=' * 60}\n"
        "ANFIS Training\n"
        f"{'=' * 60}"
    )

    X = df[
        FEATURES
    ].values

    anfis_results = {}

    for resp in RESPONSES:

        print(
            f"\n--- ANFIS for {resp} ---"
        )

        model = ANFIS(
            n_inputs=4,
            n_mf=3,
            epochs=150,
            lr=0.005
        )

        model.fit(
            X,
            df[resp].values
        )

        anfis_results[resp] = model

    # ========================================================
    # SHAP
    # ========================================================

    print(
        f"\n{'=' * 60}\n"
        "SHAP Feature Attribution (R1)\n"
        f"{'=' * 60}"
    )

    shap_vals = shapley_analysis(
        anfis_results["R1"],
        X,
        n_samples=30
    )

    plot_shap_summary(
        shap_vals,
        FEATURES,
        title="SHAP_ANFIS_R1"
    )

    # ========================================================
    # MSO
    # ========================================================

    print(
        f"\n{'=' * 60}\n"
        "MSO Multi-Objective Optimization\n"
        f"{'=' * 60}"
    )

    bounds = [
        (
            df[col].min(),
            df[col].max()
        )
        for col in FEATURES
    ]

    mso = MSO(
        bounds=bounds,
        n_particles=30,
        n_iter=80,
        w=[
            0.35,
            0.30,
            0.05,
            0.30
        ]
    )

    opt = mso.optimize(
        rsm_models
    )

    print(
        "\nOptimal settings:"
    )

    for name, value in zip(
        FEATURES,
        opt["best_x"]
    ):
        print(
            f"  {name} = {value:.4f}"
        )

    print(
        f"Best objective = "
        f"{opt['best_obj']:.5f}"
    )

    # MSO convergence plot

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        opt["history"]
    )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "Best objective"
    )

    plt.title(
        "MSO Convergence"
    )

    plt.tight_layout()

    plt.savefig(
        "MSO_convergence.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    print(
        f"\n{'=' * 60}\n"
        "Error Analysis\n"
        f"{'=' * 60}"
    )

    err_summary = {}

    for resp in RESPONSES:

        print(
            f"\n>>> {resp} — RSM <<<"
        )

        err_summary[
            f"{resp}_RSM"
        ] = error_analysis(
            df[resp].values,
            rsm_models[resp]["y_hat"],
            f"{resp}_RSM"
        )

        print(
            f">>> {resp} — ANFIS <<<"
        )

        anfis_pred = (
            anfis_results[resp]
            .predict(X)
        )

        err_summary[
            f"{resp}_ANFIS"
        ] = error_analysis(
            df[resp].values,
            anfis_pred,
            f"{resp}_ANFIS"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"\n{'=' * 60}\n"
        "SUMMARY\n"
        f"{'=' * 60}"
    )

    print(
        summarize_errors(
            err_summary
        )
    )

    print(
        "\n✓ Pipeline complete."
    )

    print(
        "✓ Output files generated."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()