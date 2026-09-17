import numpy as np
import matplotlib.pyplot as plt


def shapley_analysis(
    model,
    X,
    n_samples=30,
    seed=0
):
    rng = np.random.default_rng(seed)

    n, d = X.shape

    shap_vals = np.zeros(
        (n, d)
    )

    for i in range(
        min(n, n_samples)
    ):

        x = X[i]

        for j in range(d):

            marginal = 0.0
            n_perm = 20

            for _ in range(n_perm):

                subset = [
                    k for k in range(d)
                    if k != j
                ]

                rng.shuffle(subset)

                k = rng.integers(
                    1,
                    d
                )

                S = subset[:k]

                z = X[
                    rng.integers(0, n)
                ].copy()

                x_S = x.copy()
                x_S[S] = z[S]

                marginal += model.predict(
                    x_S[None, :]
                )[0]

            shap_vals[i, j] = (
                model.predict(
                    x[None, :]
                )[0]
                - marginal / n_perm
            )

    return shap_vals


def plot_shap_summary(
    shap_vals,
    feature_names,
    title="SHAP Feature Importance"
):
    plt.figure(figsize=(8, 5))

    mean_abs = np.abs(
        shap_vals
    ).mean(axis=0)

    order = np.argsort(
        mean_abs
    )

    plt.barh(
        [
            feature_names[i]
            for i in order
        ],
        mean_abs[order]
    )

    plt.xlabel(
        "mean |SHAP value|"
    )

    plt.title(title)

    plt.tight_layout()

    filename = (
        title.replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        filename,
        dpi=150
    )

    plt.show()