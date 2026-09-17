import numpy as np


class ANFIS:
    """
    Simplified ANFIS using Gaussian membership functions
    with consequent and premise parameter learning.
    """

    def __init__(
        self,
        n_inputs=4,
        n_mf=3,
        epochs=150,
        lr=0.005,
        seed=42
    ):
        self.n_inputs = n_inputs
        self.n_mf = n_mf
        self.n_rules = n_mf ** n_inputs
        self.epochs = epochs
        self.lr = lr

        self.rng = np.random.default_rng(seed)

        self.centers = None
        self.sigmas = None
        self.consequents = None

    def _init_mfs(self, X):
        self.centers = np.zeros(
            (self.n_inputs, self.n_mf)
        )

        self.sigmas = np.zeros(
            (self.n_inputs, self.n_mf)
        )

        for i in range(self.n_inputs):
            lo = X[:, i].min()
            hi = X[:, i].max()

            self.centers[i] = np.linspace(
                lo, hi, self.n_mf
            )

            self.sigmas[i] = (
                (hi - lo) / (self.n_mf - 1)
                + 1e-6
            )

    def _membership(self, X):
        mu = np.zeros(
            (X.shape[0], self.n_inputs, self.n_mf)
        )

        for i in range(self.n_inputs):
            diff = (
                X[:, i:i + 1]
                - self.centers[i][None, :]
            )

            mu[:, i, :] = np.exp(
                -0.5
                * (
                    diff
                    / self.sigmas[i][None, :]
                ) ** 2
            )

        return mu

    def _rule_strengths(self, mu):
        n = mu.shape[0]

        W = np.ones(
            (n, self.n_rules)
        )

        for idx, combo in enumerate(
            np.ndindex(
                *([self.n_mf] * self.n_inputs)
            )
        ):
            w = np.ones(n)

            for i, m in enumerate(combo):
                w *= mu[:, i, m]

            W[:, idx] = w

        return W

    def _normalize(self, W):
        return W / (
            W.sum(axis=1, keepdims=True)
            + 1e-12
        )

    def fit(self, X, y):
        self._init_mfs(X)

        n = X.shape[0]

        self.consequents = self.rng.normal(
            0,
            0.01,
            (
                self.n_rules,
                self.n_inputs + 1
            )
        )

        Xa = np.hstack([
            X,
            np.ones((n, 1))
        ])

        for epoch in range(self.epochs):

            mu = self._membership(X)
            W = self._rule_strengths(mu)
            Wn = self._normalize(W)

            f_i = Xa @ self.consequents.T

            y_pred = np.sum(
                Wn * f_i,
                axis=1
            )

            err = y_pred - y

            # Consequent learning
            self.consequents -= (
                self.lr
                * (
                    Wn.T
                    @ (err[:, None] * Xa)
                )
                / n
            )

            # Premise learning
            for i in range(self.n_inputs):
                for m in range(self.n_mf):

                    dC = np.zeros(n)
                    dS = np.zeros(n)

                    for r, combo in enumerate(
                        np.ndindex(
                            *([self.n_mf] * self.n_inputs)
                        )
                    ):

                        if combo[i] != m:
                            continue

                        w_other = np.ones(n)

                        for j, mj in enumerate(combo):
                            if j != i:
                                w_other *= (
                                    mu[:, j, mj]
                                )

                        common = (
                            w_other
                            * (f_i[:, r] - y_pred)
                            * Wn[:, r]
                            * (1 - Wn[:, r])
                        )

                        dC += common
                        dS += common

                    diff = (
                        X[:, i]
                        - self.centers[i, m]
                    )

                    g = np.exp(
                        -0.5
                        * (
                            diff
                            / self.sigmas[i, m]
                        ) ** 2
                    )

                    self.centers[i, m] -= (
                        self.lr
                        * np.mean(
                            dC
                            * g
                            * diff
                            / (
                                self.sigmas[i, m] ** 2
                                + 1e-12
                            )
                        )
                    )

                    self.sigmas[i, m] -= (
                        self.lr
                        * np.mean(
                            dS
                            * g
                            * diff ** 2
                            / (
                                self.sigmas[i, m] ** 3
                                + 1e-12
                            )
                        )
                    )

                    self.sigmas[i, m] = max(
                        self.sigmas[i, m],
                        1e-4
                    )

            if epoch % 50 == 0:
                rmse = np.sqrt(
                    np.mean(err ** 2)
                )

                print(
                    f"  ANFIS epoch {epoch:4d} "
                    f"RMSE = {rmse:.5f}"
                )

        return self

    def predict(self, X):
        mu = self._membership(X)
        W = self._rule_strengths(mu)
        Wn = self._normalize(W)

        Xa = np.hstack([
            X,
            np.ones((X.shape[0], 1))
        ])

        f_i = Xa @ self.consequents.T

        return np.sum(
            Wn * f_i,
            axis=1
        )