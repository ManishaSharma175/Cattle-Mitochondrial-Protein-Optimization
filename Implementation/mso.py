import numpy as np
import pandas as pd

from rsm import build_rsm_design


class MSO:

    def __init__(
        self,
        bounds,
        n_particles=30,
        n_iter=80,
        w=None,
        seed=1
    ):
        self.bounds = np.array(
            bounds
        )

        self.n_particles = n_particles
        self.n_iter = n_iter

        self.w = (
            np.array(w)
            if w is not None
            else np.ones(4) / 4
        )

        self.rng = np.random.default_rng(
            seed
        )

    def _decode(self, pos):
        return (
            self.bounds[:, 0]
            + pos
            * (
                self.bounds[:, 1]
                - self.bounds[:, 0]
            )
        )

    def _objective(self, X, rsm_models):

        Xd, _ = build_rsm_design(
            pd.DataFrame(
                X,
                columns=[
                    "A", "B", "C", "D"
                ]
            )
        )

        preds = np.column_stack([
            Xd @ np.array(
                list(
                    rsm_models[r][
                        "coefficients"
                    ].values()
                )
            )
            for r in [
                "R1",
                "R2",
                "R3",
                "R4"
            ]
        ])

        return (
            self.w[0] * preds[:, 0]
            + self.w[1] * preds[:, 1]
            - self.w[2] * preds[:, 2]
            + self.w[3] * preds[:, 3]
        )

    def optimize(self, rsm_models):

        d = self.bounds.shape[0]

        pos = self.rng.random(
            (self.n_particles, d)
        )

        vel = self.rng.normal(
            0,
            0.1,
            (self.n_particles, d)
        )

        pbest = pos.copy()

        pbest_val = self._objective(
            self._decode(pos),
            rsm_models
        )

        gbest = pbest[
            np.argmax(pbest_val)
        ].copy()

        gbest_val = pbest_val.max()

        history = []

        for it in range(
            self.n_iter
        ):

            r1 = self.rng.random(
                (self.n_particles, d)
            )

            r2 = self.rng.random(
                (self.n_particles, d)
            )

            vel = (
                0.7 * vel
                + 1.5 * r1
                * (pbest - pos)
                + 1.5 * r2
                * (gbest - pos)
            )

            pos = np.clip(
                pos + vel,
                0,
                1
            )

            vals = self._objective(
                self._decode(pos),
                rsm_models
            )

            imp = vals > pbest_val

            pbest[imp] = pos[imp]
            pbest_val[imp] = vals[imp]

            if pbest_val.max() > gbest_val:
                gbest_val = pbest_val.max()

                gbest = pbest[
                    np.argmax(pbest_val)
                ].copy()

            history.append(
                gbest_val
            )

            if it % 20 == 0:
                print(
                    f"  MSO iter {it:3d} "
                    f"best obj = "
                    f"{gbest_val:.5f}"
                )

        return {
            "best_x": self._decode(gbest),
            "best_obj": gbest_val,
            "history": history
        }