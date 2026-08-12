import numpy as np

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.dataassociator.probability import JPDAwithEHM2
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.types.array import CovarianceMatrix
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.probability import PDAUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
# from stonesoup.dataassociator.probability import JPDA


def setup_tracker():
    # ==============================================================
    # 1. TARGET MOTION MODEL
    # ==============================================================
    #
    # State layout:
    #
    # [x, vx, y, vy, z, vz]
    #
    # ==============================================================

    transition_model = CombinedLinearGaussianTransitionModel(
        [
            ConstantVelocity(1.0),  # x / vx
            ConstantVelocity(1.0),  # y / vy
            ConstantVelocity(0.5),  # z / vz
        ]
    )

    # Predict tracks forward in time.
    predictor = KalmanPredictor(transition_model)

    # ==============================================================
    # 2. RADAR MEASUREMENT MODEL
    # ==============================================================
    #
    # Detection format:
    #
    # [elevation, bearing, range]
    #
    # ==============================================================

    radar_noise = CovarianceMatrix(
        np.diag(
            [
                np.deg2rad(0.20) ** 2,  # elevation variance
                np.deg2rad(0.20) ** 2,  # bearing variance
                30.0**2,  # range variance
            ]
        )
    )

    measurement_model = CartesianToElevationBearingRange(
        ndim_state=6,
        # x, y, z are at indices:
        #
        # 0 -> x
        # 2 -> y
        # 4 -> z
        mapping=(0, 2, 4),
        noise_covar=radar_noise,
    )

    # ==============================================================
    # 3. EKF UPDATER
    # ==============================================================

    updater = PDAUpdater(measurement_model=measurement_model)

    # ==============================================================
    # 4. PDA / JPDA ASSOCIATION
    # ==============================================================

    hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        # Expected clutter density.
        clutter_spatial_density=1e-9,
        # Probability that a real target produces a detection.
        prob_detect=0.95,
    )

    # associator = JPDA(hypothesiser=hypothesiser)
    associator = JPDAwithEHM2(hypothesiser=hypothesiser)

    return (
        associator,
        updater,
        measurement_model,
    )
