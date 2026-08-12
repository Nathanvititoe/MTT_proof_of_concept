import numpy as np
from datetime import datetime, timedelta

# CHAT GPT Generated Stone Soup example

# -------------------------------------------------------------------
# STONE SOUP IMPORTS
# -------------------------------------------------------------------

# Motion-model classes.
# ConstantVelocity represents motion with approximately constant velocity,
# while CombinedLinearGaussianTransitionModel lets us combine independent
# motion models for x, y, and z into one 3D target model.
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)

# Nonlinear radar measurement model.
# This converts a Cartesian target state [x, vx, y, vy, z, vz]
# into radar-style measurements:
#
#     [elevation, bearing, range]
#
# Because angles/range are nonlinear functions of x/y/z, an EKF updater
# is used later.
from stonesoup.models.measurement.nonlinear import (
    CartesianToElevationBearingRange,
)

# Stone Soup matrix/vector container types.
from stonesoup.types.array import CovarianceMatrix, StateVector

# A Detection represents one sensor measurement.
from stonesoup.types.detection import Detection

# GaussianState contains:
#   - state estimate
#   - covariance
#   - timestamp
from stonesoup.types.state import GaussianState

# Track stores the history of estimated states for one target.
from stonesoup.types.track import Track

# Kalman predictor predicts each track forward in time according
# to the motion model.
from stonesoup.predictor.kalman import KalmanPredictor

# Extended Kalman updater is used because the radar measurement
# relationship is nonlinear.
from stonesoup.updater.kalman import ExtendedKalmanUpdater

# PDAHypothesiser creates possible measurement-to-track associations
# and computes probabilities for those possibilities.
from stonesoup.hypothesiser.probability import PDAHypothesiser

# JPDA looks at the associations jointly across all tracks so that
# ambiguous measurements can contribute probabilistically to tracks.
from stonesoup.dataassociator.probability import JPDA


# ===================================================================
# 1. TARGET MOTION MODEL
# ===================================================================
#
# We define each aircraft state as:
#
#     [x, vx, y, vy, z, vz]
#
# where:
#
#     x, y, z    = position in metres
#     vx, vy, vz = velocity in metres/second
#
# Example:
#
#     [20000, 220, -5000, 40, 8500, 2]
#
# means:
#
#     x  = 20000 m
#     vx =   220 m/s
#     y  = -5000 m
#     vy =    40 m/s
#     z  =  8500 m
#     vz =     2 m/s
#
# Each ConstantVelocity model operates on one position/velocity pair:
#
#     [x, vx]
#     [y, vy]
#     [z, vz]
#
# The argument passed to ConstantVelocity is the process-noise
# diffusion coefficient. It represents how much deviation from ideal
# constant-velocity motion we expect.
#
# Larger values:
#     more maneuvering allowed
#     more uncertainty in prediction
#
# Smaller values:
#     assume smoother/straighter motion
#     more confidence in prediction
# ===================================================================

transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(1.0),   # x / vx motion uncertainty
    ConstantVelocity(1.0),   # y / vy motion uncertainty
    ConstantVelocity(0.5),   # z / vz motion uncertainty
])


# The predictor uses the motion model to propagate a track from its
# previous timestamp to the current measurement timestamp.
#
# Conceptually:
#
#     previous state
#          |
#          v
#     motion model
#          |
#          v
#     predicted state
#
# Example:
#
#     x_new ~= x_old + vx * dt
#
predictor = KalmanPredictor(transition_model)


# ===================================================================
# 2. RADAR MEASUREMENT MODEL
# ===================================================================
#
# Assume the radar reports:
#
#     [elevation, bearing, range]
#
# rather than directly reporting:
#
#     [x, y, z]
#
# The tracker state still uses Cartesian coordinates:
#
#     [x, vx, y, vy, z, vz]
#
# Therefore Stone Soup needs a mathematical measurement model that
# relates the Cartesian target state to what the radar measures.
#
#
#                  target
#                    *
#                   /|
#                  / |
#             range  | z
#                /   |
#               /    |
#            radar---+
#
#
# mapping=(0, 2, 4)
#
# tells Stone Soup where the POSITION values live inside the
# six-dimensional target state:
#
#     state[0] = x
#     state[2] = y
#     state[4] = z
#
# The velocity entries:
#
#     state[1] = vx
#     state[3] = vy
#     state[5] = vz
#
# are not directly measured by this radar model.
# ===================================================================


# Measurement noise covariance.
#
# This represents how uncertain the radar measurements are.
#
# Here we assume:
#
#     elevation standard deviation = 0.20 degrees
#     bearing standard deviation   = 0.20 degrees
#     range standard deviation     = 30 metres
#
# Covariance matrices contain VARIANCE, not standard deviation.
#
# Therefore every standard deviation must be squared.
#
# np.diag(...) creates:
#
#     [ elev_var       0          0      ]
#     [    0       bearing_var    0      ]
#     [    0           0       range_var ]
#
radar_noise = CovarianceMatrix(
    np.diag([
        np.deg2rad(0.20) ** 2,   # elevation variance [rad^2]
        np.deg2rad(0.20) ** 2,   # bearing variance [rad^2]
        30.0 ** 2,               # range variance [m^2]
    ])
)


# Build the nonlinear measurement model.
#
# ndim_state=6
#     The target state contains six variables:
#     [x, vx, y, vy, z, vz]
#
# mapping=(0, 2, 4)
#     Use x, y, z from the state to predict radar measurements.
#
# noise_covar=radar_noise
#     Describes expected radar measurement uncertainty.
#
measurement_model = CartesianToElevationBearingRange(
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar=radar_noise,
)


# Since converting Cartesian coordinates to:
#
#     elevation / bearing / range
#
# is nonlinear, we use an Extended Kalman Filter updater.
#
# The EKF locally linearizes the nonlinear measurement function
# around the predicted state.
#
updater = ExtendedKalmanUpdater(
    measurement_model=measurement_model
)


# ===================================================================
# 3. JPDA DATA ASSOCIATION
# ===================================================================
#
# The main challenge in multi-target tracking is determining which
# detection belongs to which track.
#
# Example:
#
#               detection A
#                    *
#                  /   \
#                 /     \
#            Track 1   Track 2
#                 \     /
#                  \   /
#                    *
#               detection B
#
# If the two aircraft are close together, there may not be an obvious
# one-to-one measurement assignment.
#
# JPDA = Joint Probabilistic Data Association
#
# Instead of immediately deciding:
#
#     detection A -> Track 1
#     detection B -> Track 2
#
# JPDA can estimate probabilities such as:
#
#                     Det A    Det B
#     Track 1          0.70     0.25
#     Track 2          0.25     0.70
#
# and update each track according to the weighted possibilities.
# ===================================================================


# PDAHypothesiser creates the possible association hypotheses between
# each predicted track and each measurement.
#
# It also calculates probabilities based on:
#
#     - predicted target location
#     - measurement uncertainty
#     - probability of detection
#     - expected clutter
#
hypothesiser = PDAHypothesiser(
    predictor=predictor,
    updater=updater,

    # Expected density of false detections / clutter.
    #
    # This value must ultimately match the units of your measurement
    # space and expected false-alarm environment.
    #
    # 1e-9 is only an example value here.
    clutter_spatial_density=1e-9,

    # Probability that an actual target produces a detection
    # during a scan.
    #
    # 0.95 means:
    #     approximately 95% chance of detection
    #     approximately 5% chance of missed detection
    prob_detect=0.95,
)


# JPDA combines the hypotheses across all tracks.
#
# This ensures that associations are treated jointly rather than
# independently.
#
# For example, one detection should not simply be assigned with
# probability 1.0 to multiple targets.
#
associator = JPDA(
    hypothesiser=hypothesiser
)


# ===================================================================
# 4. CREATE SOME EXISTING TRACKS
# ===================================================================
#
# For simplicity, this example starts with two tracks that already
# exist.
#
# In a complete tracking system, these would normally be created by a
# track initiator after receiving previously unassociated detections.
#
#
# Each track starts with:
#
#     estimated state
#     covariance
#     timestamp
#
# ===================================================================


# Starting time of the example.
t0 = datetime.now()


# -------------------------------------------------------------------
# TRACK 1
# -------------------------------------------------------------------

track1 = Track([
    GaussianState(

        # Initial estimate of aircraft state:
        #
        # [x, vx, y, vy, z, vz]
        #
        StateVector([
            20_000,   # x position [m]
            220,      # x velocity [m/s]

            -5_000,   # y position [m]
            40,       # y velocity [m/s]

            8_500,    # altitude / z position [m]
            2,        # vertical velocity [m/s]
        ]),

        # Initial uncertainty in the state estimate.
        #
        # Standard deviations:
        #
        #     x  = 500 m
        #     vx = 30 m/s
        #     y  = 500 m
        #     vy = 30 m/s
        #     z  = 200 m
        #     vz = 10 m/s
        #
        # Again, covariance contains variances, so all values are
        # squared.
        #
        CovarianceMatrix(
            np.diag([
                500 ** 2,   # x variance
                30 ** 2,    # vx variance

                500 ** 2,   # y variance
                30 ** 2,    # vy variance

                200 ** 2,   # z variance
                10 ** 2,    # vz variance
            ])
        ),

        timestamp=t0,
    )
])


# -------------------------------------------------------------------
# TRACK 2
# -------------------------------------------------------------------

track2 = Track([
    GaussianState(
        StateVector([
            22_000,   # x [m]
            -180,     # vx [m/s]

            4_000,    # y [m]
            -50,      # vy [m/s]

            8_700,    # z [m]
            -1,       # vz [m/s]
        ]),

        CovarianceMatrix(
            np.diag([
                500 ** 2,   # x variance
                30 ** 2,    # vx variance

                500 ** 2,   # y variance
                30 ** 2,    # vy variance

                200 ** 2,   # z variance
                10 ** 2,    # vz variance
            ])
        ),

        timestamp=t0,
    )
])


# Stone Soup expects a collection of current tracks.
#
# A Python set is commonly used here.
tracks = {track1, track2}


# ===================================================================
# 5. EXAMPLE RADAR DETECTIONS
# ===================================================================
#
# One second after our initial track states, the radar produces
# another scan.
#
# Each detection contains:
#
#     [elevation, bearing, range]
#
# IMPORTANT:
#
#     elevation = radians
#     bearing   = radians
#     range     = metres
#
# ===================================================================


# Current radar scan occurs one second after the previous state.
timestamp = t0 + timedelta(seconds=1)


detections = {

    # ---------------------------------------------------------------
    # Detection 1
    # ---------------------------------------------------------------
    Detection(
        StateVector([
            np.deg2rad(22.5),    # elevation [rad]
            np.deg2rad(-12.0),   # bearing [rad]
            22_200,              # range [m]
        ]),

        # Timestamp tells the predictor how far forward it must
        # propagate the tracks.
        timestamp=timestamp,

        # This tells Stone Soup what the three measurement values mean.
        measurement_model=measurement_model,
    ),


    # ---------------------------------------------------------------
    # Detection 2
    # ---------------------------------------------------------------
    Detection(
        StateVector([
            np.deg2rad(21.0),    # elevation [rad]
            np.deg2rad(10.5),    # bearing [rad]
            23_100,              # range [m]
        ]),

        timestamp=timestamp,
        measurement_model=measurement_model,
    ),
}


# ===================================================================
# 6. RUN JPDA DATA ASSOCIATION
# ===================================================================
#
# This is where Stone Soup determines how likely each measurement is
# to belong to each existing track.
#
# Internally, this broadly involves:
#
#     Track states at t0
#             |
#             v
#        prediction to t1
#             |
#             v
#      compare predictions
#      against detections
#             |
#             v
#      association likelihoods
#             |
#             v
#          JPDA
#
#
# The returned "hypotheses" object contains the probabilistic
# measurement hypotheses for each track.
# ===================================================================

hypotheses = associator.associate(
    tracks,
    detections,
    timestamp,
)


# ===================================================================
# 7. UPDATE EACH TRACK
# ===================================================================
#
# JPDA gives each track a collection of possible measurement
# hypotheses.
#
# For example:
#
# Track 1:
#
#     no detection     probability = 0.05
#     detection 1      probability = 0.75
#     detection 2      probability = 0.20
#
#
# The updater combines these hypotheses into a single posterior
# state estimate.
#
#
#             predicted track
#                    |
#                    v
#             JPDA hypotheses
#                    |
#                    v
#               EKF update
#                    |
#                    v
#            posterior state
#
# ===================================================================

for track in tracks:

    # Retrieve the hypotheses associated with this particular track.
    hypothesis = hypotheses[track]

    # Perform the probabilistic EKF update.
    #
    # The result is the new estimated state at the current timestamp.
    posterior = updater.update(hypothesis)

    # Add the newly estimated state to this target's track history.
    #
    # A Track contains all states over time rather than only the
    # current state.
    track.append(posterior)


# ===================================================================
# 8. PRINT THE UPDATED TRACK ESTIMATES
# ===================================================================
#
# After prediction, association, and measurement update, each track's
# latest state should represent our best estimate of that aircraft's:
#
#     position
#     velocity
#
# ===================================================================

for i, track in enumerate(tracks):

    # track.state is the latest state in the track.
    state = track.state.state_vector

    # Extract individual components.
    #
    # State ordering is still:
    #
    #     [x, vx, y, vy, z, vz]
    #
    x = float(state[0])
    vx = float(state[1])

    y = float(state[2])
    vy = float(state[3])

    z = float(state[4])
    vz = float(state[5])

    print(f"\nTrack {i}")

    print(
        f"Position: "
        f"({x:.1f}, {y:.1f}, {z:.1f}) m"
    )

    print(
        f"Velocity: "
        f"({vx:.1f}, {vy:.1f}, {vz:.1f}) m/s"
    )