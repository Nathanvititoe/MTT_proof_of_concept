"""
test_tracks_simple.py

Moderate 60-second Stone Soup JPDA test scenario.

Purpose
-------
This is intended as a more realistic baseline than the 20-target stress test.

Scenario:
- 8 physical airborne targets
- 60 seconds at 1 Hz
- 2 intentionally ambiguous close/crossing encounters
- Remaining targets stay reasonably separated
- 3D Cartesian truth
- Radar measurements:
      [elevation, azimuth, range]
- Continuous/unwrapped azimuth history retained per target
- Gaussian measurement noise
- Occasional missed detections
- Light clutter
- Reproducible random seed

IMPORTANT
---------
If you are still using Stone Soup's standard
CartesianToElevationBearingRange model, use:

    pass_unwrapped_to_tracker=False

The test data itself remains unwrapped for diagnostics, but the value sent
to the standard Stone Soup model is wrapped to [-pi, pi).

Once you implement your custom unwrapped-bearing measurement model, use:

    pass_unwrapped_to_tracker=True
"""

import numpy as np
from datetime import datetime, timedelta

from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track


# ======================================================================
# CONFIGURATION
# ======================================================================

NUM_TARGETS = 8
DURATION_SECONDS = 60
SCAN_PERIOD_SECONDS = 1.0

ELEVATION_NOISE_DEG = 0.15
AZIMUTH_NOISE_DEG = 0.15
RANGE_NOISE_M = 25.0

# Slightly cleaner than the stress test.
DETECTION_PROBABILITY = 0.985

# One false detection per scan.
CLUTTER_PER_SCAN = 1

RANDOM_SEED = 21


# ======================================================================
# ANGLE HELPERS
# ======================================================================

def wrap_to_pi(angle_rad):
    """Wrap angle to [-pi, pi)."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_near(wrapped_angle_rad, previous_unwrapped_rad):
    """
    Move a wrapped angle onto the 2*pi branch nearest the previous
    continuous angle.
    """

    if previous_unwrapped_rad is None:
        return wrapped_angle_rad

    return wrapped_angle_rad + 2.0 * np.pi * np.round(
        (previous_unwrapped_rad - wrapped_angle_rad) / (2.0 * np.pi)
    )


# ======================================================================
# TRUTH TARGETS
# ======================================================================

def build_simple_truth_targets():
    """
    Build eight aircraft.

    Targets T01/T02 are designed for a close encounter near t ~= 25 s.
    Targets T03/T04 are designed for another close encounter near t ~= 42 s.
    T05-T08 remain comparatively well separated.

    State format:
        [x, vx, y, vy, z, vz]
    """

    targets = [

        # --------------------------------------------------------------
        # Pair 1: ambiguous encounter near 25 seconds
        # --------------------------------------------------------------

        {
            "id": "T01",
            "initial_state": np.array([
                -22_000.0,   # x
                90.0,        # vx
                -4_000.0,    # y
                160.0,       # vy
                8_200.0,     # z
                1.0,         # vz
            ])
        },

        {
            "id": "T02",
            "initial_state": np.array([
                -21_800.0,
                82.0,
                4_100.0,
                -162.0,
                8_260.0,
                -0.7,
            ])
        },


        # --------------------------------------------------------------
        # Pair 2: ambiguous encounter later in the minute
        # --------------------------------------------------------------

        {
            "id": "T03",
            "initial_state": np.array([
                -31_000.0,
                110.0,
                -6_300.0,
                150.0,
                9_600.0,
                -0.5,
            ])
        },

        {
            "id": "T04",
            "initial_state": np.array([
                -30_750.0,
                104.0,
                6_400.0,
                -152.0,
                9_680.0,
                0.6,
            ])
        },


        # --------------------------------------------------------------
        # Four relatively unambiguous aircraft
        # --------------------------------------------------------------

        {
            "id": "T05",
            "initial_state": np.array([
                18_000.0,
                180.0,
                -13_000.0,
                45.0,
                7_400.0,
                2.0,
            ])
        },

        {
            "id": "T06",
            "initial_state": np.array([
                28_000.0,
                -95.0,
                14_000.0,
                -35.0,
                10_200.0,
                -1.2,
            ])
        },

        {
            "id": "T07",
            "initial_state": np.array([
                -38_000.0,
                120.0,
                -15_000.0,
                30.0,
                11_500.0,
                0.3,
            ])
        },

        {
            "id": "T08",
            "initial_state": np.array([
                34_000.0,
                -130.0,
                -9_000.0,
                65.0,
                8_900.0,
                1.1,
            ])
        },
    ]

    return targets


def truth_state_at_time(initial_state, elapsed_seconds):
    """
    Constant-velocity propagation of the truth state.
    """

    x0, vx, y0, vy, z0, vz = initial_state

    return np.array([
        x0 + vx * elapsed_seconds,
        vx,
        y0 + vy * elapsed_seconds,
        vy,
        z0 + vz * elapsed_seconds,
        vz,
    ])


# ======================================================================
# RADAR MODEL HELPER
# ======================================================================

def xyz_to_radar(x, y, z):
    """
    Convert radar-relative Cartesian XYZ to:
        elevation
        wrapped azimuth
        slant range
    """

    horizontal_range = np.hypot(x, y)
    slant_range = np.sqrt(x*x + y*y + z*z)

    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, horizontal_range)

    return elevation, azimuth, slant_range


# ======================================================================
# INITIAL STONE SOUP TRACKS
# ======================================================================

def create_simple_test_tracks(
    start_time=None,
    position_sigma_m=250.0,
    velocity_sigma_mps=20.0,
):
    """
    Create eight initial Stone Soup tracks.

    Returns
    -------
    tracks
    track_ids
    start_time
    truth_targets
    """

    if start_time is None:
        start_time = datetime.now()

    truth_targets = build_simple_truth_targets()

    initial_covariance = CovarianceMatrix(
        np.diag([
            position_sigma_m ** 2,
            velocity_sigma_mps ** 2,

            position_sigma_m ** 2,
            velocity_sigma_mps ** 2,

            position_sigma_m ** 2,
            velocity_sigma_mps ** 2,
        ])
    )

    tracks = set()
    track_ids = {}

    for target in truth_targets:

        track = Track([
            GaussianState(
                StateVector(target["initial_state"]),
                initial_covariance.copy(),
                timestamp=start_time,
            )
        ])

        tracks.add(track)
        track_ids[track] = target["id"]

    return tracks, track_ids, start_time, truth_targets


# ======================================================================
# GENERATE ONE MINUTE OF SCANS
# ======================================================================

def generate_simple_one_minute_scans(
    measurement_model,
    start_time,
    truth_targets=None,
    pass_unwrapped_to_tracker=False,
    detection_probability=DETECTION_PROBABILITY,
    clutter_per_scan=CLUTTER_PER_SCAN,
    random_seed=RANDOM_SEED,
):
    """
    Generate 60 scans containing the eight targets.

    Raw unwrapped azimuth is retained in the diagnostic records even when
    a wrapped angle is passed into Stone Soup.
    """

    if truth_targets is None:
        truth_targets = build_simple_truth_targets()

    rng = np.random.default_rng(random_seed)

    previous_unwrapped = {
        target["id"]: None
        for target in truth_targets
    }

    scans = []

    for scan_index in range(1, DURATION_SECONDS + 1):

        elapsed = scan_index * SCAN_PERIOD_SECONDS
        timestamp = start_time + timedelta(seconds=elapsed)

        detections = set()
        records = []

        # --------------------------------------------------------------
        # Real target measurements
        # --------------------------------------------------------------

        for target in truth_targets:

            target_id = target["id"]

            truth = truth_state_at_time(
                target["initial_state"],
                elapsed,
            )

            x = truth[0]
            y = truth[2]
            z = truth[4]

            elevation_true, wrapped_azimuth_true, range_true = xyz_to_radar(
                x,
                y,
                z,
            )

            unwrapped_azimuth_true = unwrap_near(
                wrapped_azimuth_true,
                previous_unwrapped[target_id],
            )

            previous_unwrapped[target_id] = unwrapped_azimuth_true

            # Simulate occasional missing reports.
            detected = rng.random() <= detection_probability

            if not detected:

                records.append({
                    "kind": "missed",
                    "target_id": target_id,
                    "elapsed_seconds": elapsed,
                    "truth_state": truth.copy(),
                    "azimuth_unwrapped_true_rad": unwrapped_azimuth_true,
                })

                continue


            elevation_measured = (
                elevation_true
                + np.deg2rad(
                    rng.normal(0.0, ELEVATION_NOISE_DEG)
                )
            )

            unwrapped_azimuth_measured = (
                unwrapped_azimuth_true
                + np.deg2rad(
                    rng.normal(0.0, AZIMUTH_NOISE_DEG)
                )
            )

            range_measured = (
                range_true
                + rng.normal(0.0, RANGE_NOISE_M)
            )


            if pass_unwrapped_to_tracker:
                tracker_azimuth = unwrapped_azimuth_measured
            else:
                tracker_azimuth = wrap_to_pi(
                    unwrapped_azimuth_measured
                )


            detection = Detection(
                StateVector([
                    elevation_measured,
                    tracker_azimuth,
                    range_measured,
                ]),
                timestamp=timestamp,
                measurement_model=measurement_model,
                metadata={
                    "target_id": target_id,
                    "is_clutter": False,
                    "unwrapped_azimuth_rad": unwrapped_azimuth_measured,
                },
            )

            detections.add(detection)


            records.append({
                "kind": "target",
                "target_id": target_id,
                "elapsed_seconds": elapsed,

                "truth_state": truth.copy(),

                "elevation_true_rad": elevation_true,
                "azimuth_wrapped_true_rad": wrapped_azimuth_true,
                "azimuth_unwrapped_true_rad": unwrapped_azimuth_true,
                "range_true_m": range_true,

                "elevation_measured_rad": elevation_measured,
                "azimuth_unwrapped_measured_rad": unwrapped_azimuth_measured,
                "azimuth_tracker_rad": tracker_azimuth,
                "range_measured_m": range_measured,
            })


        # --------------------------------------------------------------
        # Light clutter
        # --------------------------------------------------------------

        for _ in range(clutter_per_scan):

            elevation = np.deg2rad(
                rng.uniform(5.0, 30.0)
            )

            # Deliberately place clutter in a realistic radar sector,
            # but not always directly on top of the target cluster.
            raw_azimuth = np.deg2rad(
                rng.uniform(-180.0, 180.0)
            )

            range_m = rng.uniform(
                16_000.0,
                42_000.0,
            )


            tracker_azimuth = (
                raw_azimuth
                if pass_unwrapped_to_tracker
                else wrap_to_pi(raw_azimuth)
            )


            detection = Detection(
                StateVector([
                    elevation,
                    tracker_azimuth,
                    range_m,
                ]),
                timestamp=timestamp,
                measurement_model=measurement_model,
                metadata={
                    "target_id": None,
                    "is_clutter": True,
                    "unwrapped_azimuth_rad": raw_azimuth,
                },
            )

            detections.add(detection)


            records.append({
                "kind": "clutter",
                "target_id": None,
                "elapsed_seconds": elapsed,
                "elevation_measured_rad": elevation,
                "azimuth_unwrapped_measured_rad": raw_azimuth,
                "azimuth_tracker_rad": tracker_azimuth,
                "range_measured_m": range_m,
            })


        scans.append({
            "scan_index": scan_index,
            "elapsed_seconds": elapsed,
            "timestamp": timestamp,
            "detections": detections,
            "records": records,
        })


    return scans


# ======================================================================
# RUN TEST
# ======================================================================

def run_simple_one_minute_test(
    tracks,
    associator,
    updater,
    measurement_model,
    start_time,
    truth_targets=None,
    track_ids=None,
    pass_unwrapped_to_tracker=False,
    verbose=True,
):
    """
    Process the complete moderate-complexity 60-second scenario.
    """

    import time

    scans = generate_simple_one_minute_scans(
        measurement_model=measurement_model,
        start_time=start_time,
        truth_targets=truth_targets,
        pass_unwrapped_to_tracker=pass_unwrapped_to_tracker,
    )

    scan_times = []

    if verbose:
        print("=" * 76)
        print("JPDA MODERATE ONE-MINUTE TEST")
        print("=" * 76)
        print(f"Physical targets:       {NUM_TARGETS}")
        print(f"Duration:               {DURATION_SECONDS} seconds")
        print(f"Scan period:            {SCAN_PERIOD_SECONDS:.1f} second")
        print(f"Clutter per scan:       {CLUTTER_PER_SCAN}")
        print(f"Detection probability:  {DETECTION_PROBABILITY:.3f}")
        print(f"Unwrapped -> tracker:   {pass_unwrapped_to_tracker}")
        print()


    for scan in scans:

        scan_start = time.perf_counter()

        timestamp = scan["timestamp"]
        detections = scan["detections"]


        # JPDA association.
        association_start = time.perf_counter()

        hypotheses = associator.associate(
            tracks,
            detections,
            timestamp,
        )

        association_time = (
            time.perf_counter()
            - association_start
        )


        # Track updates.
        update_start = time.perf_counter()

        for track in tracks:

            posterior = updater.update(
                hypotheses[track]
            )

            track.append(posterior)

        update_time = (
            time.perf_counter()
            - update_start
        )


        total_scan_time = (
            time.perf_counter()
            - scan_start
        )

        scan_times.append(total_scan_time)


        if verbose:

            real_count = sum(
                record["kind"] == "target"
                for record in scan["records"]
            )

            missed_count = sum(
                record["kind"] == "missed"
                for record in scan["records"]
            )

            clutter_count = sum(
                record["kind"] == "clutter"
                for record in scan["records"]
            )


            print(
                f"t={scan['elapsed_seconds']:5.1f}s | "
                f"dets={len(detections):2d} | "
                f"real={real_count:2d} | "
                f"missed={missed_count:1d} | "
                f"clutter={clutter_count:1d} | "
                f"assoc={association_time:7.4f}s | "
                f"update={update_time:7.4f}s | "
                f"total={total_scan_time:7.4f}s"
            )


    scan_times = np.asarray(scan_times)


    if verbose:

        print()
        print("=" * 76)
        print("PERFORMANCE")
        print("=" * 76)

        print(
            f"Total processing time:  "
            f"{scan_times.sum():.3f} s"
        )

        print(
            f"Average scan time:      "
            f"{scan_times.mean():.4f} s"
        )

        print(
            f"Worst scan time:        "
            f"{scan_times.max():.4f} s"
        )

        print(
            f"95th percentile:        "
            f"{np.percentile(scan_times, 95):.4f} s"
        )

        print(
            f"Scans over 1 second:    "
            f"{np.sum(scan_times > 1.0)} / {len(scan_times)}"
        )

        print(
            f"Real-time speed factor: "
            f"{DURATION_SECONDS / scan_times.sum():.2f}x"
        )


        print()
        print("FINAL TRACK STATES")
        print("-" * 76)

        for index, track in enumerate(tracks, start=1):

            state = track.state.state_vector

            if track_ids is not None:
                label = track_ids.get(
                    track,
                    f"Track {index:02d}",
                )
            else:
                label = f"Track {index:02d}"


            print(
                f"{label:>8}: "
                f"xyz=("
                f"{float(state[0]):10.1f}, "
                f"{float(state[2]):10.1f}, "
                f"{float(state[4]):8.1f}) m   "
                f"v=("
                f"{float(state[1]):7.1f}, "
                f"{float(state[3]):7.1f}, "
                f"{float(state[5]):6.1f}) m/s"
            )


    return tracks, scans, scan_times


# ======================================================================
# DIAGNOSTIC: UNWRAPPED AZIMUTH
# ======================================================================

def print_simple_unwrapped_samples(
    scans,
    target_id="T01",
    every_n_seconds=5,
):
    """
    Print raw continuous azimuth values for one target.
    """

    print()
    print(f"Unwrapped azimuth samples for {target_id}")
    print("-" * 55)

    for scan in scans:

        second = int(scan["elapsed_seconds"])

        if second % every_n_seconds != 0:
            continue


        for record in scan["records"]:

            if (
                record.get("kind") == "target"
                and record.get("target_id") == target_id
            ):

                unwrapped_deg = np.rad2deg(
                    record[
                        "azimuth_unwrapped_measured_rad"
                    ]
                )

                tracker_deg = np.rad2deg(
                    record["azimuth_tracker_rad"]
                )


                print(
                    f"t={second:02d}s | "
                    f"unwrapped={unwrapped_deg:9.3f} deg | "
                    f"tracker={tracker_deg:9.3f} deg"
                )

                break