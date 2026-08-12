"""
test_tracks.py

Synthetic one-minute stress test for a Stone Soup JPDA multi-target tracker.

Scenario
--------
- 20 physical airborne targets
- 60 seconds at 1 Hz
- 10 pairs of targets arranged to pass close to each other near t=30 s
- 3D Cartesian truth: [x, vx, y, vy, z, vz]
- Radar measurements: [elevation, azimuth, range]
- Azimuth is explicitly UNWRAPPED over time for each physical target
- Gaussian measurement noise
- Optional missed detections
- Optional clutter
- Reproducible random seed

IMPORTANT
---------
Stone Soup's standard CartesianToElevationBearingRange model is based on
wrapped angular measurements. If your measurement_model is still the
standard Stone Soup model, set:

    pass_unwrapped_to_tracker=False

This keeps the generated/raw azimuth history unwrapped for inspection,
but wraps the angle only at the interface to the standard measurement
model.

Once you replace your tracker measurement model with your custom
unwrapped-bearing model, set:

    pass_unwrapped_to_tracker=True

to pass the actual continuous azimuth directly into JPDA.
"""
import time
import numpy as np
from datetime import datetime, timedelta

from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track


# ======================================================================
# TEST CONFIGURATION
# ======================================================================

NUM_TARGETS = 20
DURATION_SECONDS = 60
SCAN_PERIOD_SECONDS = 1.0

# Radar measurement noise standard deviations.
ELEVATION_NOISE_DEG = 0.20
AZIMUTH_NOISE_DEG = 0.20
RANGE_NOISE_M = 30.0

# Probability that a real target produces a measurement on a scan.
DETECTION_PROBABILITY = 0.97

# False detections per scan.
CLUTTER_PER_SCAN = 2

# Fixed random seed makes the test repeatable.
RANDOM_SEED = 7


# ======================================================================
# ANGLE HELPERS
# ======================================================================

def wrap_to_pi(angle_rad):
    """
    Wrap a numeric angle to [-pi, pi).
    """
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_near(wrapped_angle_rad, previous_unwrapped_rad):
    """
    Select the 2*pi branch of wrapped_angle_rad nearest to the previous
    unwrapped angle.

    This preserves a continuous azimuth history such as:

        177, 179, 181, 183 deg

    instead of:

        177, 179, -179, -177 deg
    """
    if previous_unwrapped_rad is None:
        return wrapped_angle_rad

    return wrapped_angle_rad + 2.0 * np.pi * np.round(
        (previous_unwrapped_rad - wrapped_angle_rad) / (2.0 * np.pi)
    )


# ======================================================================
# TRUTH SCENARIO
# ======================================================================

def build_truth_targets():
    """
    Create 20 aircraft arranged as 10 crossing/near-crossing pairs.

    Each pair approaches nearly the same location at approximately
    t = 30 seconds. This intentionally creates association ambiguity.

    The targets are placed mostly on the negative-X side of the radar.
    Their Y positions pass through zero around the middle of the test,
    which drives the wrapped azimuth through the +/-180 degree boundary.

    Returns
    -------
    list[dict]
        Each dictionary contains:
            id
            initial_state = np.array([x, vx, y, vy, z, vz])
    """

    targets = []

    # Ten pairs -> twenty physical targets.
    for pair in range(10):

        # Spread the crossing regions in X and altitude so all 20 targets
        # are not identical, while keeping each pair deliberately close.
        crossing_x = -20_000.0 - pair * 1_250.0
        altitude = 7_500.0 + pair * 220.0

        # Small offsets keep pair members close, but not perfectly equal.
        pair_x_offset = 60.0
        pair_z_offset = 30.0

        # Different pair speeds make the scenario less artificial.
        lateral_speed = 140.0 + pair * 7.0

        # Each target is chosen so y ~= 0 near t = 30 seconds.
        y_distance = lateral_speed * 30.0

        # Pair member A moves from negative Y toward positive Y.
        state_a = np.array([
            crossing_x - pair_x_offset,   # x
            35.0 + pair * 2.0,            # vx
            -y_distance,                   # y
            lateral_speed,                 # vy
            altitude - pair_z_offset,      # z
            0.5 + 0.10 * pair,             # vz
        ], dtype=float)

        # Pair member B moves from positive Y toward negative Y.
        state_b = np.array([
            crossing_x + pair_x_offset,   # x
            32.0 + pair * 2.0,            # vx
            y_distance,                    # y
            -lateral_speed,                # vy
            altitude + pair_z_offset,      # z
            -0.4 - 0.08 * pair,            # vz
        ], dtype=float)

        targets.append({
            "id": f"T{2 * pair + 1:02d}",
            "initial_state": state_a,
        })

        targets.append({
            "id": f"T{2 * pair + 2:02d}",
            "initial_state": state_b,
        })

    return targets


def truth_state_at_time(initial_state, elapsed_seconds):
    """
    Constant-velocity truth propagation.

    State ordering:
        [x, vx, y, vy, z, vz]
    """
    x0, vx, y0, vy, z0, vz = initial_state

    return np.array([
        x0 + vx * elapsed_seconds,
        vx,
        y0 + vy * elapsed_seconds,
        vy,
        z0 + vz * elapsed_seconds,
        vz,
    ], dtype=float)


# ======================================================================
# RADAR CONVERSION
# ======================================================================

def xyz_to_radar(x, y, z):
    """
    Convert radar-relative Cartesian XYZ into:

        elevation [rad]
        wrapped azimuth [rad]
        slant range [m]

    The radar is assumed to be at (0, 0, 0).
    """

    horizontal_range = np.hypot(x, y)
    slant_range = np.sqrt(x * x + y * y + z * z)

    wrapped_azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, horizontal_range)

    return elevation, wrapped_azimuth, slant_range


# ======================================================================
# CREATE 20 STONE SOUP TRACKS
# ======================================================================

def create_test_tracks(
    start_time=None,
    position_sigma_m=300.0,
    velocity_sigma_mps=25.0,
):
    """
    Create 20 initial Stone Soup Track objects from the synthetic truth.

    Returns
    -------
    tracks : set[Track]
    track_ids : dict
        Maps Track object -> physical truth ID.
    start_time : datetime
    truth_targets : list[dict]
    """

    if start_time is None:
        start_time = datetime.now()

    truth_targets = build_truth_targets()

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
# BUILD 60 SECONDS OF RADAR SCANS
# ======================================================================

def generate_one_minute_scans(
    measurement_model,
    start_time,
    truth_targets=None,
    pass_unwrapped_to_tracker=False,
    detection_probability=DETECTION_PROBABILITY,
    clutter_per_scan=CLUTTER_PER_SCAN,
    random_seed=RANDOM_SEED,
):
    """
    Generate 60 one-second radar scans for 20 physical targets.

    Parameters
    ----------
    measurement_model
        Stone Soup measurement model attached to each Detection.

    start_time : datetime
        Timestamp corresponding to t=0.

    truth_targets : list[dict], optional
        Output of build_truth_targets(). If omitted, a new scenario is made.

    pass_unwrapped_to_tracker : bool
        False:
            Keep unwrapped azimuth in the returned diagnostic records, but
            wrap the azimuth passed into Stone Soup. Use this with the
            standard CartesianToElevationBearingRange model.

        True:
            Pass continuous/unwrapped azimuth directly into Stone Soup.
            Use this only after your tracker has a custom unwrapped-bearing
            measurement/innovation model.

    Returns
    -------
    scans : list[dict]

        Each scan contains:
            {
                "timestamp": datetime,
                "detections": set[Detection],
                "records": list[dict]
            }

        records preserve the truth and raw unwrapped values for plotting
        and validation.
    """

    if truth_targets is None:
        truth_targets = build_truth_targets()

    rng = np.random.default_rng(random_seed)

    # One unwrapped azimuth memory per physical target.
    previous_unwrapped = {
        target["id"]: None
        for target in truth_targets
    }

    scans = []

    for scan_index in range(1, DURATION_SECONDS + 1):

        elapsed = scan_index * SCAN_PERIOD_SECONDS

        timestamp = start_time + timedelta(
            seconds=elapsed
        )

        detections = set()
        records = []

        # ==============================================================
        # TRUE TARGET DETECTIONS
        # ==============================================================

        for target in truth_targets:

            target_id = target["id"]

            truth = truth_state_at_time(
                target["initial_state"],
                elapsed,
            )

            x = truth[0]
            y = truth[2]
            z = truth[4]

            elevation_true, azimuth_wrapped_true, range_true = xyz_to_radar(
                x, y, z
            )

            # Convert wrapped geometric bearing into a continuous,
            # target-specific azimuth history.
            azimuth_unwrapped_true = unwrap_near(
                azimuth_wrapped_true,
                previous_unwrapped[target_id],
            )

            previous_unwrapped[target_id] = azimuth_unwrapped_true

            # Simulate a missed detection.
            detected = rng.random() <= detection_probability

            if not detected:
                records.append({
                    "kind": "missed",
                    "target_id": target_id,
                    "elapsed_seconds": elapsed,
                    "truth_state": truth.copy(),
                    "elevation_true_rad": elevation_true,
                    "azimuth_wrapped_true_rad": azimuth_wrapped_true,
                    "azimuth_unwrapped_true_rad": azimuth_unwrapped_true,
                    "range_true_m": range_true,
                })
                continue

            # Add measurement noise.
            elevation_measured = (
                elevation_true
                + np.deg2rad(
                    rng.normal(0.0, ELEVATION_NOISE_DEG)
                )
            )

            azimuth_unwrapped_measured = (
                azimuth_unwrapped_true
                + np.deg2rad(
                    rng.normal(0.0, AZIMUTH_NOISE_DEG)
                )
            )

            range_measured = (
                range_true
                + rng.normal(0.0, RANGE_NOISE_M)
            )

            if pass_unwrapped_to_tracker:
                tracker_azimuth = azimuth_unwrapped_measured
            else:
                # Compatibility mode for Stone Soup's standard wrapped
                # elevation/bearing/range model.
                tracker_azimuth = wrap_to_pi(
                    azimuth_unwrapped_measured
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
                    "unwrapped_azimuth_rad": azimuth_unwrapped_measured,
                },
            )

            detections.add(detection)

            records.append({
                "kind": "target",
                "target_id": target_id,
                "elapsed_seconds": elapsed,
                "truth_state": truth.copy(),

                "elevation_true_rad": elevation_true,
                "azimuth_wrapped_true_rad": azimuth_wrapped_true,
                "azimuth_unwrapped_true_rad": azimuth_unwrapped_true,
                "range_true_m": range_true,

                "elevation_measured_rad": elevation_measured,
                "azimuth_unwrapped_measured_rad": azimuth_unwrapped_measured,
                "azimuth_tracker_rad": tracker_azimuth,
                "range_measured_m": range_measured,
            })

        # ==============================================================
        # CLUTTER / FALSE DETECTIONS
        # ==============================================================

        for clutter_index in range(clutter_per_scan):

            elevation = np.deg2rad(
                rng.uniform(5.0, 35.0)
            )

            # Generate false measurements around the same difficult
            # +/-180-degree azimuth region occupied by the true targets.
            raw_azimuth_deg = rng.uniform(
                165.0,
                195.0,
            )

            raw_azimuth_rad = np.deg2rad(
                raw_azimuth_deg
            )

            if pass_unwrapped_to_tracker:
                tracker_azimuth = raw_azimuth_rad
            else:
                tracker_azimuth = wrap_to_pi(
                    raw_azimuth_rad
                )

            range_m = rng.uniform(
                18_000.0,
                38_000.0,
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
                    "unwrapped_azimuth_rad": raw_azimuth_rad,
                },
            )

            detections.add(detection)

            records.append({
                "kind": "clutter",
                "target_id": None,
                "elapsed_seconds": elapsed,
                "elevation_measured_rad": elevation,
                "azimuth_unwrapped_measured_rad": raw_azimuth_rad,
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
# RUN THE FULL ONE-MINUTE JPDA TEST
# ======================================================================

def run_one_minute_test(
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
    Generate and process the complete 60-second stress test.

    Each second:
        1. Generate detections for 20 physical targets.
        2. Add misses and clutter.
        3. Give ALL detections at that timestamp to JPDA.
        4. Update ALL existing Stone Soup tracks.
        5. Append each posterior to its Track history.

    Returns
    -------
    tracks
        Updated Stone Soup track set containing 60 seconds of history.

    scans
        Complete generated scan records for validation and plotting.
    """

    scans = generate_one_minute_scans(
        measurement_model=measurement_model,
        start_time=start_time,
        truth_targets=truth_targets,
        pass_unwrapped_to_tracker=pass_unwrapped_to_tracker,
    )

    if verbose:
        print("=" * 72)
        print("JPDA ONE-MINUTE STRESS TEST")
        print("=" * 72)
        print(f"Physical targets:       {NUM_TARGETS}")
        print(f"Duration:               {DURATION_SECONDS} seconds")
        print(f"Scan period:            {SCAN_PERIOD_SECONDS:.1f} second")
        print(f"Initial tracker tracks: {len(tracks)}")
        print(f"Unwrapped -> tracker:   {pass_unwrapped_to_tracker}")
        print()

    scan_times = []
    for scan in scans:
        scan_start = time.perf_counter()
        timestamp = scan["timestamp"]
        detections = scan["detections"]

        # --------------------------------------------------------------
        # JPDA association of all current tracks against all detections
        # at this timestamp.
        # --------------------------------------------------------------
        hypotheses = associator.associate(
            tracks,
            detections,
            timestamp,
        )

        association_time = time.perf_counter() - scan_start
        update_start = time.perf_counter()

        # --------------------------------------------------------------
        # Probability-weighted update for every track.
        #
        # The updater must be PDAUpdater (or another updater capable of
        # consuming JPDA's MultipleHypothesis output), not a plain
        # ExtendedKalmanUpdater.
        # --------------------------------------------------------------
        for track in tracks:

            posterior = updater.update(
                hypotheses[track]
            )

            track.append(posterior)

        update_time = time.perf_counter() - update_start
        total_scan_time = time.perf_counter() - scan_start
        scan_times.append(total_scan_time)

        if verbose:
            target_detections = sum(
                record["kind"] == "target"
                for record in scan["records"]
            )

            missed = sum(
                record["kind"] == "missed"
                for record in scan["records"]
            )

            clutter = sum(
                record["kind"] == "clutter"
                for record in scan["records"]
            )

            print(
                f"t={scan['elapsed_seconds']:5.1f}s | "
                f"detections={len(detections):2d} | "
                f"association={association_time:.3f}s | "
                f"update={update_time:.3f}s | "
                f"total={total_scan_time:.3f}s"
            )

    if verbose:
        print()
        print("Final track states:")
        print("-" * 72)

        for index, track in enumerate(tracks, start=1):

            state = track.state.state_vector

            label = (
                track_ids.get(track, f"Track {index:02d}")
                if track_ids is not None
                else f"Track {index:02d}"
            )

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
    scan_times = np.asarray(scan_times)

    print("\n=== PERFORMANCE ===")

    print(
        f"Total processing time: "
        f"{scan_times.sum():.3f} s"
    )

    print(
        f"Average scan time: "
        f"{scan_times.mean():.3f} s"
    )

    print(
        f"Worst scan time: "
        f"{scan_times.max():.3f} s"
    )

    print(
        f"Scans over 1 second: "
        f"{np.sum(scan_times > 1.0)} / {len(scan_times)}"
    )
    
    return tracks, scans, scan_times


# ======================================================================
# OPTIONAL DIAGNOSTIC OUTPUT
# ======================================================================

def print_unwrapped_azimuth_samples(scans, target_id="T01", every_n_seconds=5):
    """
    Print a target's continuous/unwrapped azimuth history.

    Useful for confirming that the generated test data actually crosses
    the +/-180 degree discontinuity without wrapping.
    """

    print()
    print(f"Unwrapped azimuth samples for {target_id}")
    print("-" * 50)

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
                    record["azimuth_unwrapped_measured_rad"]
                )

                tracker_deg = np.rad2deg(
                    record["azimuth_tracker_rad"]
                )

                print(
                    f"t={second:02d}s  "
                    f"unwrapped={unwrapped_deg:9.3f} deg   "
                    f"sent_to_tracker={tracker_deg:9.3f} deg"
                )

                break


# ======================================================================
# EXAMPLE MAIN
# ======================================================================
#
# Normally main.py should import these functions and provide your actual
# associator, updater, and measurement_model.
#
# Example:
#
#   from stone_soup_simple import (
#       associator,
#       updater,
#       measurement_model,
#   )
#
#   from test_tracks import (
#       create_test_tracks,
#       run_one_minute_test,
#       print_unwrapped_azimuth_samples,
#   )
#
#   tracks, track_ids, start_time, truth_targets = create_test_tracks()
#
#   tracks, scans = run_one_minute_test(
#       tracks=tracks,
#       associator=associator,
#       updater=updater,
#       measurement_model=measurement_model,
#       start_time=start_time,
#       truth_targets=truth_targets,
#       track_ids=track_ids,
#
#       # False while using standard CartesianToElevationBearingRange.
#       # Change to True after installing your custom unwrapped model.
#       pass_unwrapped_to_tracker=False,
#   )
#
#   print_unwrapped_azimuth_samples(
#       scans,
#       target_id="T01",
#   )
#
# =============================================================