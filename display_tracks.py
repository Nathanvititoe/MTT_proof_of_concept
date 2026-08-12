import matplotlib

# WSL2 / WSLg interactive plotting
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np


def radar_measurement_to_xyz(elevation, azimuth, range_m):
    """
    Convert radar measurement [elevation, azimuth, range]
    into Cartesian XYZ.
    """

    horizontal_range = range_m * np.cos(elevation)

    x = horizontal_range * np.cos(azimuth)
    y = horizontal_range * np.sin(azimuth)
    z = range_m * np.sin(elevation)

    return float(x), float(y), float(z)


def capture_raw_detection_points(scans):
    """
    Extract all pre-association radar detections from all scans.

    Returns
    -------
    target_points : list[dict]
    clutter_points : list[dict]
    """

    target_points = []
    clutter_points = []

    for scan in scans:

        elapsed = scan.get(
            "elapsed_seconds",
            0.0,
        )

        for record in scan["records"]:

            kind = record.get("kind")

            # Missed detections do not create a point.
            if kind not in ("target", "clutter"):
                continue

            elevation = record[
                "elevation_measured_rad"
            ]

            # Prefer the continuous/unwrapped measurement.
            azimuth = record.get(
                "azimuth_unwrapped_measured_rad",
                record.get("azimuth_tracker_rad"),
            )

            range_m = record[
                "range_measured_m"
            ]

            x, y, z = radar_measurement_to_xyz(
                elevation,
                azimuth,
                range_m,
            )

            point = {
                "x": x,
                "y": y,
                "z": z,
                "time": elapsed,
                "target_id": record.get("target_id"),
            }

            if kind == "clutter":
                clutter_points.append(point)
            else:
                target_points.append(point)

    return target_points, clutter_points


def get_track_history(track):
    """
    Extract complete XYZ history from one Stone Soup Track.

    State layout:
        [x, vx, y, vy, z, vz]
    """

    history = []

    for state in track.states:

        vector = state.state_vector

        history.append(
            (
                float(vector[0]),
                float(vector[2]),
                float(vector[4]),
            )
        )

    return history


def get_track_history_with_time(track):
    """
    Return the full XYZ history plus elapsed seconds from the first
    timestamp in the track.

    Returns
    -------
    list[tuple]
        [(x, y, z, elapsed_seconds), ...]
    """
    if not track.states:
        return []

    first_timestamp = track.states[0].timestamp
    history = []

    for state in track.states:
        vector = state.state_vector

        elapsed_seconds = 0.0
        if (
            first_timestamp is not None
            and state.timestamp is not None
        ):
            elapsed_seconds = (
                state.timestamp - first_timestamp
            ).total_seconds()

        history.append(
            (
                float(vector[0]),
                float(vector[2]),
                float(vector[4]),
                float(elapsed_seconds),
            )
        )

    return history


def _annotate_track_times_3d(
    axis,
    history,
    label_interval_seconds=10,
):
    """
    Add time labels at roughly fixed elapsed-time intervals.
    """
    if not history:
        return

    last_labeled = -1e9

    for x, y, z, elapsed in history:
        if elapsed - last_labeled >= label_interval_seconds:
            axis.text(
                x,
                y,
                z,
                f"{elapsed:.0f}s",
                fontsize=8,
                alpha=0.8,
            )
            last_labeled = elapsed


def _annotate_track_times_xy(
    axis,
    history,
    label_interval_seconds=10,
):
    """
    Add time labels at roughly fixed elapsed-time intervals.
    """
    if not history:
        return

    last_labeled = -1e9

    for x, y, _z, elapsed in history:
        if elapsed - last_labeled >= label_interval_seconds:
            axis.text(
                x,
                y,
                f"{elapsed:.0f}s",
                fontsize=8,
                alpha=0.8,
            )
            last_labeled = elapsed


def _collect_all_xyz(
    tracks,
    raw_target_points,
    raw_clutter_points,
):
    """
    Collect all coordinates so before and after use matching axes.
    """

    xs = [0.0]
    ys = [0.0]
    zs = [0.0]

    for point in raw_target_points:
        xs.append(point["x"])
        ys.append(point["y"])
        zs.append(point["z"])

    for point in raw_clutter_points:
        xs.append(point["x"])
        ys.append(point["y"])
        zs.append(point["z"])

    for track in tracks:

        history = get_track_history(track)

        for x, y, z in history:
            xs.append(x)
            ys.append(y)
            zs.append(z)

    return xs, ys, zs


def _padded_limits(values, padding_fraction=0.05):

    minimum = min(values)
    maximum = max(values)

    span = maximum - minimum

    if span <= 0:
        span = 1.0

    padding = span * padding_fraction

    return (
        minimum - padding,
        maximum + padding,
    )


def plot_before_after_3d(
    tracks,
    raw_target_points,
    raw_clutter_points,
    track_ids=None,
):
    """
    3D comparison.

    LEFT:
        All raw radar measurements before JPDA.

    RIGHT:
        Complete associated Stone Soup track histories.
    """

    fig = plt.figure(
        figsize=(17, 8)
    )


    # ==============================================================
    # BEFORE
    # ==============================================================

    ax_before = fig.add_subplot(
        121,
        projection="3d",
    )


    if raw_target_points:

        before_scatter = ax_before.scatter(
            [p["x"] for p in raw_target_points],
            [p["y"] for p in raw_target_points],
            [p["z"] for p in raw_target_points],
            c=[p["time"] for p in raw_target_points],
            cmap="viridis",
            s=18,
            alpha=0.75,
            marker="o",
            label="Raw detections",
        )

        colorbar = fig.colorbar(
            before_scatter,
            ax=ax_before,
            pad=0.08,
            shrink=0.75,
        )
        colorbar.set_label(
            "Elapsed time [s]"
        )


    if raw_clutter_points:

        ax_before.scatter(
            [p["x"] for p in raw_clutter_points],
            [p["y"] for p in raw_clutter_points],
            [p["z"] for p in raw_clutter_points],
            s=30,
            alpha=0.8,
            marker="x",
            label="Clutter",
        )


    ax_before.scatter(
        0,
        0,
        0,
        marker="^",
        s=100,
        label="Radar",
    )


    ax_before.set_title(
        "BEFORE JPDA / ASSOCIATION\nRaw Radar Detections Colored by Time"
    )

    ax_before.set_xlabel("X [m]")
    ax_before.set_ylabel("Y [m]")
    ax_before.set_zlabel("Z [m]")

    ax_before.legend()


    # ==============================================================
    # AFTER
    # ==============================================================

    ax_after = fig.add_subplot(
        122,
        projection="3d",
    )


    for index, track in enumerate(
        tracks,
        start=1,
    ):

        history_with_time = get_track_history_with_time(track)

        if not history_with_time:
            continue

        xs = [p[0] for p in history_with_time]
        ys = [p[1] for p in history_with_time]
        zs = [p[2] for p in history_with_time]


        if track_ids is not None:

            label = track_ids.get(
                track,
                f"Track {index:02d}",
            )

        else:

            label = f"Track {index:02d}"


        ax_after.plot(
            xs,
            ys,
            zs,
            linewidth=2.0,
            label=label,
        )

        # Start of associated track.
        ax_after.scatter(
            xs[0],
            ys[0],
            zs[0],
            marker="s",
            s=55,
        )

        ax_after.text(
            xs[0],
            ys[0],
            zs[0],
            f"{label} start",
            fontsize=8,
        )

        # End/current associated state.
        ax_after.scatter(
            xs[-1],
            ys[-1],
            zs[-1],
            marker="X",
            s=80,
        )

        ax_after.text(
            xs[-1],
            ys[-1],
            zs[-1],
            f"{label} end",
            fontsize=9,
        )

        _annotate_track_times_3d(
            ax_after,
            history_with_time,
            label_interval_seconds=10,
        )


    ax_after.scatter(
        0,
        0,
        0,
        marker="^",
        s=100,
    )


    ax_after.set_title(
        "AFTER JPDA / ASSOCIATION\nEstimated Track Histories"
    )

    ax_after.set_xlabel("X [m]")
    ax_after.set_ylabel("Y [m]")
    ax_after.set_zlabel("Z [m]")
    ax_after.legend(
        loc="best",
        fontsize=8,
    )


    # ==============================================================
    # SAME AXIS LIMITS
    # ==============================================================

    all_x, all_y, all_z = _collect_all_xyz(
        tracks,
        raw_target_points,
        raw_clutter_points,
    )

    x_limits = _padded_limits(all_x)
    y_limits = _padded_limits(all_y)
    z_limits = _padded_limits(all_z)


    for axis in (
        ax_before,
        ax_after,
    ):

        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_zlim(*z_limits)


    plt.tight_layout()

    plt.show()


def plot_before_after_xy(
    tracks,
    raw_target_points,
    raw_clutter_points,
    track_ids=None,
):
    """
    Top-down X/Y comparison.

    LEFT:
        Raw radar detections.

    RIGHT:
        Associated Stone Soup track histories.
    """

    fig, (
        ax_before,
        ax_after,
    ) = plt.subplots(
        1,
        2,
        figsize=(17, 8),
    )


    # ==============================================================
    # BEFORE
    # ==============================================================

    if raw_target_points:

        before_scatter = ax_before.scatter(
            [p["x"] for p in raw_target_points],
            [p["y"] for p in raw_target_points],
            c=[p["time"] for p in raw_target_points],
            cmap="viridis",
            s=18,
            alpha=0.75,
            marker="o",
            label="Raw detections",
        )

        colorbar = fig.colorbar(
            before_scatter,
            ax=ax_before,
            pad=0.02,
        )
        colorbar.set_label(
            "Elapsed time [s]"
        )


    if raw_clutter_points:

        ax_before.scatter(
            [p["x"] for p in raw_clutter_points],
            [p["y"] for p in raw_clutter_points],
            s=30,
            alpha=0.8,
            marker="x",
            label="Clutter",
        )


    ax_before.scatter(
        0,
        0,
        marker="^",
        s=100,
        label="Radar",
    )


    ax_before.set_title(
        "BEFORE JPDA / ASSOCIATION\nRaw Radar Detections Colored by Time"
    )

    ax_before.set_xlabel("X [m]")
    ax_before.set_ylabel("Y [m]")

    ax_before.grid(True)
    ax_before.legend()


    # ==============================================================
    # AFTER
    # ==============================================================

    for index, track in enumerate(
        tracks,
        start=1,
    ):

        history_with_time = get_track_history_with_time(track)

        if not history_with_time:
            continue

        xs = [p[0] for p in history_with_time]
        ys = [p[1] for p in history_with_time]


        if track_ids is not None:

            label = track_ids.get(
                track,
                f"Track {index:02d}",
            )

        else:

            label = f"Track {index:02d}"


        ax_after.plot(
            xs,
            ys,
            linewidth=2.0,
            label=label,
        )

        ax_after.scatter(
            xs[0],
            ys[0],
            marker="s",
            s=55,
        )

        ax_after.text(
            xs[0],
            ys[0],
            f"{label} start",
            fontsize=8,
        )

        ax_after.scatter(
            xs[-1],
            ys[-1],
            marker="X",
            s=80,
        )

        ax_after.text(
            xs[-1],
            ys[-1],
            f"{label} end",
            fontsize=9,
        )

        _annotate_track_times_xy(
            ax_after,
            history_with_time,
            label_interval_seconds=10,
        )


    ax_after.scatter(
        0,
        0,
        marker="^",
        s=100,
    )


    ax_after.set_title(
        "AFTER JPDA / ASSOCIATION\nEstimated Track Histories"
    )

    ax_after.set_xlabel("X [m]")
    ax_after.set_ylabel("Y [m]")

    ax_after.grid(True)
    ax_after.legend(
        loc="best",
        fontsize=8,
    )


    # ==============================================================
    # SAME AXIS LIMITS
    # ==============================================================

    all_x, all_y, _ = _collect_all_xyz(
        tracks,
        raw_target_points,
        raw_clutter_points,
    )

    x_limits = _padded_limits(all_x)
    y_limits = _padded_limits(all_y)


    for axis in (
        ax_before,
        ax_after,
    ):

        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)

        axis.set_aspect(
            "equal",
            adjustable="box",
        )


    plt.tight_layout()

    plt.show()


def display_tracks(
    tracks,
    scans,
    track_ids=None,
    show_3d=True,
    show_xy=True,
):
    """
    Main plotting entry point.

    Parameters
    ----------
    tracks
        Final Stone Soup tracks after association.

    scans
        Raw scan records from test_tracks_simple.py or test_tracks.py.

    track_ids
        Optional mapping:
            Track object -> test target ID

    show_3d
        Display 3D comparison.

    show_xy
        Display XY top-down comparison.
    """

    raw_target_points, raw_clutter_points = (
        capture_raw_detection_points(
            scans
        )
    )


    print(
        f"Raw target detections: "
        f"{len(raw_target_points)}"
    )

    print(
        f"Raw clutter detections: "
        f"{len(raw_clutter_points)}"
    )

    print(
        f"Associated tracks: "
        f"{len(tracks)}"
    )


    if show_3d:

        plot_before_after_3d(
            tracks=tracks,
            raw_target_points=raw_target_points,
            raw_clutter_points=raw_clutter_points,
            track_ids=track_ids,
        )


    if show_xy:

        plot_before_after_xy(
            tracks=tracks,
            raw_target_points=raw_target_points,
            raw_clutter_points=raw_clutter_points,
            track_ids=track_ids,
        )