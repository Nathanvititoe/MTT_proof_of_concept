from stone_soup_simple import setup_tracker
from test_cases.mild_test import (
    create_simple_test_tracks,
    run_simple_one_minute_test,
    print_simple_unwrapped_samples,
)
from test_cases.highly_ambig import (
    create_test_tracks,
    run_one_minute_test,
    print_unwrapped_azimuth_samples,
)

from display_tracks import display_tracks

def ambig_test():
    associator, updater, measurement_model = setup_tracker()
    tracks, track_ids, start_time, truth_targets = create_test_tracks()

    # Run the scan.
    tracks, scans, scan_times = run_one_minute_test(
        tracks=tracks,
        associator=associator,
        updater=updater,
        measurement_model=measurement_model,
        start_time=start_time,
        truth_targets=truth_targets,
        track_ids=track_ids,
        # Keep FALSE while using Stone Soup's standard
        # CartesianToElevationBearingRange model.
        pass_unwrapped_to_tracker=False,
    )

    print_unwrapped_azimuth_samples(
        scans,
        target_id="T01",
    )

    display_tracks(tracks=tracks, scans=scans, track_ids=track_ids)

def simple_test():
    associator, updater, measurement_model = setup_tracker()


    tracks, track_ids, start_time, truth_targets = create_simple_test_tracks()


    tracks, scans, scan_times = run_simple_one_minute_test(
        tracks=tracks,
        associator=associator,
        updater=updater,
        measurement_model=measurement_model,
        start_time=start_time,
        truth_targets=truth_targets,
        track_ids=track_ids,

        # False while using the standard Stone Soup wrapped-bearing model.
        pass_unwrapped_to_tracker=False,
    )

    print_simple_unwrapped_samples(scans, target_id="T01")

    display_tracks(tracks=tracks,scans=scans,track_ids=track_ids)

if __name__ == "__main__":
#    simple_test()
    ambig_test()