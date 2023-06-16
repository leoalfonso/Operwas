import os

import numpy as np
import rasterio

from src.faster import config
from src.faster.custom_typing import Coordinate
from src.faster.simplified_aokp import read_outlet_locations


def intersperse_points(coordinate_start: Coordinate, coordinate_end: Coordinate, n_points: int = 200) -> list[Coordinate]:
    # Add 2 to `n_points` since we always want to at least include the start- and end points.
    # np.linspace creates evenly spaced arrays.
    points_array = np.linspace(coordinate_start, coordinate_end, num=n_points + 2)
    # convert the array to a list, row by row.
    return [tuple(points_array[i, :].tolist()) for i in range(points_array.shape[0])]


def get_elevations(path_xy: list[Coordinate]) -> list[float]:
    """
    Returns the elevation of the points (coordinates).
    input: list with coordinates.
    example: path_xy = [(720400.5, 3516290.0), (716584, 3514880)]
    output: elevations = list containing the values for the elevation of each coordinate in the input list (path_xy)
    """
    with rasterio.open(config.ELEVATION_FILE_PATH, mode='r') as dem:  # Open the DEM.\
        # The DEM expects x- and y-positions. DEM is in epsg:32636 projection mercator.
        vals = dem.sample(path_xy)
        # First element is the elevation
        elevations = [val[0] for val in vals]
    return elevations


def get_path_length(path: list[tuple[float, float, float]]) -> float:
    """ Finds the path length, where the path is a list of points. The distance is calculated by the Pythagorean
    theorem, so it only works for Euclidean spaces (not longitude/latitude!). """
    path_array = np.array(path)
    deltas = np.diff(path_array, axis=0)
    path_length: float = np.sqrt((deltas ** 2).sum(axis=1)).sum()
    return path_length


def get_pumping_info(coordinates: list[Coordinate]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates three arrays (rows and columns = candidate coordinates) containing info for all the possible connections
    regarding the pumping geometric height and the length of the pipeline (pump line and by gravity line).

    inputs: a list with the coordinates of all possible locations (candidates). The value need to be in the epsg:32636,
            same projection mercator of the DEM and the coordinates in all_coords.
    ex: all_points = [(720400.5, 3516290.0), (716584, 3514880)]

    outputs:pumping_heights:    array containing the geometric height that the WW has to be pumped in each connection.
            path_lengths_pump:  array containing the length of the pressurized section of the pipeline for each connect.
            path_lengths_grav:  array containing the length of the section of the pipeline where the WW flows by gravity
                                for each connection.

    EXAMPLE: get_pumping_info([(720400.5, 3516290.0), (716584, 3514880)])

            """""

    # Pre-allocate value arrays for storing
    n_coords = len(coordinates)
    pumping_heights = np.zeros((n_coords, n_coords))
    pumping_lengths_pump = np.zeros((n_coords, n_coords))
    pumping_lengths_grav = np.zeros((n_coords, n_coords))

    for i_row, coord_start in enumerate(coordinates):
        for i_col, coord_end in enumerate(coordinates):
            # Create path between start- and end point in straight line with equally spaced n_points between them.
            path_xy = intersperse_points(coord_start, coord_end, n_points=200)

            # Find elevations along path and add to path
            elevations: np.ndarray = np.array(get_elevations(path_xy))
            path_xyz: list[tuple[float, float, float]] = [
                tuple(list(xy) + [elevation]) for xy, elevation in zip(path_xy, elevations)]

            # Find maximum elevation and pumping height
            max_elevation = elevations.max()
            start_elevation = elevations[0]
            pumping_height = max_elevation - start_elevation

            # Cut up path into pump- and gravity segments
            i_max_elevation: int = elevations.argmax()  # arg.max() returns the index (row and column)\
            # of the maximum value in the array.
            path_xyz_pump = path_xyz[:i_max_elevation + 1]
            path_xyz_grav = path_xyz[i_max_elevation:]

            # Calculate path lengths for pump- and gravity segments
            path_length_pump = get_path_length(path_xyz_pump)
            path_length_grav = get_path_length(path_xyz_grav)

            # Store values in arrays
            pumping_heights[i_row, i_col] = pumping_height
            pumping_lengths_pump[i_row, i_col] = path_length_pump
            pumping_lengths_grav[i_row, i_col] = path_length_grav

    return pumping_heights, pumping_lengths_pump, pumping_lengths_grav


def save_to_file(save_path: str, array: np.ndarray) -> None:
    save_dir, _ = os.path.split(save_path)
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, array)


def run_pumping_info_experiments() -> None:
    print("Starting pumping info experiments...")
    coordinates, _ = read_outlet_locations(config.OUTLETS_FILE_PATH)
    pumping_heights, path_lengths_pump, path_lengths_grav = get_pumping_info(coordinates)
    save_to_file(config.EXP_PUMPING_HEIGHT_FILE_PATH, pumping_heights)
    save_to_file(config.EXP_PUMPING_LENGTH_PUMP_FILE_PATH, path_lengths_pump)
    save_to_file(config.EXP_PUMPING_LENGTH_GRAV_FILE_PATH, path_lengths_grav)
    print("Done.")


if __name__ == "__main__":
    run_pumping_info_experiments()
