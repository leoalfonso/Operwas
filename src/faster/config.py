#This configuration file sets the paths and filenames and pump characteristics limits.
#Read by genetic_pumping and by no_pumping.

import os
import pathlib
from typing import Optional

ROOT_DIR = pathlib.Path(os.path.realpath(__file__)).parent.parent.parent

OUTLETS_FILE_PATH = os.path.join(ROOT_DIR, "inputs", "wwtp_locations_many.csv")
ELEVATION_FILE_PATH = os.path.join(ROOT_DIR, "inputs", "DEM_study_area.tif")

SCRIBBLE_RESULTS_BASE_DIR = os.path.join(ROOT_DIR, "scribble_results")
RUN_RESULTS_DIR = os.path.join(ROOT_DIR, "optimization_results")
RUN_RESULTS_ARCHIVE_DIR = os.path.join(ROOT_DIR, "optimization_results_archive")
EXP_RESULTS_BASE_DIR_PATH = os.path.join(ROOT_DIR, "experimental_results")

# Experiments: subcatchments
EXP_SUBCATCHMENTS_RESULTS_DIR_PATH = os.path.join(EXP_RESULTS_BASE_DIR_PATH, "subcatchments")
EXP_SUBCATCHMENTS_RESULTS_FILE_PREFIX = "subcatchment"
GRAPH_DATA_FILE_PATH = os.path.join(EXP_SUBCATCHMENTS_RESULTS_DIR_PATH, "subcatchment_all.csv")

# Experiments: reuse network length
EXP_NETWORKLENGTH_RESULTS_DIR_PATH = os.path.join(EXP_RESULTS_BASE_DIR_PATH, "reuse_network_length")
EXP_NETWORKLENGTH_RESULTS_FILE_PREFIX = "reuse_network_length"
EXP_NETWORKLENGTH_RADIUS_MAX = 1e3
EXP_NETWORKLENGTH_N_SAMPLES = 200

# Experiments: area per reuse
EXP_AREA_RESULTS_DIR_PATH = os.path.join(EXP_RESULTS_BASE_DIR_PATH, "area_per_reuse")
EXP_AREA_RESULTS_FILE_PREFIX = "area_per_reuse"
EXP_AREA_RADIUS_MAX = 1e3
EXP_AREA_N_SAMPLES = 200

# Experiments: population per town
EXP_POPULATION_RESULTS_DIR_PATH = os.path.join(EXP_RESULTS_BASE_DIR_PATH, "population_per_town")
EXP_POPULATION_RESULTS_FILE_PREFIX = "population_per_town"
EXP_POPULATION_RADIUS_MAX = 1e3
EXP_POPULATION_N_SAMPLES = 200

# Experiments: pumping info
EXP_PUMPING_RESULTS_DIR_PATH = os.path.join(EXP_RESULTS_BASE_DIR_PATH, "pumping")
EXP_PUMPING_HEIGHT_FILE_PATH = os.path.join(
    EXP_PUMPING_RESULTS_DIR_PATH, "pumping_height.npy")
EXP_PUMPING_LENGTH_PUMP_FILE_PATH = os.path.join(
    EXP_PUMPING_RESULTS_DIR_PATH, "pumping_length_pump.npy")
EXP_PUMPING_LENGTH_GRAV_FILE_PATH = os.path.join(
    EXP_PUMPING_RESULTS_DIR_PATH, "pumping_length_grav.npy")

# Pumping feasibility
PUMPING_HEIGHT_MIN = 0.0
PUMPING_HEIGHT_MAX = 40.0
PUMPING_LENGTH_PUMP_MIN = 0.0
PUMPING_LENGTH_PUMP_MAX = 1e3
PUMPING_LENGTH_GRAV_MIN = 0.0
PUMPING_LENGTH_GRAV_MAX = 1e3
PUMPING_LENGTH_TOTAL_MIN = 0.0
PUMPING_LENGTH_TOTAL_MAX = 2e3

PUMPING_SIMPLE_PATH_LENGTH_MAX = 1

# Genetic pumping parameters
WEIGHT_NODE_TYPE_NOTHING = 1.0
WEIGHT_NODE_TYPE_WWTP = 1.0
WEIGHT_NODE_TYPE_WWPS = 1.0
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.2

OPT_NUM_GENERATIONS = 5
OPT_POPULATION_SIZE = 10

RUN_RESULTS_STORE_RESULTS_SOLUTIONS_INTERVAL: Optional[int] = 1
RUN_RESULTS_STORE_RESULTS_TOTAL_INTERVAL: Optional[int] = 1
RUN_RESULTS_STORE_RESULTS_NODES_INTERVAL: Optional[int] = None
RUN_RESULTS_STORE_RESULTS_NODES_SUMMARY_INTERVAL: Optional[int] = None
