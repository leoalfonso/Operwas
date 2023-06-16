from .pop_per_town import run_pop_per_town_experiments
from .pumping import run_pumping_info_experiments
from .reuse_area import run_area_per_reuse_experiments
from .reuse_network import run_reuse_network_length_experiments
from .subcatchments import run_subcatchment_experiments


def run_all_experiments():
    run_pop_per_town_experiments()
    run_area_per_reuse_experiments()
    run_reuse_network_length_experiments()
    run_subcatchment_experiments()
    run_pumping_info_experiments()
