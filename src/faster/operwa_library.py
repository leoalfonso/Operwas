import logging
from typing import Callable, Optional, List

import numpy as np

import src.user_inputs as usin
from src.faster.custom_typing import DataInterpolator, Scalar
from src.faster.node_types import NodeType
from src.faster.pumping_stuff import (calc_wwps_capex,
                                      calc_wwps_opex_and_maint_year)
from src.faster.reuse_types import ReuseType
from src.faster import config

from src.faster.optimization.Odur.operwa_TT_module import runIntegration, \
    runTreatmentModuleIntegration, toV1CostArray, normalizeV2ZeroFlows  # From Gerald Odur

logging.basicConfig()


################################################
## Description of functions in AOKP algorithm ##
################################################


def present_value(DR, n, c_op_year):
    """
        Calculate the present value cost for a determined number of operation years.

        Args:
            DR (float): Discount rate.
            n (int): Expected operating life in years.
            cost_year (float): Cost expended per year for operation in ILS/year.

        Returns:
            float: Total present value cost in ILS.
        """
    c_op_total = c_op_year * (1 - (1 + DR) ** (-n)) / DR  # [ILS/year]

    return c_op_total


def calculate_scale_factor(Inv_1, Inv_2, Q1, Q2):
    """
    Calculate the scale factor for the cost to capacity method.

    Args:
        Inv_1 (float): Factor scale for the Investment Cost - Price 1 [-].
        Inv_2 (float): Factor scale for the O&M Cost - Price 2 [-].
        Q1 (float): Design Capacity of WWTP1 [m³/day].
        Q2 (float): Design Capacity of WWTP2 [m³/day].

    Returns:
        float: The calculated scale factor.
            If the scale factor is greater than 1, implying diseconomies of scale exist
            and the incremental cost becomes more expensive for every added unit of capacity,
            it returns a value of 0.7.
            Otherwise, it returns the calculated scale factor.
    """

    sc_factor = (np.log(Inv_2 / Inv_1)) / (np.log(Q2 / Q1))

    if sc_factor > 1:
        return 0.7
    else:
        return sc_factor


def calculate_flow_based_on_population(pop_per_town_subcatchment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the flow at each treatment plant, considering the designed flow and
    population in each subcatchment.

    Args:
        pop_per_town_subcatchment (np.ndarray): Population in each subcatchment connected to WWTP (inhab).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            flow_wwtp (np.ndarray): Capacity of the WWTP [m³/day].
            flow_to_be_treated_peak (np.ndarray): Peak flow to be treated [m³/day].
    """

    water_consumption_per_person_arr = np.array(usin.water_consumption)
    water_consumption_total: np.ndarray = np.einsum("ij,i->j", pop_per_town_subcatchment,
                                                    water_consumption_per_person_arr)
    flow_to_be_treated: np.ndarray = water_consumption_total * (1 - usin.losses_infiltration)
    flow_to_be_treated_peak = flow_to_be_treated * usin.coef_peak

    return flow_to_be_treated, flow_to_be_treated_peak


def coverage_check(pop_per_subcatchment_wwtp: np.ndarray, real_pop: float) -> float:
    """
    Calculate the coverage of the wastewater network in the region.

    Args:
        pop_per_subcatchment_wwtp (np.ndarray): Population in the region connected to a WWTP (inhab).
        real_pop (float): Total population given by the user.

    Returns:
        float: Percentage of connected population to the total population.
    """
    connected_pop = pop_per_subcatchment_wwtp.sum()
    coverage: float = connected_pop / real_pop

    return coverage


def calc_benefit_connections(pop_per_catchment: np.ndarray) -> np.ndarray:
    """
    Calculate the benefits obtained through the payment of connection fees by the population.

    This payment is made once in the investment time (usually in the first year) and takes into account
    the average number of floors per building in the region and the area of buildings per subcatchment.

    Args:
        pop_per_catchment (np.ndarray): Population in each subcatchment.

    Returns:
        np.ndarray: Benefits obtained through connection fees paid by the population.
    """
    fee_per_person = usin.fee_connection / usin.inhabit_per_household
    benefit_connection: np.ndarray = pop_per_catchment * fee_per_person

    return benefit_connection


def calc_benefit_ww_yearly_fees(pop_per_town_per_subcatchment: np.ndarray) -> np.ndarray:
    """
    Calculate the benefits obtained through the payment of yearly fees by the population to the water authorities.
    The present value calculation is made over the investment period time.

    Args:
        pop_per_town_per_subcatchment (np.ndarray): Population in each town per subcatchment.

    Returns:
        np.ndarray: Benefits obtained through yearly fees paid by the population to the water authorities
                    per subcatchment.
    """
    water_consumtion_arr = np.array(usin.water_consumption)
    fee_sanitary_arr = np.array(usin.fee_sanitary)
    c_arr = water_consumtion_arr * fee_sanitary_arr * 365
    benefit_fees_per_subcatchment: np.ndarray = np.einsum(
        "ij,i->j", pop_per_town_per_subcatchment, c_arr)

    return benefit_fees_per_subcatchment


def get_land_cost_by_wwtp(land_cost_per_m2: np.ndarray, flow_wwtp: np.ndarray, type_reuse: np.ndarray) -> np.ndarray:
    """
        Calculate the land cost based on the wastewater treatment plant (WWTP) characteristics and reuse type.

        Args:
            land_cost_per_m2 (np.ndarray): Land cost per square meter.
            flow_wwtp (np.ndarray): Capacity of the WWTP [m³/day].
            type_reuse (np.ndarray): Array indicating the type of reuse (AGRICULTURAL or URBAN).

        Returns:
            np.ndarray: Calculated land cost based on the WWTP type and flow.
        """
    land_cost = np.zeros(land_cost_per_m2.shape)

    # Agricultural
    # TODO: could just assume that when there is reuse the NodeType is automatically WWTP
    idx_agricultural = type_reuse == ReuseType.AGRICULTURAL
    land_cost[idx_agricultural] = usin.area_usage_cas * \
                                  flow_wwtp[idx_agricultural] * land_cost_per_m2[idx_agricultural]

    # Urban
    # TODO: could just assume that when there is reuse the NodeType is automatically WWTP
    idx_urban = type_reuse == ReuseType.URBAN
    land_cost[idx_urban] = usin.area_usage_mbr * \
                           flow_wwtp[idx_urban] * land_cost_per_m2[idx_urban]

    return land_cost


def calculate_land_cost_wwps(land_cost_per_m2: np.ndarray) -> np.ndarray:
    """
        Calculate the land cost for Wastewater Pumping Stations (WWPS).

        Args:
            land_cost_per_m2 (np.ndarray): Land cost per square meter.

        Returns:
            np.ndarray: Calculated land cost for WWPS.
        """
    land_cost = np.zeros(land_cost_per_m2.shape)
    # Agricultural
    # idx_wwps = type_node == NodeType.WWPS
    # # TODO: need to have WWPS specific land-cost parameters
    # land_cost[idx_wwps] = usin.area_usage_cas * flow[idx_wwps] * land_cost_per_m2[idx_wwps]
    # Decided that the pumping station size is so small it doesnt have land cost

    return land_cost


def calculate_wwtp_costs(flow_wwtp: np.ndarray, type_reuse: np.ndarray) -> np.ndarray:

    """
    Calculate wastewater treatment plant (WWTP) costs based on technology selection criteria for different reuse types.

    This function uses the original treatment_costs of Maria W, which is based on CAS for agricultural irrigation use,
    and MBR for urban irrigation or No-reuse. If costs are to be calculated based on technology selection criteria for
    a given reuse and flow/person equivalent, then use the version of Odur, implemented in the function
    "calculate_wwtp_costs_trains".

    Args:
        flow_wwtp (np.ndarray): Capacity of the WWTP [m³/day].
        type_reuse (np.ndarray): Array indicating the type of reuse (AGRICULTURAL or URBAN).

    Returns:
        np.ndarray: Calculated treatment costs for WWTP based on the specified reuse type.
    """

    treatment_costs = np.zeros(flow_wwtp.shape)

    # Agricultural
    idx_agricultural = type_reuse == ReuseType.AGRICULTURAL
    treatment_costs[idx_agricultural] = wwtp_costs(flow_wwtp[idx_agricultural], usin.known_flow_cas_1,
                                                   usin.known_flow_cas_2, usin.known_inv_cost_cas_1,
                                                   usin.known_inv_cost_cas_2, usin.known_oem_cost_cas_1,
                                                   usin.known_oem_cost_cas_2, usin.DR, usin.n)
    # Urban
    idx_urban = type_reuse == ReuseType.URBAN
    treatment_costs[idx_urban] = wwtp_costs(flow_wwtp[idx_urban], usin.known_flow_mbr_1, usin.known_flow_mbr_2,
                                            usin.known_inv_cost_mbr_1, usin.known_inv_cost_mbr_2,
                                            usin.known_oem_cost_mbr_1, usin.known_oem_cost_mbr_2,
                                            usin.DR, usin.n)

    return treatment_costs


def calculate_wwtp_costs_trains(flow_wwtp: np.ndarray, land_cost_in_wwtp_location:List[float],type_reuse: np.ndarray):
    """
        Calculate wastewater treatment plant (WWTP) costs based on technology selection criteria for a given reuse and
        flow/person equivalent and integrating it pumping options analysis.

        This function aims to implement the technology selection code by running integration for the given flow rates of the WWTP and
        land costs in the WWTP location. It's designed to test these costs for compatibility with the pumping analysis existing code.

        Args:
            flow_wwtp (np.ndarray): Capacity of the WWTP [m³/day].
            land_cost_in_wwtp_location (List[float]): Land costs in the WWTP location.
            type_reuse (np.ndarray): Array indicating the type of reuse.

        Returns:
            np.ndarray: Calculated treatment costs for WWTP based on the specified flow rates, land costs, and reuse type.
    """

    # todo: implement here Gerald Odur's code: runIntegration(flow_wwtp, land_cost_in_wwtp_location)
    # todo: I need to test these costs for compatibility with Maria A's code
    runIntegration(flow_wwtp, land_cost_in_wwtp_location)
    treatment_train_data_v2 = runTreatmentModuleIntegration(
        flow_wwtp, type_reuse
    )
    treatment_costs = toV1CostArray(treatment_train_data_v2, ["capital_cost", "operational_cost", "energy_cost"])
    #land_cost_by_wwtp = toV1CostArray(treatment_train_data_v2, ["land_requirement_cost"])

    return treatment_costs

def calculate_wwps_costs(flow_wwps_m3pd: np.ndarray,
                         type_node: np.ndarray,
                         pipe_length_pump: np.ndarray,
                         pipe_length_grav: np.ndarray,
                         pump_height: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    """
        Calculate the costs associated with Wastewater Pumping Stations (WWPS).

        Args:
            flow_wwps_m3pd (np.ndarray): Flow rate for WWPS [m³/day].
            type_node (np.ndarray): Array indicating the type of node (WWPS or other).
            pipe_length_pump (np.ndarray): Pipe length connected to pumps [m].
            pipe_length_grav (np.ndarray): Pipe length connected to gravity systems [m].
            pump_height (np.ndarray): Pump height [m].

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Investment costs for WWPS [currency].
                - Operational lifetime costs for WWPS [currency].
    """

    secs_per_day = 24 * 3600
    flow_wwps_m3ps = flow_wwps_m3pd * (1 / secs_per_day)

    idx_pump_station = type_node == NodeType.WWPS

    # Construction investment costs
    wwps_costs_investment = np.zeros(flow_wwps_m3ps.shape)
    wwps_costs_investment[idx_pump_station] = calc_wwps_capex(
        flow_wwps_m3ps[idx_pump_station], pipe_length_pump[idx_pump_station], pipe_length_grav[idx_pump_station])

    # Operational costs
    wwps_costs_operational_year = np.zeros(flow_wwps_m3ps.shape)
    wwps_costs_operational_year[idx_pump_station] = calc_wwps_opex_and_maint_year(
        flow_wwps_m3ps[idx_pump_station], pipe_length_pump[idx_pump_station], pump_height[idx_pump_station])
    scale_factor_present_value: float = present_value(DR=usin.DR, n=usin.n, c_op_year=1.0)
    wwps_costs_operational_lifetime = scale_factor_present_value * wwps_costs_operational_year

    return wwps_costs_investment, wwps_costs_operational_lifetime


def wwtp_costs(flow_wwtp: np.ndarray, known_flow_1, known_flow_2: float, known_inv_1: float,
               known_inv_2: float, known_oem_1: float, known_oem_2: float, DR: float, n: float):
    """
        Calculate wastewater treatment plant (WWTP) costs based on known parameters.

        Args:
            flow_wwtp (np.ndarray): Capacity of the WWTP [m³/day].
            known_flow_1 (float): Known flow rate 1 [m³/day].
            known_flow_2 (float): Known flow rate 2 [m³/day].
            known_inv_1 (float): Known investment cost 1.
            known_inv_2 (float): Known investment cost 2.
            known_oem_1 (float): Known O&M cost 1.
            known_oem_2 (float): Known O&M cost 2.
            DR (float): Discount rate.
            n (float): Number of operation years.

        Returns:
            np.ndarray: Total cost of the WWTP based on the provided parameters.
    """

    # Present value for O&M - costs  1
    pv_oem_1 = present_value(DR=DR, n=n, c_op_year=known_oem_1)
    # Present value for O&M - costs 2
    pv_oem_2 = present_value(DR=DR, n=n, c_op_year=known_oem_2)
    # Scale factor for O&M
    sc_oem = calculate_scale_factor(Inv_1=pv_oem_1, Inv_2=pv_oem_2,
                                    Q1=known_flow_1, Q2=known_flow_2)
    # Scale factor for investment
    sc_inv = calculate_scale_factor(
        Inv_1=known_inv_1, Inv_2=known_inv_2, Q1=known_flow_1, Q2=known_flow_2)

    # Cost to capacity for O&M
    oem_costs = ((flow_wwtp * (1 / known_flow_1)) ** (sc_oem)) * known_oem_1
    investment_costs = ((flow_wwtp * (1 / known_flow_1)) ** (sc_inv)) * known_inv_1
    wwtp_total_cost = investment_costs + oem_costs

    return wwtp_total_cost

#REVISE AND IMPROVE DOCSTRINGS
def calc_pipeline_reuse_inv_cost_factory(pipe_cost: list[float], hRan: list[float]) -> Callable[[float], float]:
    """
    This function calculates the costs of the construction of pipelines.

    :param: pipe_cost = Cost pipe [ILS/m] (list with 8 elements)
            hRan= highest range value of the length of the pipes per specific diameter [m] (list with 7 elements)
            pipe_length = length of the network in each catchment [m]
    :return: pipe_inv_cost
    """
    # TODO: pre-allocate all these arrays
    pipe_cost_arr = np.array(pipe_cost)
    hRan_arr = np.array(hRan)
    hRan_start_zero = np.append(np.array([0.0]), hRan_arr)
    hRan_diff = np.diff(hRan_start_zero)
    cost_arr_start_zero = hRan_diff * pipe_cost_arr[:-1]
    hRan_arr_end_inf = np.append(hRan_arr, [np.inf])
    cost_arr_total = np.append(np.array([0.0]), cost_arr_start_zero)
    cost_arr_total_cumsum = np.cumsum(cost_arr_total) - hRan_start_zero * pipe_cost_arr

    def inner(pipe_length: float) -> float:
        for i, hran in enumerate(hRan_arr_end_inf):
            if pipe_length < hran:
                pipeline_inv_cost: float = cost_arr_total_cumsum[i] + pipe_length * pipe_cost_arr[i]
                return pipeline_inv_cost
        raise RuntimeError("Unreachable reached.")

    return inner

# REVISE
PIPELINE_REUSE_INV_COST_CALCULATOR = calc_pipeline_reuse_inv_cost_factory(
    pipe_cost=usin.Pw, hRan=usin.hRan)


def calculate_reuse_network_costs(pipe_length_reuse: np.ndarray, types_reuse: np.ndarray) -> np.ndarray:
    """
        Calculate the costs associated with the reuse network based on pipe length and types of reuse.

        Args:
            pipe_length_reuse (np.ndarray): Length of pipes in the reuse network [m].
            types_reuse (np.ndarray): Array indicating the type of reuse for each pipe.

        Returns:
            np.ndarray: Calculated costs for the reuse network based on the provided pipe lengths and reuse types.
    """
    reuse_network_costs_new = np.zeros(pipe_length_reuse.shape)
    for i, (type_reuse, pipe_length) in enumerate(zip(types_reuse, pipe_length_reuse)):
        if type_reuse in {ReuseType.AGRICULTURAL, ReuseType.URBAN}:
            reuse_network_costs_new[i] = PIPELINE_REUSE_INV_COST_CALCULATOR(pipe_length)

    return reuse_network_costs_new


# ***************************************************************************************************************
# *****************HOSSEIN'S COSTS OF NETWORKS DESIGNED USING SWIMM for three WWTPs*******************************
# ***************************************************************************************************************
# # todo: read the file outside this function. Instead, use a variable with the content of the file as an argument.
# def network_cost_swimm(cost_path, WWTP1, WWTP2, WWTP3):
#     # Convert WWTP1, WWTP2, WWTP3 to numbers and sort them
#     sorted_costs = sorted([int(WWTP1), int(WWTP2), int(WWTP3)])
#     # Create the comb variable as a string
#
#     comb = f"{sorted_costs[0]},{sorted_costs[1]},{sorted_costs[2]}"
#     cost_file = open(cost_path, "r")
#
#     for line in cost_file:
#         if line.startswith(comb):
#             cost = line.split(',')[3]
#             cost_file.close()
#             return int(cost)
#     cost_file.close()
#     return

# REVISE: where is this files_directory and what does it contain?
def network_cost_swimm(files_directory, WWTP_IDs):
    """
        Retrieve network cost information based on specified WWTP IDs.

        Args:
            files_directory (str): Directory containing the cost files.
            WWTP_IDs (list): List of WWTP IDs.

        Returns:
            float: Network cost information based on the provided WWTP IDs.
                   Returns 0 if no matching cost information is found.
    """

    # Convert WWTP IDs to numbers and sort them

    WWTP_IDs = [int(WWTP_ID) for WWTP_ID in WWTP_IDs]
    WWTP_IDs.sort()

    # Create the comb variable as a string
    comb = ""

    for WWTP_ID in WWTP_IDs:
        comb += f"{WWTP_ID},"

    file_path = files_directory + '\\' + str(len(WWTP_IDs)) + '.txt'
    cost_file = open(file_path, "r")

    for line in cost_file:
        if line.startswith(comb):
            cost = line.split(',')[len(WWTP_IDs)]
            cost_file.close()
            return cost

    cost_file.close()

    return 0


# ***************************************************************************************************************
# *****************HOSSEIN'S COSTS OF NETWORKS DESIGNED USING SWIMM for three WWTPs*******************************
# ***************************************************************************************************************

def calculate_flows_reuse(area_reuse: np.ndarray, flow_reuse: np.ndarray,
                          types_reuse: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Calculate flows based on different types of reuse (Agricultural, Urban, No-reuse).

        Args:
            area_reuse (np.ndarray): Area for each type of reuse.
            flow_reuse (np.ndarray): Flow data for different types of reuse.
            types_reuse (np.ndarray): Array indicating the type of reuse for each data entry.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - Flow data for Agricultural reuse.
                - Flow data for Urban reuse.
                - Flow data for No-reuse.
                - Flow per area for Agricultural reuse.
                - Flow per area for Urban reuse.
    """

    types_reuse = np.array(types_reuse)

    # Agricultural
    flow_agricultural = np.zeros(flow_reuse.shape)
    flow_per_area_agricultural = np.zeros(flow_reuse.shape)
    idx_agricultural = types_reuse == ReuseType.AGRICULTURAL
    flow_agricultural[idx_agricultural] = flow_reuse[idx_agricultural]
    flow_per_area_agricultural[idx_agricultural] = flow_reuse[idx_agricultural] / \
                                                   area_reuse[idx_agricultural]

    # Urban
    flow_urban = np.zeros(flow_reuse.shape)
    flow_per_area_urban = np.zeros(flow_reuse.shape)
    idx_urban = types_reuse == ReuseType.URBAN
    flow_urban[idx_urban] = flow_reuse[idx_urban]
    flow_per_area_urban[idx_urban] = flow_reuse[idx_urban] / area_reuse[idx_urban]

    # No_reuse
    flow_no_reuse = np.zeros(flow_reuse.shape)
    idx_no_reuse = types_reuse == ReuseType.NO_REUSE
    flow_no_reuse[idx_no_reuse] = flow_reuse[idx_no_reuse]

    return flow_agricultural, flow_urban, flow_no_reuse, flow_per_area_agricultural, flow_per_area_urban


def benefits_freshwater_with_reuse(population: np.ndarray, flow_urban_reuse: Optional[np.ndarray] = None) -> np.ndarray:
    """
        Calculate benefits of freshwater usage considering urban water reuse.

        Args:
            population (np.ndarray): Population data for various towns.
            flow_urban_reuse (Optional[np.ndarray]): Flow data for urban water reuse (if available). Default is None.

        Returns:
            np.ndarray: Calculated benefits of freshwater usage per subcatchment.
    """

    # TODO: cast lists to arrays already in the input script
    water_consumption_arr = np.array(usin.water_consumption)  # (n_towns,)
    tariff_arr = np.array(usin.tariff)
    coll_eff_arr = np.array(usin.collection_efficiency)

    # Volumes
    volume_consumption = np.einsum(
        "ij,i->ij", population, water_consumption_arr) * 365  # (n_towns, n_outlets)
    if flow_urban_reuse is not None:
        # REVISE
        # If there is urban reuse, then decrease volume_consumption by fixed percentage
        # TODO: this makes no sense, as the percentage is fixed and not dependent on the
        # actual urban reuse
        c_flow_urban_reuse = np.ones(volume_consumption.shape[1])
        c_flow_urban_reuse[flow_urban_reuse > 0] = 1 - usin.urb_irrig_use_coef
        volume_consumption = np.einsum("ij,j->ij", volume_consumption, c_flow_urban_reuse)

    c_tarrif_collection = tariff_arr * coll_eff_arr

    benefit_freshwater_per_subcatch: np.ndarray = np.einsum(
        "ij,i->j", volume_consumption, c_tarrif_collection)

    return benefit_freshwater_per_subcatch


def benefits_reuse_selling(flow_agriculture_reuse: np.ndarray, flow_urban_reuse: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    """
        Calculate benefits from selling reclaimed water for agriculture and urban use.

        Args:
            flow_agriculture_reuse (np.ndarray): Flow data for reclaimed water used in agriculture.
            flow_urban_reuse (np.ndarray): Flow data for reclaimed water used in urban areas.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Benefits from selling reclaimed water for agriculture.
                - Benefits from selling reclaimed water for urban use.
        """

    tariff_reclaimed_arr = np.array(usin.tariff_reclaimed)  # (n_towns,)
    collection_efficiency_arr = np.array(usin.collection_efficiency)  # (n_towns,)

    c_benf_arr = tariff_reclaimed_arr * collection_efficiency_arr * 365  # (n_towns,)
    c_benf_total = c_benf_arr.sum()

    benefits_agriculture = flow_agriculture_reuse * c_benf_total
    benefits_urban = flow_urban_reuse * c_benf_total

    return benefits_agriculture, benefits_urban


def calculate_energy_savings(idx_towns: np.ndarray, flow_reuse: np.ndarray) -> np.ndarray:
    """
        Calculate energy savings achieved by substituting a source for reclaimed water.

        Args:
            idx_towns (np.ndarray): Indices of towns.
            flow_reuse (np.ndarray): Flow data for reclaimed water.

        Returns:
            np.ndarray: Calculated energy savings for each town.
        """

    # TODO: cast to array in the input script
    c_source_substitute_per_year = np.array(usin.c_source_substitute) * 365
    energy_savings: np.ndarray = flow_reuse * c_source_substitute_per_year[idx_towns]

    return energy_savings


def calculate_environmental_savings(flow_reuse: np.ndarray, flow_discharged: np.ndarray) -> np.ndarray:
    """
        Calculate environmental savings obtained by reducing discharged water through reuse.

        Args:
            flow_reuse (np.ndarray): Flow data for reclaimed water.
            flow_discharged (np.ndarray): Flow data for discharged water.

        Returns:
            np.ndarray: Calculated environmental savings achieved by reducing discharged water through reuse.
        """
    environmental_savings: np.ndarray = (flow_reuse - flow_discharged) * usin.ww_treatment_after_discharge * 365

    return environmental_savings

# REVISE
def costs_freshwater_without_reuse_factory() -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
        Create a function to calculate costs of freshwater without reuse based on provided parameters.

        Returns:
            Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]: A function to calculate costs of freshwater without reuse.
        """
    # TODO: cast lists to arrays already in the input script
    water_consumption_arr = np.array(usin.water_consumption)  # (n_towns,)
    losses_distribution_arr = np.array(usin.losses_distribution)  # (n_towns,)
    portion_s_arr = np.array(
        [usin.portion_s1, usin.portion_s2, usin.portion_s3]).T  # (n_towns, 3)
    losses_transmission_arr = np.array(usin.losses_transmission)  # (n_towns,)
    c_prod_s = np.array(
        [usin.c_prod_s1, usin.c_prod_s2, usin.c_prod_s3]).T  # (n_towns, 3)
    c_tran_s = np.array(
        [usin.c_tran_s1, usin.c_tran_s2, usin.c_tran_s3]).T  # (n_towns, 3)
    c_tran_adm_arr = np.array(usin.c_tran_adm)  # (n_towns,)
    c_dist_staff_arr = np.array(usin.c_dist_staff)  # (n_towns,)
    c_dist_energy_arr = np.array(usin.c_dist_energy)  # (n_towns,)
    c_dist_adm_arr = np.array(usin.c_dist_adm)  # (n_towns,)

    losses_distribution_arr_inv = 1 / (1 - losses_distribution_arr)
    losses_transmission_arr_inv = 1 / (1 - losses_transmission_arr)

    c_dist_cost_arr = c_dist_staff_arr + c_dist_energy_arr + c_dist_adm_arr

    def inner(population: np.ndarray, flow_urban_reuse: Optional[np.ndarray]) -> np.ndarray:
        # Volumes
        volume_consumption: np.ndarray = np.einsum(
            "ij,i->ij", population, water_consumption_arr) * 365  # (n_towns, n_outlets)
        if flow_urban_reuse is not None:
            # If there is urban reuse, then decrease volume_consumption by fixed percentage
            # TODO: this makes no sense, as the percentage is fixed and not dependent on the
            # actual urban reuse
            c_flow_urban_reuse = np.ones(volume_consumption.shape[1])
            c_flow_urban_reuse[flow_urban_reuse > 0] = 1 - usin.urb_irrig_use_coef
            volume_consumption = np.einsum("ij,j->ij", volume_consumption, c_flow_urban_reuse)
        volume_consumption_s = np.einsum("ij,ik->ijk", volume_consumption,
                                         portion_s_arr * 1e-2)  # (n_towns, n_outlets, 3)
        volume_dist_s: np.ndarray = np.einsum("ijk,i->ijk", volume_consumption_s,
                                              losses_distribution_arr_inv)  # (n_towns, n_outlets, 3)
        volume_tran_s: np.ndarray = np.einsum("ijk,i->ijk", volume_dist_s,
                                              losses_transmission_arr_inv)  # (n_towns, n_outlets, 3)

        # Costs: transmission:
        c_tran_all_s = c_tran_s + c_tran_adm_arr[:, np.newaxis]
        total_costs_tran = np.einsum("ijk,ik->j", volume_tran_s, c_tran_all_s)

        # Costs: production
        total_costs_prod = np.einsum("ijk,ik->j", volume_tran_s, c_prod_s)

        # Costs: distribution
        total_costs_dist = np.einsum("ijk,i->j", volume_dist_s, c_dist_cost_arr)

        # Costs: total
        total_costs_water_per_subcatch_without_reuse: np.ndarray = total_costs_prod + \
                                                                   total_costs_tran + total_costs_dist

        return total_costs_water_per_subcatch_without_reuse

    return inner


COSTS_FRESHWATER_CALCULATOR = costs_freshwater_without_reuse_factory()


def calc_pumping_reuse_costs(radius_buffer: np.ndarray, flow_wwtp: np.ndarray, type_reuse) -> np.ndarray:
    """
        Calculate pumping costs for water reuse based on given parameters.

        Args:
            radius_buffer (np.ndarray): Radius buffer data.
            flow_wwtp (np.ndarray): Flow data from wastewater treatment plants.
            type_reuse: Data indicating the type of reuse.

        Returns:
            np.ndarray: Calculated pumping costs for water reuse.
        """

    costs_pumping_reuse_year = np.zeros(radius_buffer.shape)
    for i, type_reuse_i in enumerate(type_reuse):
        if type_reuse_i in {ReuseType.AGRICULTURAL, ReuseType.URBAN}:
            distance = radius_buffer[i]
            delta_z = (usin.max_slope_area / 100) * distance
            flow = flow_wwtp[i] / 24
            velocity = (flow / (np.pi * (usin.diameter ** 2) / 4)) / 3600
            linear_headlosses = usin.f * (distance / usin.diameter) * \
                                ((velocity ** 2) / (2 * 9.81))
            local_losses = usin.local_losses_coef * distance
            head_to_be_saved = delta_z + usin.h_suction + usin.h_outlet + \
                               usin.h_reservoir + linear_headlosses + local_losses
            hydraulic_power = (flow * usin.density * 9.81 *
                               head_to_be_saved) / (3.6 * (10 ** 6))  # KW
            required_energy = usin.pump_operation_time * hydraulic_power  # kwh
            cost_of_energy_month = required_energy * usin.price_average_kwh * 30
            costs_with_fixed = cost_of_energy_month + usin.fixed_cost
            taxes = costs_with_fixed * usin.tax_percent
            total_costs_month = costs_with_fixed + taxes
            total_costs_year = 12 * total_costs_month
            costs_pumping_reuse_year[i] = total_costs_year

    return costs_pumping_reuse_year


def onsite_treat_costs(total_pop: int, cluster_pop: int) -> tuple[int, int, int]:
    """
        Calculate the costs associated with onsite wastewater treatment systems.

        Args:
            total_pop (int): Total population in the area.
            cluster_pop (int): Population connected to a centralized wastewater treatment system.

        Returns:
            tuple[int, int, int]: Tuple containing investment costs, operation costs per year, and onsite population.
        """
    # assumptions
    people_by_unit = 6.2 * 4  # person/septictank 4 people/building
    empty_frequency_year = 2  # times/year/septictank
    empty_costs = 1500  # ILS/time
    construction_costs = 8500  # ILS/unit

    onsite_pop = total_pop - cluster_pop
    num_units = onsite_pop / people_by_unit
    operation_costs_year = empty_frequency_year * empty_costs * num_units
    investment_costs = construction_costs * num_units

    return investment_costs, operation_costs_year, onsite_pop

def calculate_reuse(idx_nodes: np.ndarray, flows_available: np.ndarray,
                    interpolators: dict[int, dict[str, DataInterpolator]]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the parameters for reuse based on available flows and provided interpolators.

    This function iterates through available nodes and their corresponding flow data to determine
    the reuse parameters such as radius, type of reuse, and required areas by utilizing provided interpolators.

    Args:
        idx_nodes (np.ndarray): Array of indices representing nodes.
        flows_available (np.ndarray): Array containing available flow data.
        interpolators (dict[int, dict[str, DataInterpolator]]): Dictionary of interpolators for each node.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing arrays representing
        radiuses of reuse, applied flows for reuse, types of reuse, and areas required for reuse.
    """

    radiuses_reuse = np.zeros(flows_available.shape)
    flows_applied_reuse = np.zeros(flows_available.shape)
    types_reuse = np.full(idx_nodes.shape, ReuseType.NO_REUSE)
    areas_reuse = np.zeros(flows_available.shape)

    # Iterate through nodes and available flow data to calculate reuse parameters
    for i, (idx_node, flow) in enumerate(zip(idx_nodes, flows_available)):
        if flow == 0.0:
            # Available flow is zero, indicating a pumping station node
            continue
        # Retrieve parameters for reuse using the interpolator for the current node
        radius_min_winner, type_reuse_winner, area_required = get_reuse_from_interpolator(
            flow, interpolators[idx_node])

        # Update arrays with calculated reuse parameters for each node
        radiuses_reuse[i] = radius_min_winner
        types_reuse[i] = type_reuse_winner
        areas_reuse[i] = area_required

        # Applied reuse flow is set equal to the available flow for the current node
        flows_applied_reuse[i] = flow

    return radiuses_reuse, flows_applied_reuse, types_reuse, areas_reuse


def get_reuse_from_interpolator(flow_available: float, interpolators: dict[str, DataInterpolator]) -> Tuple[
    float, ReuseType, float]:
    """
        Determine the optimal reuse parameters based on the available flow and the interpolators provided.

        This function iterates over known types of reuse (e.g., Agricultural or Urban), calculates the required area for
        a given flow based on ideal operational values, and uses interpolators to estimate the minimum radius needed
        for the calculated area.

        Args:
            flow_available (float): Available flow for reuse.
            interpolators (dict[str, DataInterpolator]): Dictionary of interpolators for different types of reuse.

        Returns:
            Tuple[float, ReuseType, float]: A tuple containing the optimal minimum radius, the type of reuse chosen
            (Agricultural or Urban), and the corresponding area required for the chosen reuse type.
        """
    # Dictionary defining optimal operational values for different types of reuse
    aap = {
        ReuseType.AGRICULTURAL: usin.irr_oper_agr,
        ReuseType.URBAN: usin.irr_oper_urb
    }
    # Initializing variables for the winning radius and reuse type
    radius_min_winner, type_reuse_winner = float('inf'), None

    # Iterate over each type of reuse and calculate the corresponding area required
    for type_reuse, flow_per_area_optimal in aap.items():
        area_required = flow_available / flow_per_area_optimal
        interpolator = interpolators[f"area__{str(type_reuse)}_to_radius"]
        # Attempt to interpolate the minimum radius based on the required area
        try:
            radius_min = interpolator(area_required) # Assign a default value in case of interpolation failure
        except:
            radius_min = float('inf')

        # Determine the winning radius and type of reuse
        if radius_min < radius_min_winner:
            radius_min_winner = radius_min
            type_reuse_winner = type_reuse
            area_required_winner = area_required

    # Check if a winning reuse type was determined; raise an assertion error if not found
    assert type_reuse_winner is not None, "Probably all interpolators returned 'inf'."

    # Return the winning radius, type of reuse, and the corresponding area required for the chosen reuse type
    return radius_min_winner, type_reuse_winner, area_required_winner


def get_reuse_network_length(idx_nodes: np.ndarray, radiuses: np.ndarray,
                             interpolators: dict[int, dict[str, DataInterpolator]]) -> np.ndarray:
    """
        Calculate the length of reuse network based on given radiuses and interpolators.

        This function iterates through the nodes' indices and corresponding radiuses
        to determine the length of the reuse network using provided interpolators.

        Args:
            idx_nodes (np.ndarray): Array of indices representing nodes.
            radiuses (np.ndarray): Array containing radiuses for the reuse network.
            interpolators (Dict[int, Dict[str, DataInterpolator]]): Dictionary of interpolators for each node.

        Returns:
            np.ndarray: Array representing the length of the reuse network for each node.
        """

    # Calculate the length of reuse network for each node using the provided interpolators
    network_length_reuse = np.array([interpolators[idx_node]["radius_to_reuse_network_length"](
        radius) for idx_node, radius in zip(idx_nodes, radiuses)])

    return network_length_reuse


def get_pop_per_town_per_buffer(idx_nodes: np.ndarray, radiuses: np.ndarray,
                                interpolators: dict[int, dict[str, DataInterpolator]]) -> np.ndarray:
    """
        Calculate the population per town per buffer based on indices, radiuses, and interpolators.

        This function calculates the population per town per buffer based on the nodes' indices,
        associated radiuses, and the provided interpolators.

        Args:
            idx_nodes (np.ndarray): Array of indices representing nodes.
            radiuses (np.ndarray): Array containing radiuses for the buffers.
            interpolators (Dict[int, Dict[str, DataInterpolator]]): Dictionary of interpolators for each node.

        Returns:
            np.ndarray: Array representing the population per town per buffer.
        """

    # Initialize an array to store population per town per buffer
    pop_per_town_per_buffer = np.full((4, idx_nodes.shape[0]), np.nan)

    # Iterate through each town and buffer to calculate the population
    for i_town in range(pop_per_town_per_buffer.shape[0]):
        for i_buffer, (i_node, radius_buffer) in enumerate(zip(idx_nodes, radiuses)):
            # Calculate population using the appropriate interpolator for each town
            pop_per_town_per_buffer[i_town, i_buffer] = interpolators[i_node][f"radius_to_population__town_{i_town}"](
                radius_buffer)

    return pop_per_town_per_buffer

# REVISE: get the number of towns somehow from input
def get_pop_per_town_per_subcatchment(data_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
        Generate an array representing the population per town per subcatchment.

        Args:
            data_dict (Dict[str, np.ndarray]): Dictionary containing relevant data arrays.

        Returns:
            np.ndarray: Array representing population per town per subcatchment.
        """
    # TODO: don't hardcode the number of towns
    # Define the number of towns (Assuming it is hardcoded for now)
    n_towns = 4

    # Initialize an array to store population per town per subcatchment
    pop_per_town_per_subcatchment = np.zeros((n_towns, data_dict["idx_node"].shape[0]))

    # Iterate through each town to populate the array with population data
    for i_town in range(n_towns):
        pop_per_town_per_subcatchment[i_town,
        :] = data_dict[f"population_served__town_{i_town}"]

    return pop_per_town_per_subcatchment


def calculate_reuse_reservoir_costs(flow: np.ndarray, type_reuse: np.ndarray) -> np.ndarray:
    """
        Calculate the costs of reservoirs for agricultural and urban reuse based on given flow and reuse types.

        Args:
            flow (np.ndarray): Array containing flow data.
            type_reuse (np.ndarray): Array containing type of reuse data.

        Returns:
            np.ndarray: Array representing the costs of reservoirs for each entry based on flow and reuse type.
        """

    # Calculate the scale factor for reservoir costs based on known data
    sc_reservoir = calculate_scale_factor(Inv_1=usin.known_inv_cost_reservoir_1,
                                          Inv_2=usin.known_inv_cost_reservoir_2,
                                          Q1=usin.known_reservoir_volume_1,
                                          Q2=usin.known_reservoir_volume_2)
    # Initialize an array to store reuse reservoir costs
    reuse_reservoir_costs = np.zeros(flow.shape)

    # Calculate costs for agricultural and urban reuse
    idx_agri_or_urb = (type_reuse == ReuseType.AGRICULTURAL) | (type_reuse == ReuseType.URBAN)
    reuse_reservoir_costs[idx_agri_or_urb] = ((flow[idx_agri_or_urb] * (
                                                             1.25 / usin.known_reservoir_volume_1)) ** (
                                                         sc_reservoir)) * usin.known_inv_cost_reservoir_1
    return reuse_reservoir_costs


def calculate_flow_available_reuse(flow_wwtp: np.ndarray) -> np.ndarray:
    return flow_wwtp.copy()


def calculate_flow_discharged(flow_available_reuse: np.ndarray, flow_applied_reuse: np.ndarray) -> np.ndarray:
    """
        Calculate the available flow for reuse based on the given flow from WWTP.

        Args:
            flow_wwtp (np.ndarray): Array containing flow data from WWTP.

        Returns:
            np.ndarray: Copy of the input array representing the available flow for reuse.
        """
    flow_discharged: np.ndarray = flow_available_reuse - flow_applied_reuse
    return flow_discharged


def calculate_decentralization_degree(flow_wwtp: np.ndarray, pop_per_subcatchment_wwtp: np.ndarray) -> tuple[
    float, float]:
    """
        Calculate two measures of decentralization based on WWTP flow and population data.

        Args:
            flow_wwtp (np.ndarray): Array containing flow data from WWTP.
            pop_per_subcatchment_wwtp (np.ndarray): Array containing population data per subcatchment.

        Returns:
            tuple[float, float]: Tuple containing two measures of decentralization (Eggimann and Huang).
        """

    # Calculate total flow from WWTPs
    total_flow = flow_wwtp.sum()

    # Calculate total number of households
    numbers_households = pop_per_subcatchment_wwtp * (1 / usin.inhabit_per_household)
    total_households = numbers_households.sum()

    # Calculate sum of flow per household
    sum_flow_per_hh = (flow_wwtp / numbers_households).sum()

    # Calculate Eggimann's decentralization degree
    dec_degree_eggiman = (total_flow - sum_flow_per_hh) / total_flow

    # Calculate sum of flow by households
    sum_flow_by_hh = (flow_wwtp * numbers_households).sum()

    # Calculate Huang's decentralization degree
    dec_degree_huang = sum_flow_by_hh / (total_flow * total_households)

    return dec_degree_eggiman, dec_degree_huang


def split_population_wwtp_wwps(pop_per_town_per_subcatchment: np.ndarray, node_types: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    """
        Split population data per town per subcatchment into WWTP and WWPS categories.

        Args:
            pop_per_town_per_subcatchment (np.ndarray): Population data per town per subcatchment.
            node_types (np.ndarray): Array containing node types.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing populations for WWTPs and WWPSs.
        """
    # Initialize arrays to store populations for WWTPs and WWPSs
    pop_per_town_per_subcatchment_wwtp = np.zeros(pop_per_town_per_subcatchment.shape)
    pop_per_town_per_subcatchment_wwps = np.zeros(pop_per_town_per_subcatchment.shape)

    # Check for WWTP nodes and assign population data accordingly
    idx_wwtp = node_types == NodeType.WWTP
    pop_per_town_per_subcatchment_wwtp[:, idx_wwtp] = pop_per_town_per_subcatchment[:, idx_wwtp]

    # Check for WWPS nodes and assign population data accordingly
    idx_wwps = node_types == NodeType.WWPS
    pop_per_town_per_subcatchment_wwps[:, idx_wwps] = pop_per_town_per_subcatchment[:, idx_wwps]

    return pop_per_town_per_subcatchment_wwtp, pop_per_town_per_subcatchment_wwps


def Join_Calcul(
        data_dict: dict[str, np.ndarray],
        interpolators: dict[int, dict[str, DataInterpolator]],
        store_node_results: bool = True,
        store_total_results: bool = True
) -> tuple[float, float, Optional[dict[str, Scalar]], Optional[dict[str, np.ndarray]]]:
    """
        Perform comprehensive calculations related to wastewater and freshwater management.

        This function handles various computations including costs, benefits, coverage, and metrics
        related to wastewater treatment, reuse, freshwater management, and onsite treatment.

        Args:
            data_dict (dict[str, np.ndarray]): Dictionary containing various data arrays.
            interpolators (dict[int, dict[str, DataInterpolator]]): Dictionary of interpolators.
            store_node_results (bool, optional): Flag to indicate storing node-wise results (default is True).
            store_total_results (bool, optional): Flag to indicate storing total results (default is True).

        Returns:
            tuple[float, float, Optional[dict[str, Scalar]], Optional[dict[str, np.ndarray]]]: A tuple containing
            benefits over costs ratio, coverage in catchment, results for total metrics, and node-wise results.
        """
    
    ww_pipe_inv_costs_old = np.array(
        [PIPELINE_REUSE_INV_COST_CALCULATOR(pipe_length) for pipe_length in data_dict["network_length"]])

    # ***** HOSSEIN CODE network  cost swimm *****
    ww_pipe_inv_costs = int(network_cost_swimm(
        config.NETWORK_COST_SWIMM_FILE_PATH,
        [
            data_dict['idx_node'][0]
            #        data_dict['idx_node'][1],
            #        data_dict['idx_node'][2],
            #        data_dict['idx_node'][3]
        ]
    ))

    # REVISE: Keep this?
    #    print("Estimated Maria (+" + str(len(data_dict['idx_node'])) + " WWTPs): " + str(sum(ww_pipe_inv_costs_old)) +
    #          ". Estimated Hossein: (4 WWTPs):" + str(ww_pipe_inv_costs))

    # *******************************************

    pop_per_town_per_subcatchment = get_pop_per_town_per_subcatchment(data_dict)

    pop_per_town_per_subcatchment_wwtp, pop_per_town_per_subcatchment_wwps = split_population_wwtp_wwps(
        pop_per_town_per_subcatchment, data_dict["node_type"])

    pop_per_subcatchment: np.ndarray = pop_per_town_per_subcatchment.sum(axis=0)
    pop_per_subcatchment_wwtp: np.ndarray = pop_per_town_per_subcatchment_wwtp.sum(axis=0)
    pop_per_subcatchment_wwps: np.ndarray = pop_per_town_per_subcatchment_wwps.sum(axis=0)

    flow_wwtp, flow_wwtp_peak = calculate_flow_based_on_population(
        pop_per_town_per_subcatchment_wwtp)
    flow_wwps, flow_wwps_peak = calculate_flow_based_on_population(
        pop_per_town_per_subcatchment_wwps)

    # Calculation of benefit of connection fees
    benefit_connection = calc_benefit_connections(pop_per_catchment=pop_per_subcatchment_wwtp)

    benefit_ww_fees = calc_benefit_ww_yearly_fees(
        pop_per_town_per_subcatchment=pop_per_town_per_subcatchment_wwtp)

    flow_available_reuse = calculate_flow_available_reuse(flow_wwtp)

    radiuses_reuse, flow_applied_reuse, types_reuse, areas_per_reuse_per_buffer = calculate_reuse(
        data_dict["idx_node"], flow_available_reuse, interpolators)

    flow_discharged = calculate_flow_discharged(flow_available_reuse, flow_applied_reuse)

    reuse_network_length_theoric = get_reuse_network_length(
        data_dict["idx_node"], radiuses_reuse, interpolators)

    reuse_network_costs = calculate_reuse_network_costs(
        reuse_network_length_theoric, types_reuse)

    # WASTEWATER TREATMENT/RECLAMATION SYSTEM RELATED

    # Costs of treatment plants
    treatment_costs = calculate_wwtp_costs(flow_wwtp_peak, types_reuse)

    capex_costs_wwps, oem_costs_wwps = calculate_wwps_costs(
        flow_wwps_peak, data_dict["node_type"], data_dict["pumping_length_pump"],
        data_dict["pumping_length_grav"], data_dict["pumping_height"])
    total_costs_wwps = capex_costs_wwps + oem_costs_wwps

    land_cost_by_wwtp = get_land_cost_by_wwtp(
        data_dict["land_cost"], flow_wwtp_peak, types_reuse)
    land_costs_wwps = calculate_land_cost_wwps(data_dict["land_cost"])

    # Reservoir costs
    reuse_reservoir_costs = calculate_reuse_reservoir_costs(
        flow_wwtp_peak, types_reuse)

    # Calculate flow for agriculture, reuse and no reuse
    flow_agriculture_reuse, flow_urban_reuse, flow_no_reuse, applied_flow_agriculture, applied_flow_urban = calculate_flows_reuse(
        area_reuse=areas_per_reuse_per_buffer,
        flow_reuse=flow_applied_reuse,
        types_reuse=types_reuse)

    costs_reuse_pumping = calc_pumping_reuse_costs(radius_buffer=radiuses_reuse, flow_wwtp=flow_wwtp_peak,
                                                   type_reuse=types_reuse)

    # Costs with onsite sanitation
    total_pop_served = pop_per_subcatchment_wwtp.sum()
    onsite_inv_costs, onsite_oem_year, onsite_pop = onsite_treat_costs(
        usin.total_population, total_pop_served)

    pv_oem_onsite = present_value(usin.DR, usin.n, onsite_oem_year)

    # Benefits onsite sanitation
    benefit_connection_onsite = (onsite_pop / int(6.2)) * 1500
    benefit_ww_fees_onsite_year = onsite_pop * 0.07 * 1.5 * 365

    pv_fees_onsite = present_value(usin.DR, usin.n, benefit_ww_fees_onsite_year)

    # FRESHWATER MANAGEMENT RELATED
    pop_per_town_per_buffer = get_pop_per_town_per_buffer(
        data_dict["idx_node"], radiuses_reuse, interpolators)

    total_costs_freshwater_without_reuse = COSTS_FRESHWATER_CALCULATOR(
        pop_per_town_per_buffer, None)

    total_costs_water_per_buffer = COSTS_FRESHWATER_CALCULATOR(
        pop_per_town_per_buffer, flow_urban_reuse)

    benefit_freshwater_per_subcatch = benefits_freshwater_with_reuse(
        pop_per_town_per_buffer, flow_urban_reuse=flow_urban_reuse)

    benefits_agriculture, benefits_urban = benefits_reuse_selling(flow_agriculture_reuse=flow_agriculture_reuse,
                                                                  flow_urban_reuse=flow_urban_reuse)

    # Benefit as energy savings
    benefit_energy_savings = calculate_energy_savings(
        data_dict["idx_town"], flow_reuse=flow_applied_reuse)

    # Benefit as environmental savings
    environmental_savings = calculate_environmental_savings(flow_reuse=flow_applied_reuse,
                                                            flow_discharged=flow_discharged)

    # Total benefits
    total_benefit_ww = benefit_connection.sum() + present_value(DR=usin.DR,
                                                                n=usin.n, c_op_year=benefit_ww_fees.sum())

    benefits_reclaimed_water_management = present_value(DR=usin.DR, n=usin.n, c_op_year=benefits_agriculture.sum()) + \
                                          present_value(DR=usin.DR, n=usin.n, c_op_year=benefits_urban.sum())
    benefits_freshwater_management = present_value(
        DR=usin.DR, n=usin.n, c_op_year=benefit_freshwater_per_subcatch.sum())
    benefits_energy_savings = present_value(
        DR=usin.DR, n=usin.n, c_op_year=benefit_energy_savings.sum())
    benefits_env_savings = present_value(
        DR=usin.DR, n=usin.n, c_op_year=environmental_savings.sum())

    total_benefits_onsite = benefit_connection_onsite + pv_fees_onsite

    total_benefits = total_benefit_ww + benefits_reclaimed_water_management + benefits_freshwater_management + benefits_energy_savings + benefits_env_savings \
                     + total_benefits_onsite

    # Total costs

    total_costs_ww = ww_pipe_inv_costs_old.sum() + treatment_costs.sum() + land_cost_by_wwtp.sum()  # ww_pipe_inv_costs_old is with Maria's method
    total_costs_wwps = total_costs_wwps.sum() + land_costs_wwps.sum()

    total_costs_reclaimed_ww = reuse_reservoir_costs.sum() + reuse_network_costs.sum() + \
                               present_value(DR=usin.DR, n=usin.n, c_op_year=costs_reuse_pumping.sum())

    total_costs_freshwater_mngt = present_value(
        DR=usin.DR, n=usin.n, c_op_year=total_costs_water_per_buffer.sum())

    total_costs_onsite = onsite_inv_costs + pv_oem_onsite

    total_costs = total_costs_ww + total_costs_reclaimed_ww + total_costs_freshwater_mngt \
                  + total_costs_onsite + total_costs_wwps

    # Calculation of coverage with cluster level sanitation
    coverage_region = coverage_check(pop_per_subcatchment_wwtp, usin.total_population)
    coverage_catchment = coverage_check(pop_per_subcatchment_wwtp, usin.population_catchment)

    # Calculation of decentralization degree of the layout
    idx_wwtp = data_dict["node_type"] == NodeType.WWTP
    cent_degree_eggiman, dec_degree_huang = calculate_decentralization_degree(
        flow_wwtp[idx_wwtp], pop_per_subcatchment_wwtp[idx_wwtp])

    benefits_over_costs = total_benefits / total_costs

    assert np.all(flow_no_reuse == 0.0)

    # Saving results to excel csv file
    if store_node_results:
        results_nodes = {
            'Node index': data_dict["idx_node"],
            'Node type': data_dict["node_type"],
            'Geographic coordinates (UTM)': data_dict["coordinates"],
            'Network length of ww pipeline (m)': data_dict["network_length"],
            'Cost of wastewater pipeline (ILS)': ww_pipe_inv_costs,
            'Population supplied by one WWTP (inhab)': pop_per_subcatchment_wwtp,
            # 'Population per town (inhab)': pop_per_town,
            'Treated flow (m3/d)': flow_wwtp,
            'Peak treated flow (m3/d)': flow_wwtp_peak,
            'Benefit from connections (ILS)': benefit_connection,
            'Benefit from ww fees (ILS / year)': benefit_ww_fees,
            # 'Area per type of reuse per subcatchment (m2 - Agriculture, Urban, No reuse)': areas_per_reuse_per_buffer,
            'Type of treatment of each wwtp': types_reuse,
            'Radius for reuse at each wwtp (m)': radiuses_reuse,
            'Flow applied in irrigation (m3/d)': flow_applied_reuse,
            'Length of reuse pipeline distribution (m)': reuse_network_length_theoric,
            'Costs of reuse network pipeline (ILS)': reuse_network_costs,
            'Treatment costs (ILS)': treatment_costs,
            'Land costs in each wwtp area (ILS / m2)': data_dict["land_cost"],
            'Land purchase costs at each WWTP (ILS) ': land_cost_by_wwtp,
            'Costs of reservoir investment (ILS)': reuse_reservoir_costs,
            'Flow used for agriculture (m3 / d)': flow_agriculture_reuse,
            'Flow used for urban reuse (m3/d)': flow_urban_reuse,
            'Flow not used for reuse (m3/d)': flow_no_reuse,
            'Ratio m3 / m2  used for agriculture in each buffer': applied_flow_agriculture,
            'Ratio m3/m2 used for urban reuse in each buffer': applied_flow_urban,
            'Costs with reuse pumping (ILS / year)': costs_reuse_pumping,
            'Costs in the buffers with water management - WITHOUT reuse (ILS/year)': total_costs_freshwater_without_reuse,
            'Costs in the buffers with water management - considering reuse (ILS/year)': total_costs_water_per_buffer,
            'Benefit received from the benefit of freshwater selling (ILS/year)': benefit_freshwater_per_subcatch,
            'Benefit received (in a year) for reuse selling for agriculture (ILS / year)': benefits_agriculture,
            'Benefit received (in a year) for reuse selling for urban use  (ILS/year)': benefits_urban,
            'Equivalent in energy savings (ILS/year)': benefit_energy_savings,
            'Equivalent in wastewater tariff saved (ILS/year)': environmental_savings,
            # WWTPS related
            'Population supplied by one WWPS (inhab)': pop_per_subcatchment_wwps,
            'Pumped flow (m3/d)': flow_wwps,
            'Peak pumped flow (m3/d)': flow_wwps_peak,
            'Connects to': data_dict["connects_to"],
            'Land purchase costs at each WWPS (ILS) ': land_costs_wwps,
            "Pumping pipeline length (pump) [m]": data_dict["pumping_length_pump"],
            "Pumping pipeline length (gravity) [m]": data_dict["pumping_length_grav"],
            "Pumping height [m]": data_dict["pumping_height"],
            'Wastewater pumping costs (ILS PV)': total_costs_wwps,
            "WWPS investment cost [ILS]": capex_costs_wwps,
            "WWPS OEM costs (PV) [ILS]": oem_costs_wwps,
        }
    else:
        results_nodes = None

    if store_total_results:
        results_total = {
            'Total population supplied (inhab)': total_pop_served,
            'Benefit wastewater (Total with PV summed)': total_benefit_ww,
            'Total benefit reclaimed wastewater (total PV)': benefits_reclaimed_water_management,
            'Total benefit freshwater management (total PV)': benefits_freshwater_management,
            'Total benefits of energy savings (total PV)': benefits_energy_savings,
            'Total benefits of environmental savings (total PV)': benefits_env_savings,
            'Total benefits  (ILS)': total_benefits,
            'Total costs with wastewater (ILS PV)': total_costs_ww,
            'Total costs with reclaimed ww management (ILS)': total_costs_reclaimed_ww,
            'Total costs with costs freshwater in the buffer (ILS)': total_costs_freshwater_mngt,
            'Total costs (ILS)': total_costs,
            'Costs - benefits (ILS)': total_costs - total_benefits,
            'Coverage (region)': coverage_region,
            'Coverage (town)': coverage_catchment,
            'Benefits/costs': benefits_over_costs,
            'Onsite construction costs ': onsite_inv_costs,
            'Onsite OeM costs (year) ': onsite_oem_year,
            'Onsite OeM costs (total) ': pv_oem_onsite,
            'Benefits onsite treatment (connection fee) ': benefit_connection_onsite,
            'Benefits onsite treatment (sanitation fee) ': pv_fees_onsite,
            'Population supplied with onsite treatment (inhabitants)': onsite_pop,
            'Total wastewater pumping costs (ILS PV)': total_costs_wwps.sum(),
            'Total land purchase costs at each WWPS (ILS) ': land_costs_wwps.sum(),
            'Centralization degree Eggiman': cent_degree_eggiman,
            'Centralization degree Huang': dec_degree_huang,
            'Number activated': data_dict['node_type'].size,
            'Number WWTP': (data_dict['node_type'] == NodeType.WWTP).sum(),
            'Number WWPS': (data_dict['node_type'] == NodeType.WWPS).sum(),
        }
    else:
        results_total = None

    return benefits_over_costs, coverage_catchment, results_total, results_nodes
