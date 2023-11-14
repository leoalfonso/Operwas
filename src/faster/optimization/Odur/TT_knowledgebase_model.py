import json
from typing import Dict, List


class Technology:
    id: str
    level: str
    name: str
    abbreviation: str
    removal_efficiencies: Dict[str, float]
    evaluation_scores: Dict[str, int]
    assembly_rules: Dict[str, int]
    capital_cost_coefficients: Dict[str, float]
    technical_evaluation_scores: Dict[str, int]
    environmental_evaluation_scores: Dict[str, int]
    economic_evaluation_scores: Dict[str, int]
    om_coefficients: Dict[str, float]
    energy_requirement_coefficient: Dict[str, float]
    land_requirement_coefficient: Dict[str, float]
    centralization_type: str


class Pollutant:
    id: str
    name: str
    abbreviation: str
    unit: str
    c_inf: int


class CostData:
    no_of_years: float
    discount_rate: float
    exchange_rate: float
    unit_cost_of_land: float
    unit_cost_of_energy: float
    pr: float


TECHNOLOGIES: Dict[str, Dict[str, Technology]] = {}
CENTRALIZATION_KEYED_TECHNOLOGY_DATA: Dict[str, Dict[str, Technology]] = {}
POLLUTANTS: Dict[str, Pollutant] = {}
RE_USE_STANDARDS: Dict[str, Dict[str, float]] = {}
RAW_COST_DATA = CostData()
CONFIG_LOADED: bool = False


def loadTechnologyData():
    """load our JSON file with technologies knowledge base data"""
    # if we already loaded the file, do nothing
    global CONFIG_LOADED
    if CONFIG_LOADED:
        return
    # otherwise, read in the file
    with open(r"D:\OP_pycharm\Operwas_pump\inputs\tt_knowledge_base\TT_module_knowledgebase.json") as f:
        data = json.load(f)

    # parse our json file
    # extract technology data
    knowledge_base_technologies: dict = data["technologies"]
    for centralization_type, system_data in knowledge_base_technologies.items():
        for level_name, level_data in system_data.items():
            system_level_technologies = None
            for tech_id, tech_data in level_data.items():
                tech = Technology()
                tech.id = tech_id
                tech.level = level_name
                tech.name = tech_data["name"]
                tech.abbreviation = tech_data["abbreviation"]
                tech.removal_efficiencies = tech_data["removal_efficiencies"]
                tech.assembly_rules = tech_data.get("assembly_rule", {})
                tech.capital_cost_coefficients = tech_data.get(
                    "capital_cost_coefficients", {}
                )
                tech.technical_evaluation_scores = tech_data.get(
                    "Technical_evaluation_scores", {}
                )
                tech.environmental_evaluation_scores = tech_data.get(
                    "Environmental_evaluation_scores", {}
                )
                tech.economic_evaluation_scores = tech_data.get(
                    "Economic_evaluation_scores", {}
                )
                tech.om_coefficients = tech_data.get("OM_coefficient", {})
                tech.energy_requirement_coefficient = tech_data.get(
                    "energy_requirement_coefficient", {}
                )
                tech.land_requirement_coefficient = tech_data.get(
                    "land_requirement_coefficient", {}
                )
                tech.centralization_type = centralization_type

                system_level_technologies = TECHNOLOGIES.get(level_name, {})
                system_level_technologies[tech_id] = tech
                TECHNOLOGIES[level_name] = system_level_technologies

                tech_data_for_centralization = CENTRALIZATION_KEYED_TECHNOLOGY_DATA.get(
                    centralization_type, {}
                )
                tech_data_for_centralization[tech_id] = tech
                CENTRALIZATION_KEYED_TECHNOLOGY_DATA[
                    centralization_type
                ] = tech_data_for_centralization

            centralization_level_technologies = TECHNOLOGIES.get(
                centralization_type, {}
            )
            centralization_level_technologies[level_name] = system_level_technologies
            TECHNOLOGIES[centralization_type] = centralization_level_technologies

    # extract pollutant data
    pollutants: dict = data["pollutants"]
    for pollutant_id, pollutant_data in pollutants.items():
        pollutant = Pollutant()
        pollutant.id = pollutant_id
        pollutant.name = pollutant_data["name"]
        pollutant.unit = pollutant_data["unit"]
        pollutant.c_inf = pollutant_data["c_inf"]

        POLLUTANTS[pollutant_id] = pollutant

    # extract re-use standards
    re_use_standards: dict = data.get("reuse")
    if re_use_standards is None:
        print("No re-use standards found in knowledge base file")
    else:
        for re_use_standard_id, re_use_standard_data in re_use_standards.items():
            RE_USE_STANDARDS[re_use_standard_id] = re_use_standard_data

    # prepare to extract cost data
    raw_cost_data: dict = data.get("cost data")
    if raw_cost_data is None:
        print("No cost data found in knowledge base file")
    else:
        global RAW_COST_DATA
        RAW_COST_DATA = raw_cost_data

    # set our flag to true
    CONFIG_LOADED = True


def getCostData(unit_land_cost: float) -> CostData:
    """return the cost data"""
    global RAW_COST_DATA

    cost_data = CostData()
    raw_cost_data: dict = RAW_COST_DATA
    if raw_cost_data is None:
        print("No cost data found in knowledge base file")
    else:
        present_value = raw_cost_data.get("Present Value", {"n": 0.0, "i": 0.0})
        cost_data.no_of_years = present_value["n"]
        cost_data.discount_rate = present_value["i"]
        cost_data.exchange_rate = raw_cost_data.get("Exchange rate")
        cost_data.unit_cost_of_land = unit_land_cost
        # TODO MD 5: I note that the unit land cost is still being calculated using this
        cost_data.unit_cost_of_energy = raw_cost_data.get("Unit Energy cost")
        """
        Calculate the Capital Recovery Factor using discount rate and number of operational years 
        Where:
            discount rate is the rate at which future cash flows are discounted to their present value
            number of years is the number of years over which the cash flows will occurs
        """
        cost_data.crf = cost_data.discount_rate / (
            1 - (1 + cost_data.discount_rate) ** (-cost_data.no_of_years)
        )
        cost_data.annuity_factor = (1-((1 + cost_data.discount_rate) ** (-cost_data.no_of_years)))/(
            cost_data.discount_rate
        )

    return cost_data
