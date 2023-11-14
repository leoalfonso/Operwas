import json
from typing import Tuple, List, Dict, Union

# TODO MD 4: ACTION- NOTE 1: Give new name for the library named "config".
from operwas.TT_knowledgebase_model import (
    loadTechnologyData,
    TECHNOLOGIES,
    CENTRALIZATION_KEYED_TECHNOLOGY_DATA,
    Technology,
    POLLUTANTS,
    RE_USE_STANDARDS,
    getCostData,
    CostData,
)

GENERATED_TRAINS: Dict[str, List[List[str]]] = {}
VALID_RE_USE_STANDARDS = [
    "Agricultural reuse standard",
    "Domestic non-potable reuse standard",
    "Urban reuse standard",
    "Environmental discharge standard ",
]

DECENTRALIZED = "Decentralized_Systems"
SEMI_CENTRALIZED = "Semi_centralized_systems"
ACTIVE_CENTRALIZATION_TYPE = None

INFO = 1
DEBUG = 2
VERBOSITY = INFO


def dprint(arg: str):
    if VERBOSITY == DEBUG:
        print(arg)


def getTechnologyData(tech_id: str) -> Technology:
    """return the technology data"""
    global CENTRALIZATION_KEYED_TECHNOLOGY_DATA
    return CENTRALIZATION_KEYED_TECHNOLOGY_DATA.get(ACTIVE_CENTRALIZATION_TYPE, {}).get(
        tech_id
    )


def continueTrain(prefix_multiplier: int, tech: Technology, train_text: str):
    prefix = "=" * prefix_multiplier
    if prefix_multiplier > 25:
        dprint(prefix + "> Reached max configured depth for train")
        return []

    trains = []

    assembly_rules = tech.assembly_rules
    # dprint(f"==> Found {len(assembly_rules.keys())} assembly rules for this technology")
    for assembly_rule, valid_here in assembly_rules.items():
        if valid_here == 0:
            continue

        tech_data = getTechnologyData(assembly_rule)
        if tech_data is None:
            # dprint(
            #     prefix
            #     + f"> Tech data for tech '{assembly_rule}' not found, skipping technology"
            # )
            continue

        level = tech_data.level
        if level == "preliminary_level" and assembly_rule == "Start":
            continue
        elif level == "primary_level" and assembly_rule == "No_pri":
            continue
        elif level == "secondary_level" and assembly_rule == "No_sec":
            continue
        elif level == "tertiary_level" and assembly_rule == "No_ter":
            continue
        elif level == "disinfection_level" and assembly_rule == "No_dis":
            continue

        # dprint(prefix + f"> Continuing train with technology '{assembly_rule}'")
        train_text += f" -> {assembly_rule}"
        # dprint(prefix + f"> Train is now: {train_text}")
        trains.append(train_text)

        # proceed = input("hit enter to continue train or input 'q' to quit: ")
        # if proceed == "q":
        #     exit(0)

        sub_trains = continueTrain(prefix_multiplier + 1, tech_data, train_text)
        trains.extend(sub_trains)

        train_text = train_text[: train_text.rfind(" -> ")]

        dprint("train complete")
    return trains

    """
    Split the train building into technology classifications, i.e semi-centralised & Decentralised
     in this way, we will have fewer technologies built at a time, becoz q_inf 
     will limit this in the assembly rule
     Say, if 1< q_inf< 100, build chain from decentralized, 
     if q_inf>100 build chains using semi-centralised
     So Qinf in location coordinates be used.

    """


def buildTrains(
    start_level: str, start_technologies: List[str], q_inf: float
) -> Tuple[bool, str, List[List[str]]]:
    """build trains of technologies"""
    #   Say, if 1< q_inf < 100, build chain from decentralized, if q_inf>100 build chains using semi-centralised
    centralization = None
    if q_inf >= 1 and q_inf <= 100:
        centralization = DECENTRALIZED
    elif q_inf > 100:
        centralization = SEMI_CENTRALIZED
    else:
        return (
            False,
            f"q_inf of {q_inf} is not in the range of 1 to 100 or above 100",
            None,
        )

    # TODO : what?
    print(
        f"for q_inf of {q_inf}, centralization type is {centralization}"
    )
    get_from_technologies = TECHNOLOGIES.get(centralization)
    if get_from_technologies is None:
        return (
            False,
            f"Centralization type '{centralization}' technologies not found",
            None,
        )

    global ACTIVE_CENTRALIZATION_TYPE
    ACTIVE_CENTRALIZATION_TYPE = centralization

    start = get_from_technologies.get(start_level)
    if start is None:
        return False, f"Start level '{start_level}' not found", None

    print(f"=> Building trains starting with level '{start_level}'")
    trains = []

    for start_technology in start_technologies:
        tech = start.get(start_technology)
        if tech is None:
            return (
                False,
                f"Start technology '{start_technology}' not found in start level: '{start_level}'",
                None,
            )

        if start_technology != "Start":
            print(f"==> Building trains starting with technology '{start_technology}'")
        else:
            print("==> Building trains starting at start technologies")

        tech_data = getTechnologyData(start_technology)
        if tech_data is None:
            return (
                False,
                f"Start technology '{start_technology}' not found in technology data",
                None,
            )

        if start_technology == "Start":
            train_text = ""
        else:
            train_text = start_technology

        tempTrains = continueTrain(3, tech_data, train_text)
        print(f"==> Found {len(tempTrains)} trains starting with '{start_technology}'")
        print(tempTrains[1])
        trains.extend(tempTrains)

    return True, None, trains


def calculate_effluent_conc(
    train_tech_ids: List[str], pollutant_id: str
) -> Tuple[bool, str, float]:
    """
    calculate effluent concentration for a given technology train

    C_eff_TT= C_inf * [(1-r1)*(1-r2)*(1-rn)]
    """
    pollutant = POLLUTANTS.get(pollutant_id)
    if pollutant is None:
        return False, f"Invalid pollutant '{pollutant_id}'", None

    # calculate effluent concentration
    # cumulative deductions
    cumulative_deductions = 1
    for tech_id in train_tech_ids:
        tech_data = getTechnologyData(tech_id)
        if tech_data is None:
            return (
                False,
                f"Train technology '{tech_id}' not found in technology data",
                None,
            )

        removal_efficiency = tech_data.removal_efficiencies.get(pollutant_id)
        if removal_efficiency is None:
            return (
                False,
                f"Removal efficiency for pollutant '{pollutant_id}' not found for technology '{tech_id}'",
                None,
            )

        cumulative_deductions = cumulative_deductions * (1 - removal_efficiency)

    effluent_conc = pollutant.c_inf * (cumulative_deductions)

    return True, None, effluent_conc


def dataCleanTrains(rawTrains: List[str]) -> List[List[str]]:
    """data clean trains of technologies"""
    cleaned = []
    for train in rawTrains:
        parts = train.split(" -> ")
        cleanedParts = []
        for part in parts:
            if part == "":
                continue
            cleanedParts.append(part.strip())
        cleaned.append(cleanedParts)

    return cleaned


def screen_trains_by_reuse_standard(
    trains: List[List[str]], reuse_standard_data: Dict[str, Dict[str, float]]
) -> Tuple[List[str], List[Dict[str, float]]]:
    print("filtering trains")
    ## Take in standards for each location, we have 8 locations##
    ## For each of the location, there may be 1 or more of the reuse, so we have to try and allow the buildinging of trains for all the reuse in the area##
    use_key = "c_eff"
    validTrains = []
    effs = []
    for train in trains:
        filter = False
        eff = {}
        for pollutant_id, pollutant_data in reuse_standard_data.items():
            c_inf = pollutant_data.get(use_key)
            if c_inf is None:
                dprint(
                    f"Pollutant '{pollutant_id}' {use_key} not found, skipping train"
                )
                filter = True
                continue

            calculated, reason, c_for_train = calculate_effluent_conc(
                train, pollutant_id
            )
            if not calculated:
                dprint(reason)
                filter = True
                continue

            if c_for_train > c_inf:
                dprint(
                    f"Train '{train}' failed for pollutant '{pollutant_id}' with effluent concentration of {c_for_train}"
                    + f" which is higher than the standards minimum of {c_inf}"
                )
                filter = True
                continue

            eff[f"{pollutant_id}"] = c_for_train

        if not filter:
            validTrains.append(train)
            effs.append(eff)

    return validTrains, effs


def calculateLandRequirement(
    train_tech_ids: List[str], q_inf: float, unit_cost_of_land: float
) -> Tuple[bool, str, Dict[str, float]]:
    """
    FIRSTLY: The land requirement of each technology is calculated using land requirement coefficients,
             The total land requirement of a train is obtained as the sum of the TTs' individual UP technology  req
    plus 15% for auxiliary facilities (such as roads, etc)
            Land_requirement of UP_i = l1 * Qinf^l2
            Land_requirement of TT = 1.15* Sum(Land_req of UPs in TT)
    THEN: The cost calculated by multiplying the land requirement of TT * the unit land cost in the area (from V1)
    Cost of land of _TT = 1.15* Sum(Land_req of UPs in TT) * (Unit land cost)
    """
    land_requirement = 0
    for tech_id in train_tech_ids:
        tech_data = getTechnologyData(tech_id)
        if tech_data is None:
            return (
                False,
                f"Train technology '{tech_id}' not found in technology data",
                None,
            )

        l1 = tech_data.land_requirement_coefficient.get("l1")
        l2 = tech_data.land_requirement_coefficient.get("l2")
        if l1 is None or l2 is None:
            return (
                False,
                f"Land requirement coefficients not found for technology '{tech_id}'",
                None,
            )
        # Land requirement in hectares
        land_requirement = (
            land_requirement + l1 * q_inf**l2
        )
    # CALCULATE TOTAL TREATMENT TRAIN LAND REQUIREMENT
    land_requirement = (
        1.15 * land_requirement * 1000
    )  # Land requirement converted to m2

    # CALCULATE COST OF LAND PURCHASE FOR THE TT (ILS)
    # The unit land cost is in ILS/m2 got from running operwa_library
    land_requirement_cost = land_requirement * unit_cost_of_land  # land cost  ILS

    return (
        True,
        None,
        {
            "land_requirement": land_requirement,
            "land_requirement_cost": land_requirement_cost,
        },
    )


def calculateCapitalCost(
    train_tech_ids: List[str], q_inf: float, exchange_rate: float
) -> Tuple[bool, str, float]:
    """
    calculate capital cost for a given technology train

    C_cap_TT = Sum(C_cap_i)
    """
    capital_cost = 0
    for tech_id in train_tech_ids:
        tech_data = getTechnologyData(tech_id)
        if tech_data is None:
            return (
                False,
                f"Train technology '{tech_id}' not found in technology data",
                None,
            )

        c1 = tech_data.capital_cost_coefficients.get("c1")
        c2 = tech_data.capital_cost_coefficients.get("c2")
        if c1 is None or c2 is None:
            return (
                False,
                f"Capital cost coefficients not found for technology '{tech_id}'",
                None,
            )
        # Capital cost of a TT is calculated in USD2006/year
        capital_cost = capital_cost + c1 * q_inf**c2

    # NOW CAPITAL COST IS CONVERTED TO ILS
    # Using exchange rate of 4.494 as of 2006 to capture in all components of inflation
    capital_cost = capital_cost * exchange_rate

    return True, None, capital_cost  # ILS/YEAR


def calculateOperationalCost(
    train_tech_ids: List[str], q_inf: float, exchange_rate: float, annuity_factor: float
) -> Tuple[bool, str, float]:
    """
    calculate operational cost for a given technology train

    C_op_TT = Sum(C_op_i)
    """
    operational_cost = 0
    for tech_id in train_tech_ids:
        tech_data = getTechnologyData(tech_id)
        if tech_data is None:
            return (
                False,
                f"Train technology '{tech_id}' not found in technology data",
                None,
            )

        m1 = tech_data.om_coefficients.get("m1")
        m2 = tech_data.om_coefficients.get("m2")
        if m1 is None or m2 is None:
            return (
                False,
                f"Operational cost coefficients not found for technology '{tech_id}'",
                None,
            )
        # Operational cost of a TT is calculated in USD/2006
        operational_cost = (
            operational_cost + m1 * q_inf**m2
        )  # USD/year over the project lifetime

    # NOW OPERATIONAL COST IS CONVERTED TO ILS/YEAR
    # Using exchange rate of 4.494 as of 2006 to capture in all components of inflation
    operational_cost = operational_cost * exchange_rate

    # PRESENT WORTH OF O&M COST (ILS)
    operational_cost = operational_cost * annuity_factor
    return True, None, operational_cost

def calculateEnergyConsumption(
    train_tech_ids: List[str], q_inf: float, unit_cost_of_energy: float, annuity_factor: float
) -> Tuple[bool, str, float]:
    """
    FIRSTLY: The energy consumption of each technology is calculated using energy requirement coefficients,
             The total energy consumption of a train is obtained as the sum of the TTs' individual UP technology  req
             E_TT = Sum(E_i), [kWh/y]
    THEN: The energy cost calculated by multiplying the energy requirement of TT * the unit energy cost in the area (from V1)
            E_cost_TT = Sum(E_i)*unit energy cost in the area

    """
    energy_consumption = 0
    for tech_id in train_tech_ids:
        tech_data = getTechnologyData(tech_id)
        if tech_data is None:
            return (
                False,
                f"Train technology '{tech_id}' not found in technology data",
                None,
            )

        e1 = tech_data.energy_requirement_coefficient.get("e1")
        e2 = tech_data.energy_requirement_coefficient.get("e2")
        if e1 is None or e2 is None:
            return (
                False,
                f"Energy consumption coefficients not found for technology '{tech_id}'",
                None,
            )
        # ENERGY REQUIREMENT OT TT IN kWh/y
        energy_consumption = energy_consumption + e1 * q_inf**e2
        # ENERGY COST FOR TT (ILS/y) is obtained by multiplying by unit cost 0.595ILS/kWh in West Bank(Source: Maria)
    energy_cost = energy_consumption * unit_cost_of_energy

    # PRESENT WORTH OF ENERGY COST (ILS)
    energy_cost = energy_cost * annuity_factor
    return True, None, energy_cost

def lifeCycleCost(
    train_tech_ids: List[str],
    q_inf: float,
    cost_data: CostData,
) -> Tuple[bool, str, float, Dict[str, float]]:

    """
    This function calculates the life cycle cost (LCC)
    Which is the total cost of ownership of an asset or system over its entire life cycle,
    Meaning all the annual costs (ILS/y) are converted by using annuity factor over the design life
    For example:
              given Capex= 20 ILS, o&m = 10 ILS/y, Energy= 5 ILS/year, annuity_factor = 0.098
              LCC = Capex + ( Energy+ o&m)* annuity_factor
              LCC = 20 + (0.098 + 10)*0.098
              LCC = 173.06 USD
    Or :
             Using CRF
             LCC = CC + (Energy + O&M)/CRF
    However:
            given in this case the present worth of the o&m and energy were already calculated
             LCC = CC + LC + Pw_EC + Pw_O&M (ILS)
    """
    calculated, reason, capital_cost = calculateCapitalCost(train_tech_ids, q_inf, cost_data.exchange_rate)
    if not calculated:
        return False, reason, None, None

    calculated, reason, land_requirement_dict = calculateLandRequirement(
        train_tech_ids, q_inf, cost_data.unit_cost_of_land
    )
    if not calculated:
        return False, reason, None, None

    calculated, reason, operational_cost = calculateOperationalCost(
        train_tech_ids, q_inf,cost_data.exchange_rate, cost_data.annuity_factor
    )
    if not calculated:
        return False, reason, None, None

    calculated, reason, energy_cost = calculateEnergyConsumption(
        train_tech_ids, q_inf, cost_data.unit_cost_of_energy, cost_data.annuity_factor
    )
    if not calculated:
        return False, reason, None, None

    # LIFE CYCLE COST OF TT (ILS)
    life_cycle_cost = (
        capital_cost
        + land_requirement_dict["land_requirement_cost"]
        + operational_cost + energy_cost
    )
    # Returns all the constituent costs of a given TT (ILS)
    return (
        True,
        None,
        life_cycle_cost,
        {
            "capital_cost": capital_cost,
            "land_requirement_cost": land_requirement_dict["land_requirement_cost"],
            "operational_cost": operational_cost,
            "energy_cost": energy_cost,
            "land": land_requirement_dict,
        },
    )

def calculateAnnualizedTreatmentCost(
    train_tech_ids: List[str],
    q_inf: float,
    cost_data: CostData,
) -> Tuple[bool, str, float, Dict[str, float]]:

    """
    This function calculates the annualized treatment cost (Equivalent annual cost)
    Using:
            given, LCC is already calculated, we use
            ACC = LCC* CRF
            Where, LCC-lifecycle cost, CRF-Cost recovery factor
    Otherwise:
             The formula below also gives same answer, if annual cost of energy and O&M are used
             ACC = (CC+LC)*CRF + EC + o&m
             Where:
             CC- capital cost (ILS), LC-Land cost(ILS), EC-Energy cost (ILS/y), & O&M cost(ILS/y)
    THEN:    Calculates the unit cost of treatment per train, using volume treated per annum
             UC_TT= ACC/(Qinf*365)
             where:
             UC_TT - Unit cost of treatment (ILS/m3/year), Qinf- inflow (m3/d)
    """

    calculated, reason, lifecylcle_cost, costs_dictionary = lifeCycleCost(
        train_tech_ids, q_inf, cost_data
    )
    if not calculated:
        return False, reason, None, None

    # CALCULATE ANNUALIZED TREATMENT COST (ILS/YEAR)
    annualized_treatment_cost = lifecylcle_cost * cost_data.crf

    # CALCULATE UNIT COST OF TREATMENT (ILS/YEAR)
    Unit_cost_of_treatment = annualized_treatment_cost / (q_inf*365)

    costs_dictionary["lifecylcle_cost"] = lifecylcle_cost
    costs_dictionary["Unit_cost_of_treatment"] = Unit_cost_of_treatment

    return (True, None, Unit_cost_of_treatment, costs_dictionary)

# METHOD 2:MDCA- Multiple Decision Criteria Assessment
# For the scores (0 to 3) under technical, these values are considered positive
# Example Reliability, sum the relaibility of all the unit processes in a given Treatment train , total =10
# Normailze it by averaging, meaning we dive by number of units in that treatment train, eg if n=5, 10/5
# We also have to average the scores itself,Normilized score for realibility= 10/(n*3)
# For every score, we shall return normalized scores

# Except ror the scores (0 to 3) under ENVIRONMENTAL and COST, these values are considered negative
# We normalize as say ground water pollutaion = 1-1/3(summation of scores/number of units)

# NEXT, say we now have all the normalized values for the scores, we the have to conduct an overal evaluation
# We need weights, these should be assigned by the user, for each criteria (eg Reliability ....),
#             "Weight_Rel": 2
#             "Ease to upgrade": 3
#             "Adaptability to varying flow": 5
#             "Adaptability to varying quality": 1
#             "Ease of O & M": 6
#             "Ease of construction": 7
#             "Ease of demonstration": 1
#             "Power demand": 8
#             "Chemical demand": 7
#             "Odor generation": 1
#             "Impact on ground water": 3
#             "Land requirement": 1
#             "Cost of treatment": 10
#             "sludge production": 4

# Overall TT evaluation = (Sum(Weight*NEC))/sum of Weights
# Return the best three TTs, means the ones with highest overall score
# Calculate and Return the costs of the best 5.
# Select one with least cost out of the three.

def calculateTechnicalScores(
    train_tech_ids: List[str],
) -> Tuple[bool, str, Dict[str, float]]:
    """
    calculate tech scores for a given technology train

     =
    """
    sums = {}
    scores = {}
    for tech_id in train_tech_ids:
        tech_data = getTechnologyData(tech_id)
        if tech_data is None:
            return (
                False,
                f"Train technology '{tech_id}' not found in technology data",
                None,
            )

        for key, value in tech_data.technical_evaluation_scores.items():
            current_value = sums.get(key, 0)
            sums[key] = current_value + value

    for evaluation_score, score_sum in sums.items():
        scores[evaluation_score] = (1 / 3) * (score_sum / len(train_tech_ids))

    return True, None, scores

def calculateEnvironmentalScores(
    train_tech_ids: List[str],
) -> Tuple[bool, str, Dict[str, float]]:
    """
    calculate env scores for a given technology train

     =
    """
    sums = {}
    scores = {}
    for tech_id in train_tech_ids:
        tech_data = getTechnologyData(tech_id)
        if tech_data is None:
            return (
                False,
                f"Train technology '{tech_id}' not found in technology data",
                None,
            )

        for key, value in tech_data.environmental_evaluation_scores.items():
            current_value = sums.get(key, 0)
            sums[key] = current_value + value

    for evaluation_score, score_sum in sums.items():
        scores[evaluation_score] = (1 - ((1 / 3) * (score_sum / len(train_tech_ids))))

    return True, None, scores


def calculateEconomicScores(
    train_tech_ids: List[str],
) -> Tuple[bool, str, Dict[str, float]]:
    """
    calculate env scores for a given technology train

     =
    """
    sums = {}
    scores = {}
    for tech_id in train_tech_ids:
        tech_data = getTechnologyData(tech_id)
        if tech_data is None:
            return (
                False,
                f"Train technology '{tech_id}' not found in technology data",
                None,
            )

        for key, value in tech_data.economic_evaluation_scores.items():
            current_value = sums.get(key, 0)
            sums[key] = current_value + value

    for evaluation_score, score_sum in sums.items():
        scores[evaluation_score] = (1 - ((1 / 3) * (score_sum / len(train_tech_ids))))

    return True, None, scores

#The weights are subjective and assigned by user depending on the significance of the criteria to them
# The summation of weights should be 100% implying 1

def getScoreWeightsFromUser() -> Dict[str, float]:
    return {
        "Reliability": 14,
        "Ease to upgrade": 3,
        "Adaptability to varying flow": 5,
        "Adaptability to varying quality": 8,
        "Ease of O & M": 4,
        "Ease of construction": 3.5,
        "Ease of demonstration": 1.5,
        "Power demand": 10,
        "Chemical demand": 2,
        "Odor generation": 4,
        "Impact on ground water": 6,
        "Land requirement": 14,
        "Cost of treatment": 20,
        "sludge production": 5,
    }

def calculateScores(
    train_tech_ids: List[str],
) -> Tuple[bool, str, float, Dict[str, float]]:
    """
    calculate scores for a given technology train,
    Fouteen evaluation criteria are used in the tool, to cater for economic, environmental and technical aspects
    of each unit process

    """
    calculated, reason, tech_scores = calculateTechnicalScores(train_tech_ids)
    if not calculated:
        return False, reason, None, None

    calculated, reason, env_scores = calculateEnvironmentalScores(train_tech_ids)
    if not calculated:
        return False, reason, None, None

    calculated, reason, eco_scores = calculateEconomicScores(train_tech_ids)
    if not calculated:
        return False, reason, None, None

    score = None
    score_dictionary = {}
    # merge dictionaries
    score_dictionary = {**tech_scores, **env_scores, **eco_scores}

    weights = getScoreWeightsFromUser()

    overall = 0
    for score_name, score_value in score_dictionary.items():
        score_weight = weights[score_name]
        overall += score_weight * score_value

    weight_sum = 0
    for weight_score in weights.values():
        weight_sum += weight_score

    return (True, None, (overall / weight_sum), score_dictionary)


def sortTrains(
    trains: List[List[str]],
    effluent_concentrations: List[Dict[str, float]],
    q_inf: float,
    cost_data: CostData,
    sort_key="cost",
) -> List[List[Dict[str, Union[List[str], float]]]]:
    """
    sort trains by cost
    For each area (location coordinates), return the train and the cost for each reuse in the area"
    and then the cheapest is selected.

    example:
    sortTrains([['GrCh', 'SP_fac', 'ClO2'], 250, 3000], ['GrCh', 'SP_fac', 'ClO2'], 250, 3000]) will return: blah, blah
    """
    trains_with_cost = []
    for train, eff_conc in zip(trains, effluent_concentrations):
        calculated, reason, cost, costs_dictionary = calculateAnnualizedTreatmentCost(
            train, q_inf, cost_data
        )
        if not calculated:
            print(reason)
            cost = None
            costs_dictionary = None

        calculated, reason, score, score_dictionary = calculateScores(train)
        if not calculated:
            print(reason)
            score = None
            score_dictionary = None

        display_train = " -> ".join(train)
        costs_dictionary["display_train"] = display_train
        trains_with_cost.append(
            {
                "display_train": display_train,
                "cost": cost, # ILS/m3/year
                "constituent_costs": costs_dictionary,
                "train": train,
                "score": score,
                "score_dictionary": score_dictionary,
                "effluent_concentrations": eff_conc,
            }
        )

    print(f"Sorting trains based on {sort_key}")
    if sort_key == "cost":
        print("Weights being used are ")
        json_formatted_str = json.dumps(getScoreWeightsFromUser(), indent=4)
        print(json_formatted_str)
        # Remove the Reverse=True if you are sorting with cost
        #TODO : Comment out if using SORT, and uncomment 2nd line, VISEVERSA if using COSTS
    #sorted_trains = sorted(trains_with_cost, key=lambda k: k[sort_key])
    sorted_trains = sorted(trains_with_cost, key=lambda k: k[sort_key], reverse=True)

    if len(sorted_trains) > 3:
        # print(f"first 3 trains: {sorted_trains[:3]}")
        # print(f"last 3 trains: {sorted_trains[-3:]}")
        pass
    else:
        pass

    x = 1
    for train in sorted_trains:
        print("Train number ", x)
        json_formatted_str = json.dumps(train, indent=4)
        print(json_formatted_str)
        x += 1
        if x == 6:
            break

    return sorted_trains


def generateTrainsWith(q_inf, reuse_standard, unit_land_cost: float):
    built, reason, trains = buildTrains(
        "preliminary_level",
        ["Start"],
        q_inf,
    )
    if not built:
        print(reason)
    else:
        print(f"Trains built. Found {len(trains)} total trains")

    cleanedTrains = dataCleanTrains(trains[:])

    re_use_standard_data = RE_USE_STANDARDS.get(reuse_standard)
    if re_use_standard_data is None:
        print(f"Reuse standard '{reuse_standard}' data not found")
        return

    feasible_trains, effluent_concentrations = screen_trains_by_reuse_standard(
        cleanedTrains, re_use_standard_data
    )

    print(
        f"Trains filtered down to {len(feasible_trains)} total trains fit for {reuse_standard}"
    )

    """"
    If reuse standard is not being considered, then use then use the line below
    """
    # feasible_trains = cleanedTrains

    # TODO: Swtich point. Change sort_key to either "cost" or "score"
    """
    CHANGE METHOD, depending on whether you wish to evaluate using MCDA or LCA
    for MCDA: 
            sort_key = score
    for LCA: 
            sort_key = cost              
    """
    sortedTrains = sortTrains(
        feasible_trains,
        effluent_concentrations,
        q_inf,
        getCostData(unit_land_cost),
        sort_key="score",
    )

    print(
        f"Trains sorted down to {len(sortedTrains)} "
    )

    # print(sortedTrains)
    global GENERATED_TRAINS

    if len(sortedTrains) > 0:
        # print(f"Best train: below")
        # json pretty print
        # json_formatted_str = json.dumps(sortedTrains[0], indent=4)
        # print(json_formatted_str)
        GENERATED_TRAINS[f"{q_inf}_{reuse_standard}"] = sortedTrains


def getBestTrain(q_inf, reuse_standard) -> Tuple[bool, str, Dict[str, float]]:
    global VALID_RE_USE_STANDARDS
    if reuse_standard not in VALID_RE_USE_STANDARDS:
        print(f"Reuse standard '{reuse_standard}' not found")
        return False, None, None

    train_data = GENERATED_TRAINS.get(f"{q_inf}_{reuse_standard}")
    if train_data is None:
        print(
            f"Train data for q_inf {q_inf} and reuse standard '{reuse_standard}' not found"
        )
        return False, None, None

    if len(train_data) < 1:
        print(
            f"No trains found for q_inf {q_inf} and reuse standard '{reuse_standard}'"
        )
        return False, None, None

    best_train = train_data[0]

    train = best_train.get("train")
    if train is None:
        print(
            f"Train not found for q_inf {q_inf} and reuse standard '{reuse_standard}'"
        )
        return False, None, None

    # constituent_costs = best_train.get("constituent_costs")
    # if constituent_costs is None:
    #     print(f"Cost not found for q_inf {q_inf} and reuse standard '{reuse_standard}'")
    #     return False, None, None

    display_train = " -> ".join(train)

    print(
        f"\n\n\nBest train for q_inf {q_inf} and reuse standard '{reuse_standard}': {display_train}\n\n\n"
    )

    return True, display_train, best_train


def getQInf() -> float:
    """
    get q_inf

    For every location coordinate, get the Qinf, (flow_wtp), returning Qinf for each of the coordinates
    """
    return 150.0

def getReuseStandard() -> str:
    """
    For every coordinate location, we have to get the different reuse application and use the reuse application to get the respective reuse standard from the knowledge base
    """
    return "Urban reuse standard"

def getUnitLandCost() -> float:
    """
    For every coordinate location, we have to get the unit land cost
    """
    return 1500.0


def runIntegration(q_infs, unit_land_costs: List[float]):
    try:
        print("generating trains using ", q_infs)
        loadTechnologyData()

        for q_inf, unit_land_cost in zip(q_infs, unit_land_costs):
            for reuse_standard in RE_USE_STANDARDS:
                generateTrainsWith(q_inf, reuse_standard, unit_land_cost)
    except Exception as e:
        print(f"Integration failed: {e}")
        return


def runTreatmentModuleIntegration(flow_wwtp, types_treatment_flow_based):
    """
    flow_wwtp: float containing the waste water flow of one wwtp
    types_treatment_flow_based: list containing , for example: ['urban_reuse', 'agriculture_use']
    This maps the technology selection (types_treatment_flow_based) in Version 1, into a reuse standard in V2
    These standards are used to evaluate TTs for a given reuse
    """
    dict_treatment_costs = []
    try:
        for i in range(len(types_treatment_flow_based)):
            best_train_data = None

            if types_treatment_flow_based[i] == "no_wwtp":
                dict_treatment_costs.append(0)
                standard = None
            elif types_treatment_flow_based[i] == "mbr with no reuse":
                standard = "Environmental discharge standard"
            elif types_treatment_flow_based[i] == "cas with agr. reuse":
                standard = "Agricultural reuse standard"
            elif types_treatment_flow_based[i] in [
                "mbr with urb. reuse",
            ]:
                standard = "Urban reuse standard"

            if standard is not None:
                _, _, best_train_data = getBestTrain(flow_wwtp[i], standard)
                if best_train_data:
                    json_formatted_str = json.dumps(best_train_data, indent=4)
                    print(json_formatted_str)

            dict_treatment_costs.append(best_train_data)

        global TREATMENT_COST
        TREATMENT_COST = dict_treatment_costs
        return dict_treatment_costs
    except Exception as e:
        print(f"Integration failed: {e}")
        return


# TODO: get "constituent_costs" from the dictionary result of runTreatmentModuleIntegration()
def toV1CostArray(
    treatment_costs_V2: Dict[str, float], which_costs: List[str]
) -> List[float]:
    """
    treatment_costs_V2: Dict[str, float] containing the costs for each train
    which_costs: str containing the cost type, possible values: 'capital_cost',
    'land_requirement_cost', 'operational_cost', 'energy_cost', 'lifecylcle_cost'
    or 'annualized_treatment_cost'
    """
    arrayed_treatment_cost = []
    try:
        for treatment_cost in treatment_costs_V2:
            summed_cost = 0
            for cost_type in which_costs:
                cost = treatment_cost["constituent_costs"].get(cost_type)
                if cost is None:
                    print(f"WARN: Cost not found for {cost_type}")
                    summed_cost += 0

                summed_cost += cost

            arrayed_treatment_cost.append(summed_cost)
        return arrayed_treatment_cost
    except Exception as e:
        print(f"Integration failed: {e}")
        return


def normalizeV2ZeroFlows(
    reference_summation: List[float],
    treatment_costs: Union[None, List[float]],
    land_costs: Union[None, List[float]],
) -> Tuple[List[float], List[float]]:
    zeros = [0] * len(reference_summation)

    if treatment_costs is None or len(treatment_costs) == 0:
        treatment_costs = zeros

    if land_costs is None or len(land_costs) == 0:
        land_costs = zeros

    return treatment_costs, land_costs

if __name__ == "__main__":
    loadTechnologyData()
    generateTrainsWith(getQInf(), getReuseStandard(), getUnitLandCost())


