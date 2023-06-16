import math

import numpy as np

import src.user_inputs as usin
from src.faster import config
from src.faster.custom_typing import Numpy1dFloat


def calc_kinematic_viscosity(temperature: float) -> float:
    """"
    This function calculates the value for kinematic viscosity of water in relation to the temperature.

    input: temperature of the water is a user input.
    """

    # This equation was produced in Excel from a table found online containing different values of\
    # temperature and their respective kinematic viscosity values.
    weights = [1.776260e-06, -5.508630e-08, 9.888130e-10, -9.152510e-12, 3.295300e-14]
    kinematic_viscosity = sum(w * temperature**i for i, w in enumerate(weights))

    return kinematic_viscosity


def calc_diameter(flow: Numpy1dFloat) -> Numpy1dFloat:
    """
    input: flow in m3/s
    return: diam = optimal diameter of the pipe in mm.
    """
    m2mm = 1e3
    # Equation for "diámetro óptimo" from Manual de Agua Potable, Alcantarillado y \
    # Saneamiento, Comisión Nacional del Agua, https://www.gob.mx/conagua/documentos/biblioteca-digital-de-mapas
    pipe_diameter = (1.2 * m2mm) * np.sqrt(flow)
    return pipe_diameter


def calc_flow_velocity(flow: Numpy1dFloat, pipe_diameter: Numpy1dFloat) -> Numpy1dFloat:
    mm2m = 1e-3
    # cross section area of the pipe, from continuity equation.
    area = (math.pi * 0.25) * (pipe_diameter * mm2m) ** 2
    velocity = flow / area  # flow velocity
    return velocity


def calc_friction_factor(pipe_diameter: Numpy1dFloat, v: Numpy1dFloat) -> Numpy1dFloat:
    """
    Calculates the friction factor according to Churchill equation (1974).

    input:  v = flow velocity in m/s
            pipe_diameter = diameter of the pipe in mm
    returns: friction factor dimensionless.
    """
    mm2m = 1e-3
    k = 0.0041  # Roughness coefficient for PEAD (HDPE)
    kin_viscos = 1.003e-6  # kinematic viscosity of the water at 20 degrees.
    diam_m = pipe_diameter * mm2m  # diameter in meters.
    re = v * diam_m / kin_viscos  # Reynolds number

    a1 = (7 / re) ** 0.9
    a2 = 0.27 * k / pipe_diameter
    a3 = (a1 + a2) ** -1
    a4 = 2.457 * np.log(a3)
    a = a4 ** 16
    b = (37530 / re) ** 16
    ff1 = (8 / re) ** 12
    ff2 = (a + b) ** (-1.5)
    friction_factor = 8 * (ff1 + ff2) ** (1 / 12)
    return friction_factor


def calc_total_head_height(flow: Numpy1dFloat, pump_pipe_length: Numpy1dFloat,
                           pump_height: Numpy1dFloat, pipe_diameter: Numpy1dFloat) -> Numpy1dFloat:
    """
    This function calculates the total head to transfer the wastewater from start to end.
    The total head is the summation of the geometric difference between the elevations
    of the two points and the friction losses in the way that the wastewater takes.
    The friction losses in meters is calculated according to Darcy's equation, which is an
    explicit solution of the Colebrook-White equation. It is valid for all the regimes
    in the Moody's diagram.

    Input:  d = internal pipe diameter [mm]
            l = length of the pipeline [m]
            v = wastewater velocity [m/s]
            e = absolute roughness of the pipe material [mm]
            re = Reynolds' number [-]
            ff = friction factor from Colebrook-White equation
            g = acceleration due the gravity [m2/s]
    """
    # friction factor to calculate the head loss.
    # pump_pipeline_length, gravity_pipeline_length = calc_pipeline_length(start, end)
    v = calc_flow_velocity(flow, pipe_diameter)
    ff = calc_friction_factor(pipe_diameter, v)
    # losses due friction (Darcy-Weisbach equation), 19.62 = 2*g (9,806650) plus the geometric elevation difference.
    total_head_height = ff * pump_pipe_length * v ** 2 / (19.6133 * pipe_diameter) + pump_height
    return total_head_height


def calc_pump_power(flow: Numpy1dFloat, pump_pipe_length: Numpy1dFloat,
                    pump_height: Numpy1dFloat, pipe_diameter: Numpy1dFloat) -> Numpy1dFloat:
    """The pump power is the amount of energy per unit of time consumed by the pump.
    step 1: to find the required flow.
    step 2: to calculate the total head height. For this, I will need:
                1. find the elevation difference between start and end points;
                2. find the total friction losses in the ww path.
    The pump efficiency is between 50 and 85%.

    Function description:
    This function calculates the power (potência) required (BHP - Bake horsepower) by a centrifuge pump.
    input: ww_flow - wastewater flow [m3/s]
                total_head_height - the elev difference between start and end points and the losses during the
                transportation [m]
                pump_efficiency - the efficiency of the pump in decimal [-]
                specific_weight - specific weight of the wastewater [N/m3]
    output: pump_power - required power for the pump [kW]
     """
    # The pump efficiency is assumed to be 60% based on centrifuge pump with 50mm grid
    pump_efficiency = 0.6
    # The specific weight of the wastewater is assumed to be the same as the water at 20 degrees celsius [kg/m3]
    specific_weight = 9789.0
    # Expected project life [years]
    total_head_height = calc_total_head_height(
        flow, pump_pipe_length, pump_height, pipe_diameter)
    # Required energy [W]
    pump_power_watts = total_head_height * flow * specific_weight / pump_efficiency
    # Required energy [kW]
    pump_power = pump_power_watts * 1e-3
    assert np.all(pump_power[~np.isnan(pump_power)] >= 0), "Pumping power cannot be negative."
    return pump_power


def calc_pumping_energy_cost_year(flow: Numpy1dFloat, pump_pipe_length: Numpy1dFloat,
                                  pump_height: Numpy1dFloat, pipe_diameter: Numpy1dFloat) -> Numpy1dFloat:
    """This functions calculates the yearly energy costs related to pumping sewage.

    inputs: pumping_time_period: for how long will the pump work?
                pump_capacity: it will depend on the flow and the highest elevation.

    returns: pumping_energy_costs
    """
    # TODO: base cost on literature analysis
    # kwh_cost = 0.618  # value from de Alvarenga, 2018 (pg. 54) in ILS. TODO: input real energy cost
    # It is assumed that the pump station is going to work all year round.
    pumping_hours = 24 * 365.25

    pump_power_kw = calc_pump_power(flow, pump_pipe_length, pump_height, pipe_diameter)

    pumping_energy_cost = pump_power_kw * pumping_hours * usin.price_average_kwh
    """#    flow_sewage= # flow collected at candidate 1 (location1) + the flow collected at location2.
    How can I import this data? Procurar no da MAria. No da Maria, vi que ela tem a função AOKP_coordinates, where she
    names the variable num_WWTP which represents the number of features in the path_pour_points.shp.
    # I believe that this shapefile is produced when the optimization algorithm chooses new locations for the WWTPs and
    s
    end these locations back to Operwas code (which I still don't know which file is this one). In my case, since I'm
    using the results of Maria's optimization, my WWTP will also chance, but I don't think I should be doing all this
    part of her calculation again. In my scenario the candidates will still receive the water from their sub-catchment,
    but the difference now is that the contribution of a certain candidate suggested by the algorithm as a PS, will be
    summed to the next candidate's receiving contribution. So, no need for doing the step with the shapefiles in
    operwa_library before the line 1706 (don't know where exactly).
    #   pump_capacity= # this is going to be an equation dependent on the flow_sewage and the highest elevation in the
    pipeline path.
    """
    return pumping_energy_cost


def calc_wwps_opex_and_maint_year(flow_pump: Numpy1dFloat, pipe_length_pump: Numpy1dFloat, pump_height: Numpy1dFloat) -> Numpy1dFloat:
    pipe_diameter = calc_diameter(flow_pump)
    energy_cost_year = calc_pumping_energy_cost_year(
        flow_pump, pipe_length_pump, pump_height, pipe_diameter)
    wwps_construction_cost = calc_wwps_construction_cost(flow_pump)
    maintenance_cost_year = calc_maintenance_cost_year(energy_cost_year, wwps_construction_cost)
    operational_cost_year = energy_cost_year
    return operational_cost_year + maintenance_cost_year


def calc_wwps_capex(flow_pump: Numpy1dFloat, pump_length: Numpy1dFloat, gravity_length: Numpy1dFloat) -> Numpy1dFloat:
    pipe_diameter = calc_diameter(flow_pump)
    wwps_construction_cost = calc_wwps_construction_cost(flow_pump)
    pipeline_construction_cost = calc_pipeline_cost_total(
        pump_length, gravity_length, pipe_diameter)
    wwps_investment_cost = wwps_construction_cost + pipeline_construction_cost
    return wwps_investment_cost


def calc_wwps_construction_cost(flow_pump: Numpy1dFloat) -> Numpy1dFloat:
    # TODO: base cost on literature analysis
    """This function calculates the cost related to the construction of the pumping station according to "Pumping
    Stations Design - Chapter 29: Costs".
    From the graph shown in Figure 29-2. Construction costs of custom wet well–dry well wastewater pumping stations, it
    was produced its equation in Excel.
       # Option 2:
    #construction_cost_pumping_pipeline = 382.5 * length * diameter ** 4.455  # Equation used by Ostfeld (2011) to
    calculate pumping pipeline construction costs. (Eq. 1 in the article).
   # l_w = diameter_gravity + 0.6  # This is the excavation width: diameter plus 30 cm to each side of the pipe.
    #construction_cost_gravity_pipeline_shallow = 21.6 * length_gravity * diameter_gravity ** 2.26 + 7 * l_w * ((h1**2 +
     c_min **2)/2 * (j-j_s))  # c_min is the minimum pipeline depth, h1 is the least
    # excavation cost cost to a depth of h1, j is the required slope.
    #construction_cost_gravity_pipeline_deep = construction_cost_gravity_pipeline_shallow + 10 * l_w * (length_gravity *
     c_min + ((length_gravity ** 2 / 2) * (j - j_s)) - ((h1 ** 2 - c_min ** 2) / (2*(j - j_s))))
    #construction_cost_pump = 64920 * pump_power ** 0.33
    """
    # Option 1:
    # construction_cost = 7172.7 * flow_pump**0.8891  # Equation estimated from the graph mentioned\
    # in the description of this function.
    # Option 3:
    # euro2ils = 3.75  # Currency conversion on january 31, 2023.
    # construction_cost = 1000 * euro2ils * math.e ** (4.3189 + 0.5329 * (math.log(pump_power)))
    # TODO: add reference to Cabral et al., 2018, Statistical modelling of wastewater pumping stations costs.
    # TODO: find how to bring to present value.
    # TODO: what is rand2ils?
    rand2ils = 0.2
    m3_2_liter = 1e3
    # todo: check the units of pump power in the article
    construction_cost = rand2ils * 91169 * (m3_2_liter * flow_pump) ** 0.5444
    assert np.all(construction_cost[~np.isnan(construction_cost)] >= 0.0)
    return construction_cost


def calc_pipeline_cost_pump(pump_length: Numpy1dFloat, pipe_diameter: Numpy1dFloat) -> Numpy1dFloat:
    # TODO: base cost on literature analysis
    # pump_length, gravity_length = calc_pipeline_length(start, end)
    # pipeline_cost = calc_diameter(flow) * ((2 * pump_length) + (1.5 * gravity_length))  # 2 and 1.5 are just random\
    # numbers to substitute the coefficients for unitary price for the pipes that varies with material.

    # excavation_depth = pipe_diameter + 0.5  # TODO: check an appropriate depth
    #
    # pump_pipeline_cost = pump_length * (53.4688 + 11.561 * pipe_diameter + 27.835 * pipe_diameter ** 2)
    # gravity_pipeline_cost = gravity_length * (0.1254 * pipe_diameter + 131.4391 * excavation_depth -
    #                                           0.044 * pipe_diameter * excavation_depth - 203.311)
    rand2ils = 0.2
    pump_pipeline_cost = rand2ils * pump_length * \
        (0.0032 * pipe_diameter ** 2 + 4.0755 * pipe_diameter - 52)
    # assert np.all(pump_pipeline_cost[~np.isnan(pump_pipeline_cost)] >= 0.0)
    return pump_pipeline_cost


def calc_pipeline_cost_grav(gravity_length: Numpy1dFloat, pipe_diameter: Numpy1dFloat) -> Numpy1dFloat:
    # TODO: base cost on literature analysis
    # pump_length, gravity_length = calc_pipeline_length(start, end)
    # pipeline_cost = calc_diameter(flow) * ((2 * pump_length) + (1.5 * gravity_length))  # 2 and 1.5 are just random\
    # numbers to substitute the coefficients for unitary price for the pipes that varies with material.
    # excavation_depth = pipe_diameter + 0.5  # TODO: check an appropriate depth
    #
    # pump_pipeline_cost = pump_length * (53.4688 + 11.561 * pipe_diameter + 27.835 * pipe_diameter ** 2)
    # gravity_pipeline_cost = gravity_length * (0.1254 * pipe_diameter + 131.4391 * excavation_depth -
    #                                           0.044 * pipe_diameter * excavation_depth - 203.311)
    rand2ils = 0.2
    gravity_pipeline_cost = rand2ils * gravity_length * \
        (0.0024 * pipe_diameter ** 2 + 2.8788 * pipe_diameter + 300)
    assert np.all(gravity_pipeline_cost[~np.isnan(gravity_pipeline_cost)] >= 0.0)
    return gravity_pipeline_cost


def calc_pipeline_cost_total(pump_length: Numpy1dFloat, gravity_length: Numpy1dFloat, pipe_diameter: Numpy1dFloat) -> Numpy1dFloat:
    pipeline_cost = calc_pipeline_cost_pump(
        pump_length, pipe_diameter) + calc_pipeline_cost_grav(gravity_length, pipe_diameter)
    return pipeline_cost


def calc_maintenance_cost_year(energy_cost_year: Numpy1dFloat, wwps_construction_cost: Numpy1dFloat) -> Numpy1dFloat:
    maintenance_cost_year = (energy_cost_year + wwps_construction_cost) * (1/9)
    return maintenance_cost_year
