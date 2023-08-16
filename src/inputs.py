import os

#################
## DIRECTORIES ##
#################

# homedir = os.environ['HOME']

# Gets the current working directory as root for the others
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

# Name of file that will save the results of the tool. It has to be in the format: 'name_of_file.csv'
file_with_possible_locations = os.path.join(root, "inputs", "wwtp_locations.csv")

# Number of wastewater treatment plants you want to set
# nr_wwtp = 6

# population_catchment = 44177

# Number of combinations you want to analyse (number of runs)
# nr_runs = 100

path_results = os.path.join(root, "results")

# Name of file that will save the results of the optimization. It has to be in the format: 'name_of_file.txt.txt'
file_with_results = os.path.join(path_results, "list_of_results.csv")

# Name of file that will save ALL results of the runs of AOKP algorithm.
file_all_generations = os.path.join(path_results, "all_generations.csv")
file_with_last_population = os.path.join(path_results, "organized_results.csv")


# Directory of results file
path_partial_results = os.path.join(path_results, "partial_results.csv")
path_total_results = os.path.join(path_results, "total_results.csv")


# Input files:
path_inputs = os.path.join(root, "inputs")
path_user_inputs = os.path.join(path_inputs, "operwas_user_input.xls")
path_filled = os.path.join(path_inputs, "DEM_filled.tif")
path_buildings = os.path.join(path_inputs, "union.shp")
path_channel = os.path.join(path_inputs, "channels.shp")
path_population = os.path.join(path_inputs, "grid_data.shp")

# Temporary files:
path_temp = os.path.join(root, "temp")
path_pour_points = os.path.join(path_temp, "pourpoints")
path_pour_points_shp = os.path.join(path_temp, "pourpoints", 'pour_points.shp')
path_watershed = os.path.join(path_temp, "watershed.gpkg")
path_watershed_folder = os.path.join(path_temp, "watershed")
path_watershed_shp = os.path.join(path_temp, "watershed", "watersheds.shp")
path_watershed_dem = os.path.join(path_temp, "watershed_dem.tif")
path_temp_outlet = os.path.join(path_temp, "outlet.shx")
path_outputBufferfn = os.path.join(path_temp, "temporary_buffer_area.shp")
path_stream = os.path.join(path_temp, "stream.tif")
path_stream_temp = os.path.join(path_temp, "stream_trim.tif")

# Output files:
path_outputs = os.path.join(root, "outputs")
path_subcatchments = os.path.join(path_outputs, "polygonized.shp")
path_ordered_outlets = os.path.join(path_outputs, "outlet_ordered.shp")
path_buffer_areas = os.path.join(path_outputs, "buffer_area.shp")

###################
# Input Variables #
###################

# # Catchments characteristics
#
# num_locations = 4
# Population = [30000, 30000, 8500, 15000]
# total_population = 84915.36            # sum of population from the grid file
# water_consumption = [0.043, 0.060, 0.078, 0.054]      # [m3/day/capita]
# losses_infiltration = 0.10  # [%]
# coef_peak = 2.4  # [-]
# inhabit_per_household = 6.2
# losses_distribution = [0.2646, 0.4723, 0.346, 0.195]    #Losses in distribution system (%)
# losses_transmission = [0.17, 0.17, 0.17, 0.17]              #Losses in transmission system (%)
# collection_efficiency = [0.4195, 0.4748, 0.3008, 0.626]  #Collection Efficiency (%) - Residents to SP
# urb_irrig_use_coef = 0.15
# area_usage_cas = 1.028    # m2/m3.day
# area_usage_mbr = 0.514    # m2/m3.day
#
# # Reuse areas assumptions
# # vol_area_ratio = 0.001931507        # m3/d.m2 (deleted, it is not being used for anything)
# irr_oper_agr = 0.092571             # m3/d.m2
# irr_oper_urb = 0.01                 # m3/d.m2
# # minimum_flow_wwtp = 10           # m3/d  Minimum flow that will be treated
# nr_reuse_options = 3
# no_data_value_land_price = 200
# no_data_value_loc_index = 0
#
# # Environmental benefits assumptions
# ww_treatment_after_discharge = 3.36    #ILS/m3
# # Technology of the WWTP
#
# # TECHNOLOGY 1 - Conventional Activated Sludge
#
# known_flow_cas_1 = 500  # Design Capacity of WWTP1 [m3/day]
# known_flow_cas_2 = 5750  # [m3/day]# known_flow_cas_2= Design Capacity of WWTP2 [m3/day]
# known_inv_cost_cas_1 = 5.52 * 10 ** 6  # [ILS] # known_inv_cost_cas_1= Investment cost for WWTP1 [ILS]
# known_inv_cost_cas_2 = 3.472 * 10 ** 7  # [ILS] # known_inv_cost_cas_2= Investment cost for WWTP2 [ILS]
# known_oem_cost_cas_1 = 6.1171 * 10 ** 4  # [ILS/year] # known_oem_cost_cas_1= O&M cost for WWTP1 [ILS/year]
# known_oem_cost_cas_2 = 1.88 * 10 ** 6  # [ILS/year] # known_oem_cost_cas_2= O&M cost for WWTP2 [ILS/year]
#
# # TECHNOLOGY 2 - Membrane Bioreactor
#
# known_flow_mbr_1 = 500  # [m3/day] # known_flow_mbr_1= Design Capacity of WWTP1 [m3/day]
# known_flow_mbr_2 = 2000  # [m3/day] # known_flow_mbr_2= Design Capacity of WWTP2 [m3/day]
# known_inv_cost_mbr_1 = 4.04 * 10 ** 6  # [ILS] # known_inv_cost_mbr_1= Investment cost for WWTP1 [ILS]
# known_inv_cost_mbr_2 = 1.47 * 10 ** 7  # [ILS] # known_inv_cost_mbr_2= Investment cost for WWTP2 [ILS]
# known_oem_cost_mbr_1 = 4.36399 * 10 ** 5  # [ILS/year] # known_oem_cost_mbr_1= O&M cost for WWTP1 [ILS/year]
# known_oem_cost_mbr_2 = 9.55209 * 10 ** 5  # [ILS/year] # known_oem_cost_mbr_2= O&M cost for WWTP2 [ILS/year]
#
# # Sanitation fees
#
# fee_connection = 15  # [ILS/m2 of house] # fee_connection = Connection fee to the WWTP
# fee_sanitary = [1.5, 1.5, 1.5, 1.5]  # [ILS/m3 of drinkable water consumed] # fee_sanitary = Fee for collection and treatment of the wastewater
#
# # Water tariff
#
# tariff = [7.08, 6.42, 4.42, 5]                       #Average water price (ILS/m3)
#
# # Reclaimed wastewater system
#
# tariff_reclaimed = [3.54, 3.21, 2.21, 2.5]           #Average water price of reclaimed water (ILS/m3)
#
# # Pipe costs for wastewater collection and reuse water distribution network
#
# Pw = [227.1, 259.708, 291.12, 326.63, 361.14, 459.67, 529.18, 598.69]  #  Pw=Cost pipe for the transport of wastewater [ILS/m]
# # Pu = [229.1, 262.708, 295.12, 330.63, 366.14, 466.67, 539.18, 611.69]  # Pu=Cost pipe for the transport of urban water treated [ILS/m]
# # Pa = [423.14] # Pa=Cost pipe for the transport of agriculture water treated [ILS/m]
# hRan = [50, 150, 400, 700, 1000, 1500, 2200]  # hRan= highest range value of the length of the pipes per specific diameter [m]
#
# # Reservoir costs
#
# known_reservoir_volume_1 = 500  # [m3] # known_reservoir_volume_1= Design Capacity of Reservoir 1 [m3]
# known_reservoir_volume_2 = 2000  # [m3] # known_reservoir_volume_2= Design Capacity of Reservoir 2 [m3]
# known_inv_cost_reservoir_1 = 780400  # [ILS] # known_inv_cost_reservoir_1= Investment cost for Reservoir 1 [ILS]
# known_inv_cost_reservoir_2 = 1951000  # [ILS] # known_inv_cost_reservoir_2= Investment cost for Reservoir 2 [ILS]
#
# # Fresh water energy and management costs
#
# portion_s1 = [29, 25, 100, 15]          #Portion of water coming from source 1 (%)
# portion_s2 = [71, 75, 0, 0]             #Portion of water coming from source 2 (%)
# portion_s3 = [ 0, 0, 0, 85]             #Portion of water coming from source 3 (%)
#
# c_prod_s1 = [0.84, 0.84, 0.84, 0.84]    #Costs related to production of water at source 1 (ILS/m3)
# c_prod_s2 = [4.38, 4.38, 4.38, 4.38]    #Costs related to production of water at source 2 (ILS/m3)
# c_prod_s3 = [3.24, 3.24, 3.24, 3.24]    #Costs related to production of water at source 3 (ILS/m3)
#
# c_tran_s1 = [1.04, 1.04, 0.61, 1.28]    #Costs related to transmission of water from source 1 to distribution (ILS/m3)
# c_tran_s2 = [0, 0, 0, 0]                #Costs related to transmission of water from source 2 to distribution (ILS/m3)
# c_tran_s3 = [0, 0, 0, 0.49]             #Costs related to transmission of water from source 3 to distribution (ILS/m3)
#
# c_tran_adm = [0.42, 0.42, 0.42, 0.42]    #Administrative costs regarding transmission from all sources (ILS/m3)
#
# c_dist_staff = [1.97, 0.66, 0.70, 0.73]  #Distribution costs regarding staff management (ILS/m3)
# c_dist_energy = [0.01, 0.02, 0.00, 0.00] #Distribution costs regarding energy (ILS/m3)
# c_dist_adm = [1.41, 0.42, 0.15, 0.91]    #Distribution costs regarding administration (ILS/m3)
#
# c_source_substitute = [4.38, 4.38, 0.84, 3.24]
#
# # HYDRAULICS
#
# max_slope_area = 24.087  #%
# diameter = 0.150
# f = 0.025
# h_suction = 0
# h_outlet = 20
# local_losses_coef = 0.158
# h_reservoir = 5
# density = 998.21
# pump_operation_time = 6         #hours
# price_average_kwh = 0.595 #ILS/kwh
# fixed_cost = 300            #ILS/month
# tax_percent = 0.16
#
# # Present value analysis coefficients
#
# DR = 0.0765
# n = 20
#
# # Watershed delineation coefficients
#
# snap_distance = 50 # [integer] Pixels to search around      #original = 10 before = 50
# flow_threshold = 20 # [integer] Threshold for flow         #original = 10  before = 20
