# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:45 2019

@author: Maria
"""

import xlrd

from src.faster import config

from .inputs import *

workbook = xlrd.open_workbook(path_user_inputs, encoding_override='utf-8')
worksheet = workbook.sheet_by_name('Sheet1')

print("User input data is being read!")

# Catchment characteristics
num_locations = worksheet.cell(2, 4).value
num_locations = int(num_locations)
total_population = worksheet.cell(3, 4).value
population_catchment = worksheet.cell(3, 4).value
# total_population = eval(total_population)
water_consumption = worksheet.cell(4, 4).value
water_consumption = eval(water_consumption)
print('Average water consumption of population, given for each location:', water_consumption)
losses_infiltration = worksheet.cell(5, 4).value
print('Percentage of water lost by infiltration, not reaching the wastewater collection system:', losses_infiltration)
coef_peak = worksheet.cell(6, 4).value
print('Hourly peak flow of wastewater generation:', coef_peak)
inhabit_per_household = worksheet.cell(7, 4).value
losses_distribution = worksheet.cell(8, 4).value
losses_distribution = eval(losses_distribution)
losses_transmission = worksheet.cell(9, 4).value
losses_transmission = eval(losses_transmission)
collection_efficiency = worksheet.cell(10, 4).value
collection_efficiency = eval(collection_efficiency)
urb_irrig_use_coef = worksheet.cell(11, 4).value

# Reuse areas assumptions
area_usage_cas = worksheet.cell(13, 4).value
area_usage_mbr = worksheet.cell(14, 4).value
irr_oper_agr = worksheet.cell(15, 4).value
irr_oper_urb = worksheet.cell(16, 4).value
nr_reuse_options = worksheet.cell(17, 4).value
nr_reuse_options = int(nr_reuse_options)
no_data_value_land_price = worksheet.cell(18, 4).value
no_data_value_loc_index = worksheet.cell(19, 4).value

ww_treatment_after_discharge = worksheet.cell(20, 4).value

# TECHNOLOGY 1 - Conventional Activated Sludge

known_flow_cas_1 = worksheet.cell(22, 4).value
known_flow_cas_1 = int(known_flow_cas_1)
known_flow_cas_2 = worksheet.cell(23, 4).value
known_flow_cas_2 = int(known_flow_cas_2)
known_inv_cost_cas_1 = worksheet.cell(24, 4).value
known_inv_cost_cas_1 = int(known_inv_cost_cas_1)
known_inv_cost_cas_2 = worksheet.cell(25, 4).value
known_inv_cost_cas_2 = int(known_inv_cost_cas_2)
known_oem_cost_cas_1 = worksheet.cell(26, 4).value
known_oem_cost_cas_1 = int(known_oem_cost_cas_1)
known_oem_cost_cas_2 = worksheet.cell(27, 4).value
known_oem_cost_cas_2 = int(known_oem_cost_cas_2)

# TECHNOLOGY 2 - Membrane Bioreactor

known_flow_mbr_1 = worksheet.cell(29, 4).value
known_flow_mbr_1 = int(known_flow_mbr_1)
known_flow_mbr_2 = worksheet.cell(30, 4).value
known_flow_mbr_2 = int(known_flow_mbr_2)
known_inv_cost_mbr_1 = worksheet.cell(31, 4).value
known_inv_cost_mbr_1 = int(known_inv_cost_mbr_1)
known_inv_cost_mbr_2 = worksheet.cell(32, 4).value
known_inv_cost_mbr_2 = int(known_inv_cost_mbr_2)
known_oem_cost_mbr_1 = worksheet.cell(33, 4).value
known_oem_cost_mbr_1 = int(known_oem_cost_mbr_1)
known_oem_cost_mbr_2 = worksheet.cell(34, 4).value
known_oem_cost_mbr_2 = int(known_oem_cost_mbr_2)

# Sanitation fees

fee_connection = worksheet.cell(36, 4).value
fee_sanitary = worksheet.cell(37, 4).value
fee_sanitary = eval(fee_sanitary)

# Water tariff

tariff = worksheet.cell(39, 4).value
tariff = eval(tariff)
tariff_reclaimed = worksheet.cell(40, 4).value
tariff_reclaimed = eval(tariff_reclaimed)

# Pipe costs for wastewater collection and reuse water distribution network

Pw = worksheet.cell(42, 4).value
Pw = eval(Pw)
hRan = worksheet.cell(43, 4).value
hRan = eval(hRan)

# Reservoir costs

known_reservoir_volume_1 = worksheet.cell(45, 4).value
known_reservoir_volume_2 = worksheet.cell(46, 4).value
known_inv_cost_reservoir_1 = worksheet.cell(47, 4).value
known_inv_cost_reservoir_2 = worksheet.cell(48, 4).value

# Fresh water energy and management costs

portion_s1 = worksheet.cell(50, 4).value
portion_s1 = eval(portion_s1)
portion_s2 = worksheet.cell(51, 4).value
portion_s2 = eval(portion_s2)
portion_s3 = worksheet.cell(52, 4).value
portion_s3 = eval(portion_s3)

c_prod_s1 = worksheet.cell(53, 4).value
c_prod_s1 = eval(c_prod_s1)
c_prod_s2 = worksheet.cell(54, 4).value
c_prod_s2 = eval(c_prod_s2)
c_prod_s3 = worksheet.cell(55, 4).value
c_prod_s3 = eval(c_prod_s3)
c_tran_s1 = worksheet.cell(56, 4).value
c_tran_s1 = eval(c_tran_s1)
c_tran_s2 = worksheet.cell(57, 4).value
c_tran_s2 = eval(c_tran_s2)
c_tran_s3 = worksheet.cell(58, 4).value
c_tran_s3 = eval(c_tran_s3)
c_tran_adm = worksheet.cell(59, 4).value
c_tran_adm = eval(c_tran_adm)
c_dist_staff = worksheet.cell(60, 4).value
c_dist_staff = eval(c_dist_staff)
c_dist_energy = worksheet.cell(61, 4).value
c_dist_energy = eval(c_dist_energy)
c_dist_adm = worksheet.cell(62, 4).value
c_dist_adm = eval(c_dist_adm)
c_source_substitute = worksheet.cell(63, 4).value
c_source_substitute = eval(c_source_substitute)

# HYDRAULICS

max_slope_area = worksheet.cell(65, 4).value
diameter = worksheet.cell(66, 4).value
f = worksheet.cell(67, 4).value
h_suction = worksheet.cell(68, 4).value
h_outlet = worksheet.cell(69, 4).value
local_losses_coef = worksheet.cell(70, 4).value
h_reservoir = worksheet.cell(71, 4).value
density = worksheet.cell(72, 4).value
pump_operation_time = worksheet.cell(73, 4).value
price_average_kwh = worksheet.cell(74, 4).value
fixed_cost = worksheet.cell(75, 4).value
tax_percent = worksheet.cell(76, 4).value

# Present value analysis coefficients

DR = worksheet.cell(78, 4).value
n = worksheet.cell(79, 4).value

# Watershed delineation coefficients

snap_distance = worksheet.cell(81, 4).value
flow_threshold = worksheet.cell(82, 4).value
print(flow_threshold)

print('All user input data imported well!')
