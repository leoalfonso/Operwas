##################################
# Modification of AOKP algorithm #
##################################
from numpy import *

##FORMULAS FOR USE IN NEW FUNCTIONS##

##DESIGN FLOW##
""""get from design flow Maria's calculate_flow_at_wwtp function = flow_wwtp"""""

##ESTIMATED EFFLUENT QUALITY##
# First calculate effluent concentration of each pollutant for each after removal by each Unit process (UP)
# Secondly: Calculate the Final effluent concentration of each treatment train consisting of the selected UPs.

"""
                    C_eff = C_inf*(1-Ri)
                    Where Ri is the removal efficiency (%) of a UP for a given pollutant
                    C_eff: Concentration of a given pollutant in the effluent
                    C_inf: Concentration of a given pollutant in the influent 
                    NB: C_inf (selected quality parameters) entering the treatment plant is given in the knowledge base
                   
                    But we are calculating effluent quality at the end of each chain the eliminating chains tha dont qualify
                    
                    
                    NOTE: But to calculate the final effluent quality of any generated train, use:
                    C_eff_TT = C_eff at the last UP in the train
                    C_eff_TT= C_inf* (1-[(1-r1)*(1-r2)*(1-rn)]) 
                    
                    =BOD,COD,FC,TC,NP, then compare to bod ,cod etc of the standard
                           
                     
     """

## NOW CHECK IF ACHIEVES REQUIRED QUALITY##
# First: Establish the required standard basing on the intended end use, Standards to be used include (Palestine or Isreal's, WHO, US-EPA guidelines)
""""
                    If reuse is agriculture, pick agriculture standard,
                    if it is urban pick urban reuse standard,
                    if it is no reuse, pick no reuse standard.
                    These standards are in the knowledge base
                    NOTE:Reuse is got from Maria's code. so input from Maria's code has to be the area of reuse that has the reuse tyoe, population, and flow data.
"""

# Then: Compare the TT effluent quality to the required effluent quality.
# NOTE: Standard shall be determined by the selected reuse type (Modification should be made to Maria's function to add more reuse times).
"""
                    For a TT to meet a given standard
                    C_eff_TT =< C_eff_stand (for a given end use)
                    Therefore:
                    If C_eff_TT >= C_stand : Eliminate and indicate Not Feasible
                    If C_eff_TT =< C_stand : then feasible 
                    Do this for all the generated TTs,
                    Return only chains that meet the standard, then move to cost calculations
     """

## COST ESTIMATIONS ##
# For all the selected feasible trains, estimate the costs
## 1. Capex ##
# Calculate the capital cost for all the units in a given Treatment train, the sum to get total cape of the TT#
# The capital costs are calculated using generated cost functions.
# Some functions use Flow (Q) while others use Person equivalent (PE) depending on the unit process type.
# The land cost are calculated separately from the Capex cost functions but added to the final TT CAPEX.
"""
                   We are basing on flow, Q to calculate our capital cos.
                   The formula below is for calculating CC for a unit process. 
                    CC_i = c1 * Qinf^c2   (Eqn1) 
    

                   Ignore the this net one.
                    Based on PE
                    CC_i = ??? (more research needed)
                    
                    
                    
                    So Now the total capital cost of a chain is calculated by Eqn 2
                    Total capital costs
                    CC_TT = Summation of CC_i for all the UPs that make up the train
                    CC_TT = Summation of Eqn1s for all unit processes in  the chain.
                    
                    
                    Where:
                    c1 and c2 are cost 
                    CC_i is the Capital cost of UP in a TT
                    Qinf is the design flow (ave) = flow_wwtp
                    CC_TT is the Capital cost of a given TT         
     """

## 2. Land cost ##
# First Calculate the area land requirement of each UP, the sum to get footprint (FP) of TT#
# The calculation for FP shall be based on either flow or person equivalent depending on the type of unit process technology
# To allow for other auxiliary facilities in the WWTP, 15% is added to the TT area requirement.
# The cost is then calculated using local costs in the area of the WWTP
"""
                We shall only use eqn 1 to calculated the footprint.
                    FP_i = l1 * Qinf^l2         (Eqn1)
                    
                    
                ignore the eqns 2, 3 and 4
                    FP_i= Ai * HRT * Qinf       (Eqn2)
                    or
                    FP_i = l1 * Qinf^c2         (Eqn2)
                    or
                    FP_i = l1 * PE^c2           (Eqn3)
                    
                    BUT TO CALCULATE FOOTPRINT OF THE ENTIRE CHAIN
                    
                    Total TT footprint
                    FP_TT= 1.15* Summation of FP_i for all the UPs that make up the train
                    FT_TT= 1.15* SUM of Eqn1s for all the unit processes in the chain.

                    Where:
                    Aj: Surface Area per m3 of influent to be treated by a specific UP. (m2/m3)
                    c1 & c2 are land requirement coefficients
                    HRT: hydraulic retention time for a given UP (d)
                    PE: Person equivalent 
                    Qinf: Influent flow rate (m3/d) = flow_wwtp
                    FP_i: Footrint of the UPs in a TT (m2)
                    FP_TT: Sum of all footrints of the UPs in a TT (m2)
                    
                    Land Cost
                    NOTE: Check out the unit land cost from Maria's work 
                    C_land_TT = FP_TT * (Unit land cost)
                    
                    Lets assume unit land cost is 100 USD/m2 (This value should be in knowledge base though)
                    
                             
                
     """

##NOW CALCULATE Total Capex for TT##
# Add land cost to capex#
"""
                   Now Total Capital Cost
                    CAPEX_TT = CC_TT + C_land_TT
                    
                    where: 
                    CAPEX_TT is total capital cost for a given TT
                    CC_TT is the Capital cost of a given TT 
                    C_land_TT is land cost for a given TT        
     """

## 2 OPEX ##
# Calculate the operational cost for all the units in a given Treatment train, the sum to get total opex of the TT#
# The operation costs are calculated using generated cost functions.
# The operation cost
"""
                    LETS USE Eqn 1: this is for each unit process. but to the total chain we sum this for all the unit processes in that chain.
                    OMC_i = m1 * Qinf^m2 (Eqn1)
                    
                    
                    Ignore Eqn 2 for now
                    OMC_i = c1 * CC_i (Eqn2)

                    Total O&M costs
                    OMC_TT = Summation of OMC_i for all the UPs that make up the train

                    Where:
                    c1 is O&M cost coefficient
                    CC_i is the Capital cost of a given UP
                    OMC_i is the O&M cost of UP in a TT           
     """

## Sludge production and Disposal Cost ##
#Here we are only going to calculate sludge generation# we shall not use it in the cost calculation yet.
"""
                    slp = c1 * BOD_rem (Eqn1)
                    or
                    slp = c1 * PE
                    or 
                    slp = c1 * V_ann 
                    
                    Where:
                    slp is sludge production 
                    ............................
                    ..................... (Continue from here)         
     """




## ENERGY CONSUMPTION##
# The energy consumption is calculated for purposes of evaluating the treatment train.
"""
                    Energy_conUP = e1*Qinf^e2
                    Sum all the energy consumption for the train then get genergy cost from
                    Energy_consumpTT= Summation of Energy_conUP
                    
                    Therefore Energy cost:
                    EnC= Engergy_consumTT* Unit Cost of energy
                    
                    But lets assume unit cost of energy is 100USD/Kwhh (This should be in the Knowledge base)
     
     """

##TOTAL COST OF TT##

"""
                    Energy_conUP = e1*Qinf^e2
                    Sum all the energy consumption for the train then get genergy cost from
                    Energy_consumpTT= Summation of Energy_conUP

                    Therefore Energy cost:
                    EnC= Engergy_consumTT* Unit Cost of energy

                    But lets assume unit cost of energy is 100USD/Kwhh (This should be in the Knowledge base)

     """



## AMOUNT OF RECOVERED RESOURCES (Target resources -water, Energy and Nutrients##
## 1. RECOVERED WATER##
## Reclaimed water=Influent flow(m3/d)*Recycling efficiency##
"'"


"'"

##CAPITAL RECOVERY FACTOR##

