

########################
#####  Synthetic   #####
########################


# Scenario 0 
# -----------------------------------------------------------------------
python DigTWIN_irrigation_delineation.py -scenario_nb 0 -run_process 1


# Scenario 1
# -----------------------------------------------------------------------
python DigTWIN_irrigation_delineation.py -scenario_nb 1 -run_process 1

python DigTWIN_irrigation_delineation.py -scenario_nb 2 -run_process 1
python DigTWIN_irrigation_delineation.py -scenario_nb 3 -run_process 1
python DigTWIN_irrigation_delineation.py -scenario_nb 4 -run_process 1
python DigTWIN_irrigation_delineation.py -scenario_nb 5 -run_process 1


###############################
#####  Synthetic AQUACROP  ####
###############################
python DigTWIN_scenarii_AquaCrop.py -scenario_nb 0 -run_process 1 -weather_scenario reference -SMT 70
python DigTWIN_scenarii_AquaCrop.py -scenario_nb 0 -run_process 1 -weather_scenario plus20p_tp -SMT 70
python DigTWIN_scenarii_AquaCrop.py -scenario_nb 0 -run_process 1 -weather_scenario minus20p_tp -SMT 70
python DigTWIN_scenarii_AquaCrop.py -scenario_nb 0 -run_process 1 -weather_scenario plus25p_t2m -SMT 70




########################
##### Real dataset #####
########################

# Majadas 
# -----------------------------------------------------------------------
python Majadas_hydroModel.py -prj_name Majadas_daily_WTD1 -short 0 -WTD 2
python Majadas_hydroModel.py -prj_name Majadas_2024_WTD1 -short 1 -WTD 2
