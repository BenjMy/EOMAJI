
cd /home/z0272571a@CAMPUS.CSIC.ES/Nextcloud/BenCSIC/Codes/BenjMy/EOMAJI/lib/

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
python DigTWIN_scenarii_AquaCrop.py -scenario_nb 0 -run_process 1 -weather_scenario reference -SMT 70 -ApplyEOcons PERMX
python DigTWIN_scenarii_AquaCrop.py -scenario_nb 0 -run_process 1 -weather_scenario plus20p_tp -SMT 70
python DigTWIN_scenarii_AquaCrop.py -scenario_nb 0 -run_process 1 -weather_scenario minus20p_tp -SMT 70
python DigTWIN_scenarii_AquaCrop.py -scenario_nb 0 -run_process 1 -weather_scenario plus25p_t2m -SMT 70



#######################################
#####  Synthetic AQUACROP WITH DA  ####
#######################################
python DigTWIN_scenarii_AquaCrop_withDA.py -study ET_scenarii -sc 0 -weather_scenario reference -nens 24 -DAtype enkf_Evensen2009 -DAloc 1 -damping 1 -dataErr 5 -refModel EOMAJI_AquaCrop_sc0_weather_reference_SMT_70 -ApplyEOcons 0



########################
##### Real dataset #####
########################

# Majadas 
# -----------------------------------------------------------------------
python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_100 -short 1 -WTD 2 -SCF 1
python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_100 -short 0 -WTD 2 -SCF 1
python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_100 -short 1 -WTD 5 -SCF 1
python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_100 -short 0 -WTD 5 -SCF 1


python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_5000 -short 1 -WTD 2 -SCF 1
python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_5000 -short 0 -WTD 2 -SCF 1
python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_5000 -short 1 -WTD 5 -SCF 1
python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_5000 -short 0 -WTD 5 -SCF 1



python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_5000 -short 0 -WTD 5
python Majadas_hydroModel.py -prj_name Majadas -AOI Buffer_5000 -short 1 -WTD 5


# Majadas with DA
# -----------------------------------------------------------------------
python DA_ETa_Majadas.py -study ZROOT -sc 0 -nens 24 -DAtype enkf_Evensen2009 -dataErr 1e99 -refModel prj_name_Majadas_AOI_Buffer_100_WTD_5.0_short_0_SCF_1.0_OMEGA_1 -DA_loc_domain veg_map -short 0 -rainSeasonDA 0
python DA_ETa_Majadas.py -study ZROOT -sc 0 -nens 24 -DAtype enkf_Evensen2009 -dataErr 1e-13 -refModel prj_name_Majadas_AOI_Buffer_100_WTD_5.0_short_0_SCF_1.0_OMEGA_1 -DA_loc_domain veg_map -short 0 -rainSeasonDA 0

python DA_ETa_Majadas.py -study ZROOT -sc 1 -nens 32 -DAtype enkf_Evensen2009 -dataErr 1e99 -refModel prj_name_Majadas_AOI_Buffer_100_WTD_5.0_short_0_SCF_1.0_OMEGA_1 -DA_loc_domain veg_map -short 0 -rainSeasonDA 0
python DA_ETa_Majadas.py -study ZROOT -sc 1 -nens 32 -DAtype enkf_Evensen2009 -dataErr 1e-13 -refModel prj_name_Majadas_AOI_Buffer_100_WTD_5.0_short_0_SCF_1.0_OMEGA_1 -DA_loc_domain veg_map -short 0 -rainSeasonDA 0

