def load_scenario(idnb):
    if idnb==0:
        sc = load_s0()
    return sc 


# ETP PARAMETERS
# ---------------------
ETp = -1e-7 # in m/s
nb_days = 10 # simulation duration in days
nb_hours_ET = 8 # nb of hours with ET

# IRRIGATION PARAMETERS
# ---------------------
irr_time_index = 3
irr_flow = 5e-7 #m/s
irr_length = 6*60*60 # irrigation length in sec

# RAIN PARAMETERS
# ---------------------


# EARTH OBSERVATIONS PARM
# -----------------------
EO_freq_days = 1


# DELINEATION ANALYSIS PARAMETERS
# -------------------
ETp_window_size_x = 10 # size of the window in pixels to compute the rolling mean regional ETp
# The resolution of one pixel is about 10m? (depending on the hydrological mesh build step) 



def load_s0():
    
    scenario = {
        
                # ETP PARAMETERS
                # ---------------------
                'ETp': ETp,
                'nb_days': nb_days,
                'nb_hours_ET': nb_hours_ET,
                
                # IRRIGATION PARAMETERS
                # ---------------------
                'irr_time_index': irr_time_index,
                'irr_length': irr_length,
                'irr_flow': irr_flow,
        
                # EARTH OBSERVATIONS PARM
                # -----------------------
                'EO_freq_days': EO_freq_days,
                
                # DELINEATION ANALYSIS PARAMETERS
                # --------------------------------
                'ETp_window_size_x': ETp_window_size_x,
                'ETp_window_size_y': ETp_window_size_x,

        
        }
    return scenario
        
    


