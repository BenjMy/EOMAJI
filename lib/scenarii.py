def load_scenario(idnb):
    if idnb==0:
        sc = load_s0()
    if idnb==1:
        sc = load_s1()
    if idnb==2:
        sc = load_s2()
    return sc 


# ETP PARAMETERS
# ---------------------
CONSTANT_ETp = -1e-7 # in m/s
nb_days = 10 # simulation duration in days
nb_hours_ET = 8 # nb of hours with ET

# IRRIGATION PARAMETERS
# ---------------------
irr_time_index = 3  #(= day nb)
irr_flow = 3e-7 #m/s
irr_length = 3*60*60 # irrigation length in sec

# RAIN PARAMETERS
# ---------------------
rain_flow = 5e-7 #m/s
rain_time_index = 6
rain_length = 6*60*60 # irrigation length in sec

# EARTH OBSERVATIONS PARM
# -----------------------
EO_freq_days = 1


# DELINEATION ANALYSIS PARAMETERS
# -------------------
ETp_window_size_x = 10 # size of the window in pixels to compute the rolling mean regional ETp
# The resolution of one pixel is about 10m? (depending on the hydrological mesh build step) 


# SOIL PARAMETERS
# ---------------
PMIN = -1e30
pressure_head_ini = -200

threshold_localETap = 0.3
threshold_regionalETap = 0.3

def load_s0():
    
    scenario = {
        
                # SOIL PARAMETERS
                # ---------------
                'PMIN': PMIN,
                'pressure_head_ini': pressure_head_ini,

                # ETP PARAMETERS
                # ---------------------
                'ETp': CONSTANT_ETp,
                'nb_days': nb_days,
                'nb_hours_ET': nb_hours_ET,
                
                # IRRIGATION PARAMETERS
                # ---------------------
                'nb_irr_zones':1,
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
                
                # THRESHOLD
                # ---------
                'threshold_localETap': threshold_localETap,
                'threshold_regionalETap': threshold_regionalETap,

        
        }
    return scenario
        
def load_s1():
    
    scenario = load_s0()
    
    scenario['rain_flow'] = rain_flow
    scenario['rain_time_index'] = rain_time_index
    scenario['rain_length'] = rain_length
    return scenario
        
        
def load_s2():
    
    scenario = load_s0()
    
    # IRRIGATION PARAMETERS
    # ---------------------
    scenario_change = {
                'nb_irr_zones':2,
                'irr_time_index': [3,4],
                'irr_length': [irr_length,irr_length],
                'irr_flow': [irr_flow,irr_flow],
                'irr_center_point_x': [300,500], # in m
                'irr_center_point_y': [500,500], # in m
                'irr_square_size': [30,100],
    }
    
    scenario = scenario | scenario_change      
    return scenario
        


