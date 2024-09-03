def load_scenario(idnb):
    if idnb==0:
        sc = load_s0()
    elif idnb==1:
        sc = load_s1()
    elif idnb==2:
        sc = load_s2()
    elif idnb==3:
        sc = load_s3()
    elif idnb==4:
        sc = load_s4()
    elif idnb==5:
        sc = load_s5()
    elif idnb==6:
        sc = load_s0_microwawes()
    return sc 


# ETP PARAMETERS
# ---------------------
CONSTANT_ETp = -1e-7 # in m/s
nb_days = 10 # simulation duration in days
# nb_hours_ET = 8 # nb of hours with ET

# IRRIGATION PARAMETERS
# ---------------------
irr_time_index = 3  #(= day nb)
irr_flow = 3e-7 #m/s (daily in mm/day)
# irr_length = 3*60*60 # irrigation length in sec
irr_length_days = 1 # irrigation length in sec

# RAIN PARAMETERS
# ---------------------
rain_flow = 6e-7 #m/s
rain_time_index = 6
# rain_length = 6*60*60 # irrigation length in sec
rain_length_days = 2 # irrigation length in sec

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

# Boundary conditions
# -------------------
nansfdirbc = 'no_flow'
nansfneubc = 'no_flow'
sfbc = 'no_flow'

def load_s0():
    # S0 is the simpliest scenario with a constant ETp, a dry soil, 
    scenario = {
        
                # SOIL PARAMETERS
                # ---------------
                'PMIN': PMIN,
                'pressure_head_ini': pressure_head_ini,

                # ETP PARAMETERS
                # ---------------------
                'ETp': CONSTANT_ETp,
                'nb_days': nb_days,
                # 'nb_hours_ET': nb_hours_ET,
                
                # BC PARAMETERS
                # ---------------------
                'nansfdirbc':'no_flow',
                'nansfneubc': 'no_flow',
                'sfbc': 'no_flow',
                                
                # IRRIGATION PARAMETERS
                # ---------------------
                'nb_irr_zones':1,
                'irr_time_index': [irr_time_index],
                # 'irr_length': [irr_length],
                'irr_flow': [irr_flow],
                'irr_center_point_x': [500], # in m
                'irr_center_point_y': [500], # in m
                'irr_square_size': [300],

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
    # Same than s0 with a rain event all over the regional domain before irrigation
    # -----------------------------------------------------------------------------
    scenario = load_s0()
    
    scenario['rain_flow'] = [rain_flow]
    scenario['rain_time_index'] = [rain_time_index]
    # scenario['rain_length'] = [rain_length]
    return scenario
        
        
def load_s2():
    # Same than s0 with 3 irrigation zones
    # ------------------------------------
    # 3 consecutives irrigations
    # The irrigation areas are varying 
    # Irrigation flow is constant 
    # 'ZROOT': [Root depth outside irr.,irr. area 1,irr. area 1,...],

    
    scenario = load_s0()
    
    # IRRIGATION PARAMETERS
    # ---------------------
    scenario_change = {
                'nb_irr_zones':3,
                'irr_time_index': [3,4,5],
                # 'irr_length': [irr_length,irr_length,irr_length],
                'irr_flow': [irr_flow,irr_flow,irr_flow],
                'irr_center_point_x': [200,300,650], # in m
                'irr_center_point_y': [800,250,650], # in m
                'irr_square_size': [100,200,300],
                'ZROOT': [1,0.3,0.3,0.3],

    }
    
    scenario = scenario | scenario_change      
    return scenario

def load_s3():
    # Same than s2 with varying irrigation flow
    # ---------------------------------------
    scenario = load_s2()
    
    # IRRIGATION PARAMETERS
    # ---------------------
    scenario_change = {
                'nb_irr_zones':3,
                'irr_time_index': [3,4,5],
                # 'irr_length': [irr_length,irr_length,irr_length],
                'irr_flow': [irr_flow/2,irr_flow,irr_flow*2],
                'irr_square_size': [100,100,100],
    }
    
    scenario = scenario | scenario_change      
    return scenario

def load_s4():
    # Same than s2 with varying rooting depths
    # ---------------------------------------
    scenario = load_s2()
    
    # IRRIGATION PARAMETERS
    # ---------------------
    scenario_change = {
                'ZROOT': [1,0.3,0.6,0.9],
                'irr_square_size': [100,100,100],
    }
    
    scenario = scenario | scenario_change      
    return scenario

def load_s5():
    # Same than s2 with varying soil parameters
    # ---------------------------------------
    scenario = load_s2()
    
    # IRRIGATION PARAMETERS
    # ---------------------
    scenario_change = {
                'PERMX': [5e-7,5e-7,5e-6,1e-6],
                'irr_square_size': [100,100,100],
    }
    
    scenario = scenario | scenario_change      
    return scenario


def load_s0_microwawes():
    # Same than s0 with two irrigation zones
    # ---------------------------------------
    scenario = load_s0()
    
    # IRRIGATION PARAMETERS
    # ---------------------
    scenario_change = {
                        'microwaweMesh': True,
                        # 'microwaweMesh': True

    }
    
    scenario = scenario | scenario_change      
    return scenario
        


def load_s1_withDA():
    # Same than s0 with two irrigation zones
    # ---------------------------------------
    scenario = load_s1_season()
    
    # IRRIGATION PARAMETERS
    # ---------------------
    scenario_change = {
                        'microwaweMesh': True,
                        # 'microwaweMesh': True

    }
    
    scenario = scenario | scenario_change      
    return scenario
        







