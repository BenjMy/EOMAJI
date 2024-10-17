'''   
Model perturbations are: 
------------------------
'''

import numpy as np

def load_scenario(study='hetsoil',**kwargs):
    
    if 'ET_scenarii' in study:
        scenarii = ET_scenarii()
    return scenarii
        

# ic
# -------------------
pert_nom_ic = -1.5
pert_sigma_ic = 0.75

# ZROOT
# -------------------
pert_nom_ZROOT = [1]
pert_sigma_ZROOT = [0.35]
minZROOT = 0
maxZROOT = 2


# Water table
# -------------------
pert_nom_WT = 0.5
pert_sigma_WT = 0.75
minWT = 0.5
maxWT = 2

# Atmbc
# -------------------
pert_sigma_atmbcETp = 1e-7
pert_nom_atmbcETp= -2e-7
time_decorrelation_len = 40e3

# Ks
# -------------------
pert_nom_ks = 1.880e-04
pert_sigma_ks = 1.75


def ET_scenarii():
    scenarii = {
             'ZROOT_zones_withUpd': 
                                                 {'per_type': [None],
                                                  'per_name':['ZROOT'],
                                                  'per_nom':[pert_nom_ZROOT*2],
                                                  'per_mean':[pert_nom_ZROOT*2],    
                                                  'per_sigma': [pert_sigma_ZROOT*2],
                                                  'per_bounds': [{'min':minZROOT,'max':maxZROOT}
                                                                 ],
                                                  'sampling_type': ['normal'],
                                                  'transf_type':[None],
                                                  'listUpdateParm': ['St. var.', 'ZROOT'],
                                                  'listObAss': ['RS_ET'],
                                                  },                                  
                                                
            'WTD_ZROOT_zones_withUpd': 
                                                {'per_type': [None,None],
                                                 'per_name':['WTPOSITION', 'ZROOT'],
                                                 'per_nom':[pert_nom_WT,pert_nom_ZROOT],
                                                 'per_mean':[pert_nom_WT,pert_nom_ZROOT],    
                                                 'per_sigma': [pert_sigma_WT,pert_sigma_ZROOT],
                                                 'per_bounds': [{'min':minWT,'max':maxWT},
                                                                {'min':minZROOT,'max':maxZROOT}
                                                                ],
                                                 'sampling_type': ['normal','normal'],
                                                 'transf_type':[None,None,None],
                                                 'listUpdateParm': ['St. var.', 'ZROOT'],
                                                 'listObAss': ['RS_ET'],
                                                 },
                                                
            
            'ET_WTD_ZROOT_withZROOTUpdate': 
                                                {'per_type': [None,None,None],
                                                  'per_name':['WTPOSITION', 'ZROOT','atmbc'],
                                                  'per_nom':[pert_nom_WT,pert_nom_ZROOT,pert_nom_atmbcETp],
                                                  'per_mean':[pert_nom_WT,pert_nom_ZROOT,pert_nom_atmbcETp],    
                                                  'per_sigma': [pert_sigma_WT,pert_sigma_ZROOT,pert_sigma_atmbcETp],
                                                  'per_bounds': [{'min':minWT,'max':maxWT},{'min':minZROOT,'max':maxZROOT},None],
                                                  'sampling_type': ['normal','normal','normal'],
                                                  'transf_type':[None,None,None],
                                                  'time_decorrelation_len': [None,None,time_decorrelation_len],
                                                  'listUpdateParm': ['St. var.', 'ZROOT'],
                                                  'listObAss': ['RS_ET'],
                                                  },
            
            }
    return scenarii
        

