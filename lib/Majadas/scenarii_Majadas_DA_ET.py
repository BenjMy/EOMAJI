'''   
Model perturbations are: 
------------------------
'''

import numpy as np

def load_scenario(study='test',**kwargs):
    
    if 'test' in study:
        scenarii = test()
    elif 'ZROOT' in study:
        scenarii = ZROOT_scenarii(**kwargs)
    elif 'ET' in study:
        scenarii = ET_scenarii()
    return scenarii
        

# ic
# -------------------
pert_nom_ic = -1.5
pert_sigma_ic = 0.75

# ZROOT
# -------------------
pert_nom_ZROOT = 1
pert_sigma_ZROOT = 0.35
pert_sigma_ZROOT_narrow = 1e-11
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
time_decorrelation_len = 86400*30

# Ks
# -------------------
pert_nom_ks = 1.880e-04
pert_sigma_ks = 1.75


def ET_scenarii():
    scenarii = {
                                               
                                                
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
                                                 'listUpdateParm': ['St. var.',
                                                                    'ZROOT0',
                                                                    'ZROOT1'],
                                                 'listObAss': ['RS_ET'],
                                                 },
                                                            
            }
    return scenarii
        

#            }

def ZROOT_scenarii(**kwargs):
       
    nzones=2
    if 'nzones' in kwargs:
        nzones = kwargs.pop('nzones')
    
    pert_nom_ZROOT_IND = 1
    pZROOT_name = ['ZROOT' + str(i) for i in range(nzones)]
    pupdate_name = ['St. var.'] + pZROOT_name

    
    scenarii = {
            # # NON biased ZROOT normal distribution
            # # ------------------------------------
            'ZROOT_pert2zones_withUpd': 
                                                {'per_type': [None],
                                                 # 'per_name': [['ZROOT']*nzones],
                                                 'per_name': ['ZROOT'],
                                                 'per_nom':
                                                            [[pert_nom_ZROOT_IND]*nzones]
                                                            ,
                                                 'per_mean':
                                                            [[pert_nom_ZROOT_IND]*nzones]
                                                            ,    
                                                 'per_sigma':
                                                              [[pert_sigma_ZROOT]*nzones]
                                                            ,
                                                 'per_bounds': 
                                                                [[{'min':minZROOT,'max':maxZROOT}]*nzones]
                                                            ,
                                                 'sampling_type': 
                                                                  [['normal']*nzones]
                                                                ,
                                                 'transf_type':
                                                                [[None]*nzones]
                                                            ,
                                                 'listUpdateParm': pupdate_name,
                                                 'listObAss': ['RS_ET'],
                                                 },
                                                    
            # # UNBIASED ZROOT + ATMBC normal distribution
            # # ------------------------------------
            'ET_ZROOT_pert2zones_withUpd': 
                                                {'per_type': [None,None],
                                                  'per_name':['ZROOT','atmbc'],
                                                  'per_nom':[[pert_nom_ZROOT_IND]*nzones,pert_nom_atmbcETp],
                                                  'per_mean':[[pert_nom_ZROOT_IND]*nzones,pert_nom_atmbcETp],    
                                                  'per_sigma': [[pert_sigma_ZROOT]*nzones,pert_sigma_atmbcETp],
                                                  'per_bounds': [[{'min':minZROOT,'max':maxZROOT}]*nzones,None],
                                                  'sampling_type': [['normal']*nzones,'normal'],
                                                  'transf_type':[[None]*nzones,None],
                                                  'time_decorrelation_len': [None,time_decorrelation_len],
                                                  'listUpdateParm': pupdate_name,
                                                  'listObAss': ['RS_ET'],   
                                                  },    
                                                
                                                
            # # NON biased ZROOT normal distribution + Water table variations
            # # -------------------------------------------------------------
            'WTD_ZROOT_pert2zones_withUpd': 
                                                {'per_type': [None,None],
                                                 'per_name':['WTPOSITION','ZROOT'],
                                                 'per_nom':[pert_nom_WT,
                                                            [pert_nom_ZROOT_IND]*nzones
                                                            ],
                                                 'per_mean':[pert_nom_WT,
                                                            [pert_nom_ZROOT_IND]*nzones
                                                            ],    
                                                 'per_sigma': [pert_sigma_WT,
                                                              [pert_sigma_ZROOT]*nzones
                                                            ],
                                                 'per_bounds': [{'min':minWT,'max':maxWT},
                                                                [{'min':minZROOT,'max':maxZROOT}]*nzones
                                                            ],
                                                 'sampling_type': ['normal',
                                                                  ['normal']*nzones
                                                                ],
                                                 'transf_type':[None,
                                                                [None]*nzones
                                                            ],
                                                 'listUpdateParm': pupdate_name,
                                                 'listObAss': ['RS_ET'],
                                                 },
                                                
                                                
            }
        
    
    return scenarii
    
    



def test(nzones = 2):
    
    nnod = 441
    # nzones = 9
    # pert_nom_ZROOT_IND_tmp = list(np.linspace(0.25,1.25,9))
    # pert_nom_ZROOT_IND = [ pz + 0.25 for pz in pert_nom_ZROOT_IND_tmp]
    pert_nom_ZROOT_IND = [0.25,1.25]
    pZROOT_name = ['ZROOT' + str(i) for i in range(nzones)]
    pupdate_name = ['St. var.'] + pZROOT_name
    
# ------------------------------------------------------------- #
# Testing
# usually use an smaller ensemble size
# ------------------------------------------------------------- #
    scenarii = {
                    # # NON biased ZROOT narrow normal distribution
                    # # ------------------------------------
                    'ZROOT_pert2zones_withUpd': 
                                                        {'per_type': [None]*nzones,
                                                         'per_name': ['ZROOT']*nzones,
                                                         'per_nom':
                                                                    [pert_nom_ZROOT_IND]
                                                                    ,
                                                         'per_mean':
                                                                    [pert_nom_ZROOT_IND]
                                                                    ,    
                                                         'per_sigma':
                                                                      [[pert_sigma_ZROOT_narrow]*nzones]
                                                                    ,
                                                         'per_bounds': 
                                                                        [[{'min':minZROOT,'max':maxZROOT}]*nzones]
                                                                    ,
                                                         'sampling_type': 
                                                                          [['normal']*nzones]
                                                                        ,
                                                         'transf_type':
                                                                        [[None]*nzones]
                                                                    ,
                                                         'listUpdateParm': pupdate_name,
                                                         'listObAss': ['RS_ET'],
                                                         },
            
                }
    
    return scenarii




