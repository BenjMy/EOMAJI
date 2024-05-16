#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:03:19 2024
"""

import scenarii2pyCATHY
import pyCATHY
import numpy as np
from pyCATHY import CATHY
from pyCATHY.plotters import cathy_plots as cplt
from pyCATHY.importers import cathy_outputs as out_CT
import pyCATHY.meshtools as msh_CT
import utils


import os
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd

    
#%%

def read_outputs(simu):

    sw, sw_times = simu.read_outputs('sw')
    sw_df = pd.DataFrame(sw.T)
    
    psi = simu.read_outputs('psi')

    atmbc_df = simu.read_inputs('atmbc')
    
    
    fort777 = os.path.join(simu.workdir,
                            simu.project_name,
                            'fort.777'
                            )
    ETa = out_CT.read_fort777(fort777)
    ETa.drop_duplicates(inplace=True)
    
    # create a dictionnary
    out_data = {
                'times': sw_times,
                'sw': sw,
                'psi': psi,
                'atmbc_df': atmbc_df,   
                'ETa': ETa,
                }
    return out_data


#%%
def plot_3d_SatPre(path):

    cplt.show_vtk(
                    unit="pressure",
                    path=path,
                    savefig=True,
                    timeStep=4,
                    )
    
    cplt.show_vtk(
                    unit="saturation",
                    path=path,
                    savefig=True,
                    timeStep=4,
                    )
    
    #%%
    cplt.show_vtk_TL(
                    unit="saturation",
                    path=path,
                    savefig=True,
                    show=False,
                    # timeStep=4,
                    )
    
    #%%
    cplt.show_vtk_TL(
                    unit="saturation",
                    path=path,
                    savefig=True,
                    show=False,
                    # timeStep=4,
                    )
    
#%%

def plot_in_subplot(ax,
                    index,
                    out_with_IRR,
                    out_baseline,
                    prop='sw',
                    ):

    sw2plot_with_IRR =  out_with_IRR[prop].iloc[:,index].values
    sw2plot_baseline =  out_baseline[prop].iloc[:,index].values
    
    ax.plot(
            out_with_IRR['times'],
            sw2plot_with_IRR,
            label='Irrigated',
            marker='.',
            color='blue'
            )
    ax.plot(
            out_with_IRR['times'],
            sw2plot_baseline,
            label='Baseline',
            marker='.',
            color='red',
            )   
    
    ax.legend()
    ax.grid('major')
    # ax.set_title('Saturation')
    ax.set_xlabel('Time')
    ax.set_ylabel(prop)

#%%


def plot_1d_evol(simu,index,closest,
                 out_with_IRR,
                 out_baseline,
                 ETp,
                 axs
                 ):
    
   
    simu.show_input('atmbc',ax=axs[0])
    # simu_with_IRR.show_input('atmbc',ax=axs[0])

    utils.plot_in_subplot(
                            axs[1],
                            index,
                            out_with_IRR,
                            out_baseline,
                            prop='sw',
                            )
    
    
    utils.plot_in_subplot(
                            axs[2],
                            index,
                            out_with_IRR,
                            out_baseline,
                            prop='psi',
                            )
    
    ETa1d_index = np.where(out_with_IRR['ETa']['SURFACE NODE']==index[0])[0]
    ETa1d_with_IRR = out_with_IRR['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    ETa1d_baseline = out_baseline['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    
    indexplot = (3)
    axs[indexplot].plot(out_baseline['ETa'].time_sec.unique()[1:],
                        ETa1d_with_IRR,
                        label='Irr',
                        color='blue',
                        marker='*'
                        )
    axs[indexplot].plot(out_baseline['ETa'].time_sec.unique()[1:],
                        ETa1d_baseline,
                        label='baseline',
                        color='red',
                        marker='*'
                        )
    
    axs[indexplot].axhline(y=abs(ETp), 
                           color='k', 
                           linestyle='--', 
                           label='ETp')
    axs[indexplot].legend()
    axs[indexplot].set_xlabel('time')
    axs[indexplot].set_ylabel('ETa (m/s)')



