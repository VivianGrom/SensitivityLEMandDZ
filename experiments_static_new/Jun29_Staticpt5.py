#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:23:50 2025

@author: viviangrom
"""
# %% Change path
import os
path_project = r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/experiments_static_new/'
os.chdir(path_project)

# %% Import

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from library2 import (check_CFL_condition,
                      make_dataframe,
                      export_to_csv,
                      print_model_progress,
                      plot_timeseries,
                      plot_grid_fields,
                      )

from landlab import RasterModelGrid, imshow_grid, NodeStatus
from landlab.components import (ConcentrationTrackerForSpaceA,
                                ConcentrationTrackerForSpaceB,
                                ConcentrationTrackerForSpaceC,
                                ConcentrationTrackerForDiffusionA,
                                ConcentrationTrackerForDiffusionB,
                                ConcentrationTrackerForDiffusionC,
                                PriorityFloodFlowRouter,
                                SpaceLargeScaleEroder,
                                DepthDependentTaylorDiffuser
                                )

from landlab.io.netcdf import write_netcdf, read_netcdf

# other things
import os
import time, warnings, copy

#%% Import base scenario and define run
# Model scenario name
scenario_name = 'Jun29_Staticpt5'

base_name = 'feb26_Base_SmllRect_n1_1Mry.nc'
path_base = "/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/experiments_static_new"
file_export_name = f"{base_name}"  # Match exact filename

fpath = os.path.join(path_base, file_export_name)

# Read the NetCDF file
mg = read_netcdf(fpath)

mg.set_status_at_node_on_edges(right=NodeStatus.CLOSED,
                               top=NodeStatus.CLOSED,
                               left=NodeStatus.CLOSED,
                               bottom=NodeStatus.CLOSED)
mg.status_at_node[2500] = NodeStatus.FIXED_VALUE

# Now you can access the fields from the grid, such as:
topo_elevation = mg.at_node['topographic__elevation']
soil_depth = mg.at_node['soil__depth']
bedrock_elevation = mg.at_node['bedrock__elevation']
soil_production = mg.at_node['soil_production__rate']

# Concentration fields
C_sed_A = mg.add_zeros('sediment_property__concentration_A', at='node', units= ['kg/m^3','kg/m^3'])
C_br_A = mg.add_zeros('bedrock_property__concentration_A', at='node', units= ['kg/m^3','kg/m^3'])
C_sed_B = mg.add_zeros('sediment_property__concentration_B', at='node', units= ['kg/m^3','kg/m^3'])
C_br_B = mg.add_zeros('bedrock_property__concentration_B', at='node', units= ['kg/m^3','kg/m^3'])
C_sed_C = mg.add_zeros('sediment_property__concentration_C', at='node', units= ['kg/m^3','kg/m^3'])
C_br_C = mg.add_zeros('bedrock_property__concentration_C', at='node', units= ['kg/m^3','kg/m^3'])

grid_fields_to_plot = ['topographic__elevation',
                       'bedrock__elevation',
                       'soil__depth',
                       'sediment_property__concentration_A',
                       'bedrock_property__concentration_A',
                       'sediment_property__concentration_B',
                       'bedrock_property__concentration_B',
                       'sediment_property__concentration_C',
                       'bedrock_property__concentration_C',
                       ]

# Set colour maps for plotting
cmap_terrain = mpl.colormaps.get_cmap("terrain").copy()
cmap_soil = mpl.colormaps.get_cmap("YlOrBr").copy()
cmap_C = mpl.colormaps.get_cmap("Reds").copy()

cmaps = {'Terrain' : cmap_terrain,
         'Soil' : cmap_soil,
         'Concentration' : cmap_C,
         }
# plot_grid_fields(mg,grid_fields_to_plot,cmaps)

# %% Define model grid, time, and component parameters

# Grid and time parameters
nrows = 50
ncols = 100
dx = 100 #m
dy = dx
dt = 10          # (yr)
total_t = 4000000 # (yr) 8Ma
grid_seed = 1
uplift_rate = 0.001
uplift_rate2 = 0.0025

# SpaceLargeScaleEroder parameters
phi_sp=0.0
m_sp=0.5
n_sp=1.0
H_star=1
F_f_sp = 0.0
v_s = 1

# DepthDependentTaylorDiffuser parameters
D = 0.01
H_transport = 0.1

# Erodibility in each zone
K_soil = {'A' : 0.00005,
          'B' : 0.0001,
          'C' : 0.00005
          }
K_bedrock = {'A' : 0.00005,
             'B' : 0.0001,
             'C' : 0.00005
             }

# Concentration of zircons in each zone
C_zircons_soil_A = {'A' : 1,
                    'B' : 0,
                    'C' : 0
                    }
C_zircons_bedrock_A = {'A' : 1,
                       'B' : 0,
                       'C' : 0
                       }

# Concentration of zircons in each zone
C_zircons_soil_B = {'A' : 0,
                    'B' : 1,
                    'C' : 0
                    }
C_zircons_bedrock_B = {'A' : 0,
                       'B' : 1,
                       'C' : 0
                       }

# Concentration of zircons in each zone
C_zircons_soil_C = {'A' : 0,
                    'B' : 0,
                    'C' : 1
                    }
C_zircons_bedrock_C = {'A' : 0,
                       'B' : 0,
                       'C' : 1
                       }

# -------------------------------- Don't touch --------------------------------
n_nodes = nrows*ncols
n_core_nodes = (nrows-2)*(ncols-2)
node_next_to_outlet = ncols + 1
uplift_per_step = uplift_rate * dt
ndt = int(total_t // dt)
# -----------------------------------------------------------------------------
check_CFL_condition(dt,ncols,nrows,dx,dy,max(K_bedrock.values()),max(K_soil.values()),m_sp)
# -----------------------------------------------------------------------------
# %% Things to plot and save

# Create empty arrays to fill during loop
T = np.zeros(ndt)
record_C_out_A = np.zeros(ndt)
record_C_out_B = np.zeros(ndt)
record_C_out_C = np.zeros(ndt)

# Name timeseries and grid fields to plot and save
timeseries_to_save = {'Time (y)' : T,
                      'Zircon concentration A at outlet (#/kg/m^3)' : record_C_out_A,
                      'Zircon concentration B at outlet (#/kg/m^3)' : record_C_out_B,
                      'Zircon concentration C at outlet (#/kg/m^3)' : record_C_out_C,
                      }

grid_fields_to_save = ['topographic__elevation',
                       'bedrock__elevation',
                       'soil__depth',
                       'sediment_property__concentration_A',
                       'bedrock_property__concentration_A',
                       'sediment_property__concentration_B',
                       'bedrock_property__concentration_B',
                       'sediment_property__concentration_C',
                       'bedrock_property__concentration_C',
                       'soil_production__rate'
                       ]

grid_fields_to_plot = ['topographic__elevation',
                       'bedrock__elevation',
                       'soil__depth',
                       'sediment_property__concentration_A',
                       'bedrock_property__concentration_A',
                       'sediment_property__concentration_B',
                       'bedrock_property__concentration_B',
                       'sediment_property__concentration_C',
                       'bedrock_property__concentration_C',
                       ]


# %% Make equal-sized zones (rectangular or triangular)

node_x_values = mg.node_x[np.arange(0,ncols)]    
zone_a = np.zeros(n_nodes, dtype=bool)
zone_b = np.zeros(n_nodes, dtype=bool)
zone_c = np.zeros(n_nodes, dtype=bool)
    
line_boundary_ab = (1/3)*np.max(node_x_values)
line_boundary_bc = (2/3)*np.max(node_x_values)
       
zone_a[mg.node_x<line_boundary_ab] = True
zone_c[mg.node_x>line_boundary_bc] = True
zone_b[zone_a+zone_c == 0] = True

zones = {'A' : zone_a,
         'B' : zone_b,
         'C' : zone_c
         }

K_sed_field = np.zeros(n_nodes)
K_sed_field[zones['A']] += K_soil['A']
K_sed_field[zones['B']] += K_soil['B']
K_sed_field[zones['C']] += K_soil['C']

K_br_field = np.zeros(n_nodes)
K_br_field[zones['A']] += K_bedrock['A']
K_br_field[zones['B']] += K_bedrock['B']
K_br_field[zones['C']] += K_bedrock['C']

# %% Instantiate model components

fr = PriorityFloodFlowRouter(mg)
fr.run_one_step()

# diffusion
ddTd = DepthDependentTaylorDiffuser(
	mg,
	soil_transport_decay_depth = H_transport,
	nterms = 2,
	soil_transport_velocity = D,
	dynamic_dt = True,
	if_unstable = "raise",
	courant_factor = 0.9,
	)

sp = SpaceLargeScaleEroder(mg,
                           K_sed=K_sed_field,
                           K_br=K_br_field,
                           phi=phi_sp,
                           m_sp=m_sp,
                           n_sp=n_sp,
                           H_star=H_star,
                           F_f = F_f_sp,
                           v_s = v_s
                           )

ctdA = ConcentrationTrackerForDiffusionA(mg)

ctdB = ConcentrationTrackerForDiffusionA(mg)

ctdC = ConcentrationTrackerForDiffusionA(mg)

ctspA = ConcentrationTrackerForSpaceA(mg, 
                                    phi=phi_sp,
                                    fraction_fines=F_f_sp,
                                    settling_velocity=v_s
                                    )

ctspB = ConcentrationTrackerForSpaceB(mg, 
                                    phi=phi_sp,
                                    fraction_fines=F_f_sp,
                                    settling_velocity=v_s
                                    )

ctspC = ConcentrationTrackerForSpaceC(mg, 
                                    phi=phi_sp,
                                    fraction_fines=F_f_sp,
                                    settling_velocity=v_s
                                    )

# %% SCENARIO RUNS
   
# Update file export name
file_export_name = scenario_name

# Update concentrations
mg.at_node['sediment_property__concentration_A'][:] = 0
mg.at_node['sediment_property__concentration_A'][zones['A']] += C_zircons_soil_A['A']
mg.at_node['sediment_property__concentration_A'][zones['B']] += C_zircons_soil_A['B']
mg.at_node['sediment_property__concentration_A'][zones['C']] += C_zircons_soil_A['C']

mg.at_node['bedrock_property__concentration_A'][:] = 0
mg.at_node['bedrock_property__concentration_A'][zones['A']] += C_zircons_bedrock_A['A']
mg.at_node['bedrock_property__concentration_A'][zones['B']] += C_zircons_bedrock_A['B']
mg.at_node['bedrock_property__concentration_A'][zones['C']] += C_zircons_bedrock_A['C']

# Update concentrations
mg.at_node['sediment_property__concentration_B'][:] = 0
mg.at_node['sediment_property__concentration_B'][zones['A']] += C_zircons_soil_B['A']
mg.at_node['sediment_property__concentration_B'][zones['B']] += C_zircons_soil_B['B']
mg.at_node['sediment_property__concentration_B'][zones['C']] += C_zircons_soil_B['C']

mg.at_node['bedrock_property__concentration_B'][:] = 0
mg.at_node['bedrock_property__concentration_B'][zones['A']] += C_zircons_bedrock_B['A']
mg.at_node['bedrock_property__concentration_B'][zones['B']] += C_zircons_bedrock_B['B']
mg.at_node['bedrock_property__concentration_B'][zones['C']] += C_zircons_bedrock_B['C']

# Update concentrations
mg.at_node['sediment_property__concentration_C'][:] = 0
mg.at_node['sediment_property__concentration_C'][zones['A']] += C_zircons_soil_C['A']
mg.at_node['sediment_property__concentration_C'][zones['B']] += C_zircons_soil_C['B']
mg.at_node['sediment_property__concentration_C'][zones['C']] += C_zircons_soil_C['C']

mg.at_node['bedrock_property__concentration_C'][:] = 0
mg.at_node['bedrock_property__concentration_C'][zones['A']] += C_zircons_bedrock_C['A']
mg.at_node['bedrock_property__concentration_C'][zones['B']] += C_zircons_bedrock_C['B']
mg.at_node['bedrock_property__concentration_C'][zones['C']] += C_zircons_bedrock_C['C']

# Concentration fields
C_sed_A = mg.at_node['sediment_property__concentration_A']
C_br_A = mg.at_node['bedrock_property__concentration_A']

C_sed_B = mg.at_node['sediment_property__concentration_B']
C_br_B = mg.at_node['bedrock_property__concentration_B']

C_sed_C = mg.at_node['sediment_property__concentration_C']
C_br_C = mg.at_node['bedrock_property__concentration_C']

# %% Plot initial grid
# plot_grid_fields(mg,grid_fields_to_plot,cmaps)

# %% Model Run

# Set elapsed model time to 0 years
elapsed_time = 0
elevation_prev_step = None  # Initialize previous elevation

# Create empty arrays to fill during loop
T = np.zeros(ndt)
record_C_out_A = np.zeros(ndt)
record_C_out_B = np.zeros(ndt)
record_C_out_C = np.zeros(ndt)

# Initialize lists for erosion and zircon concentration storage
all_avg_e_rates_a, all_std_e_rates_a = [], []
all_avg_e_rates_b, all_std_e_rates_b = [], []
all_avg_e_rates_c, all_std_e_rates_c = [], []
all_zircon_concentration_a, all_zircon_concentration_b, all_zircon_concentration_c = [], [], []

# Set initial timestamp (used to print progress updates)
start_time = time.time()

# Define transition time for uplift rate change
transition_time = total_t / 2  

for i in range(ndt):
    # Update time counter
    elapsed_time = i * dt
    T[i] = elapsed_time
    
    # Change uplift rate after transition time
    uplift_per_step = uplift_rate * dt if elapsed_time <= transition_time else uplift_rate2 * dt

    # Apply uplift and update topographic elevation
    mg.at_node['bedrock__elevation'][mg.core_nodes] += uplift_per_step
    mg.at_node['topographic__elevation'][:] = (
        mg.at_node["bedrock__elevation"] + mg.at_node["soil__depth"]
    )

    # Skip first iteration (no previous elevation to compute erosion)
    if elevation_prev_step is not None:
        change_in_elevation = elevation_prev_step - mg.at_node['topographic__elevation']
        dz = np.reshape(change_in_elevation, (nrows, ncols))

        # Determine uplift rate based on time
        uplift_rate_current = uplift_rate if elapsed_time < transition_time else uplift_rate2
        u_dt = np.ones((nrows, ncols)) * uplift_rate_current * dt  

        # Calculate erosion rate
        e_rate = ((u_dt - dz) / dt) * 1e3  # Convert from m/yr to mm/yr

        # Compute erosion for Zones A, B, and C
        zone_a_mask = np.reshape(zones['A'], (nrows, ncols))
        zone_b_mask = np.reshape(zones['B'], (nrows, ncols))
        zone_c_mask = np.reshape(zones['C'], (nrows, ncols))

        avg_e_rate_a, std_e_rate_a = np.mean(e_rate[zone_a_mask]), np.std(e_rate[zone_a_mask])
        avg_e_rate_b, std_e_rate_b = np.mean(e_rate[zone_b_mask]), np.std(e_rate[zone_b_mask])
        avg_e_rate_c, std_e_rate_c = np.mean(e_rate[zone_c_mask]), np.std(e_rate[zone_c_mask])

        # Store erosion rates
        all_avg_e_rates_a.append(avg_e_rate_a)
        all_std_e_rates_a.append(std_e_rate_a)
        all_avg_e_rates_b.append(avg_e_rate_b)
        all_std_e_rates_b.append(std_e_rate_b)
        all_avg_e_rates_c.append(avg_e_rate_c)
        all_std_e_rates_c.append(std_e_rate_c)

        # Store zircon concentration
        all_zircon_concentration_a.append(C_sed_A[2601])
        all_zircon_concentration_b.append(C_sed_B[2601])
        all_zircon_concentration_c.append(C_sed_C[2601])

    # Update previous elevation for next step
    elevation_prev_step = np.copy(mg.at_node['topographic__elevation'])
    
    # Run flow router, SPACE, ConcentrationTracker for SPACE, and ConcetrationTracker for Diffusion
    ddTd.run_one_step(dt=dt)
    fr.run_one_step()
    ctspA.start_tracking()
    ctspB.start_tracking()
    ctspC.start_tracking()
    ctdA.start_tracking()
    ctdB.start_tracking()
    ctdC.start_tracking()
    sp.run_one_step(dt=dt)
    ctspA.stop_tracking(dt=dt)
    ctspB.stop_tracking(dt=dt)
    ctspC.stop_tracking(dt=dt)
    ctdA.stop_tracking(dt=dt)
    ctdB.stop_tracking(dt=dt)
    ctdC.stop_tracking(dt=dt)

    # Print progress and save model outputs
    if i > 0 and i * dt % 1000 == 0:
        current_time = time.time()
        print_model_progress(start_time, current_time, elapsed_time, total_t, i, ndt, mg)
        
    if i > 0 and i * dt % 1000 == 0:
        time_kyr = int(i * dt / 1000)  # Convert time to kyr
        fpath = f"{path_project}{file_export_name}_{time_kyr}kyr.nc"
        write_netcdf(fpath, mg, names=grid_fields_to_save)

# %% Create DataFrame and Export

fpath_final = str(path_project + file_export_name + '_final.nc')
write_netcdf(fpath_final, mg, names=grid_fields_to_save)

# Skip first record
df_erosion = pd.DataFrame({
    'Time': T[1:],  # Skip first timestep
    'avg_erosion_rate_A': all_avg_e_rates_a,
    'std_e_rate_A': all_std_e_rates_a,
    'avg_erosion_rate_B': all_avg_e_rates_b,
    'std_e_rate_B': all_std_e_rates_b,
    'avg_erosion_rate_C': all_avg_e_rates_c,
    'std_e_rate_C': all_std_e_rates_c,
    'Zircon concentration A at outlet (#/kg/m^3)': all_zircon_concentration_a,
    'Zircon concentration B at outlet (#/kg/m^3)': all_zircon_concentration_b,
    'Zircon concentration C at outlet (#/kg/m^3)': all_zircon_concentration_c
})

# Export to CSV
export_to_csv(df_erosion, path_project, scenario_name + '_df')
