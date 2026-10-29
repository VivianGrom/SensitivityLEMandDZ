#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 07:31:53 2024

@author: viviangrom
"""

import os
import time
import warnings
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

from landlab.io.netcdf import write_netcdf, read_netcdf
from landlab import RasterModelGrid, imshow_grid, NodeStatus
from landlab.components import (ConcentrationTrackerForSpaceA,
                                ConcentrationTrackerForSpaceB,
                                ConcentrationTrackerForSpaceC,
                                ConcentrationTrackerForDiffusionA,
                                ConcentrationTrackerForDiffusionB,
                                ConcentrationTrackerForDiffusionC,
                                PriorityFloodFlowRouter,
                                SpaceLargeScaleEroder,
                                DepthDependentTaylorDiffuser,
                                Lithology,
                                LithoLayers,
                                )

# %% Change path
path_project = r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_layers'
os.chdir(path_project)

# Model scenario name
scenario_name = 'Aug8_10'

base_name = 'feb26_Base_SmllRect_n1_1Mry.nc'
path_base = "/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_layers"

# %% Define model grid, time, and component parameters
warnings.filterwarnings('ignore')
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
cmap_terrain = mpl.cm.get_cmap("terrain").copy()
cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()
cmap_C = mpl.cm.get_cmap("Reds").copy()

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
total_t = 250000 
grid_seed = 1
uplift_rate = 0.001 #m/yr

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

# -------------------------------- Don't touch --------------------------------
n_nodes = nrows*ncols
n_core_nodes = (nrows-2)*(ncols-2)
node_next_to_outlet = ncols + 1
uplift_per_step = uplift_rate * dt
ndt = int(total_t // dt)

# %% Things to plot and save

# Create empty arrays to fill during loop
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


#%%Configure lithology and layers

K_soft = 0.00005 
K_ratio = 2

#Erodibility
K_hard = K_soft / K_ratio

#Used for deposition only in case litholayers needs to deposit something (it shouldn't)
K_avg = (K_soft + K_hard) / 2


#Set the erodibility (K) of each rock layer using a dict
lith_attrs = {'K_sp': {1: 0.0005, 
                       2: 0.00005,
                       3: 0.0005},
              
              'C_zircons_bedrock_A':{1: 1, 
                                     2: 0,
                                     3: 0},
              'C_zircons_soil_A'   :{1: 1, 
                                     2: 0,
                                     3: 0},
              
              'C_zircons_bedrock_B':{1: 0, 
                                     2: 1,
                                     3: 0},
              'C_zircons_soil_B'   :{1: 0, 
                                     2: 1,
                                     3: 0},
              
              'C_zircons_bedrock_C':{1: 0, 
                                     2: 0,
                                     3: 1},
              'C_zircons_soil_C'   :{1: 0, 
                                     2: 0,
                                     3: 1}
              }

layer_thickness = 100 # meters

#layers repeat from 1-10 4 times - Just in case
layer_ids = list(range(1, 4))

#calculate max depth needed for given number of layers
max_depth =(layer_thickness*len(layer_ids))+layer_thickness

#make the bottom layer extra thick so it doesn't get eroded through      
layer_depths = np.arange(layer_thickness, max_depth, layer_thickness)
layer_depths[-1] += 500

#instantitae litholyaers
lith = LithoLayers(mg, 
                    layer_depths, 
                    layer_ids,
                    attrs=lith_attrs,
                    layer_type='MaterialLayers',
                    rock_id=2) 

K_sp_values = lith_attrs['K_sp']

# -----------------------------------------------------------------------------
check_CFL_condition(dt,ncols,nrows,dx,dy,max(K_sp_values.values()),max(K_sp_values.values()),m_sp)
# -----------------------------------------------------------------------------
# To develop a first topography, I'm using a fixed K value, and then 
# I will use the K values varying with the lithology 

# Erodibility in each zone
K_soil_init = 0.00005
K_bedrock_init = 0.00005

K_sed_field = np.zeros(n_nodes)
K_sed_field += K_soil_init

K_br_field = np.zeros(n_nodes)
K_br_field += K_bedrock_init


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

# %% Plot initial grid
# plot_grid_fields(mg,grid_fields_to_plot,cmaps)
    
# %% SCENARIO RUNS
   
# Update file export name
file_export_name = scenario_name

# Update concentrations
mg.at_node['sediment_property__concentration_A'] += mg.at_node['C_zircons_soil_A']
mg.at_node['bedrock_property__concentration_A'] += mg.at_node['C_zircons_bedrock_A']

mg.at_node['sediment_property__concentration_B'] += mg.at_node['C_zircons_soil_B']
mg.at_node['bedrock_property__concentration_B'] += mg.at_node['C_zircons_bedrock_B']

mg.at_node['sediment_property__concentration_C'] += mg.at_node['C_zircons_soil_C']
mg.at_node['bedrock_property__concentration_C'] += mg.at_node['C_zircons_bedrock_C']

# Create empty arrays to fill during loop
T = np.zeros(ndt)
record_C_out = np.zeros(ndt)
record_M_out = np.zeros(ndt)

# %% Plot initial grid
# plot_grid_fields(mg,grid_fields_to_plot,cmaps)

#%% To calculate erosion rate 
def get_id_matrix(mg, dz):
    """
    Generate an ID matrix based on lithology thickness values.

    Parameters:
    mg : Grid object
    dz : list or np.ndarray (thickness values for different layers)

    Returns:
    np.ndarray: The generated ID matrix.
    """

    if len(dz) >= 3:
        calc_dz = dz[0:3]
    else:
        zero_layer = np.zeros(dz[0].shape)  # Assuming dz[0] exists
        
        # Ensure dz has at least 3 layers by appending a zero array if needed
        while len(dz) < 3:
            dz = np.vstack([dz, [zero_layer]])
        
        calc_dz = dz
        
    # Get thickness values (dz) for the top three layers
    lith_dz_first = calc_dz[-1].reshape(mg.shape)
    lith_dz_second = calc_dz[-2].reshape(mg.shape)

    # Initialize id_matrix with zeros
    id_matrix = np.zeros(mg.shape, dtype=int)

    # Identify valid nodes based on lithology thickness
    valid_first = lith_dz_first != 0
    valid_second = (lith_dz_first == 0) & (lith_dz_second != 0)
    valid_third = (lith_dz_second == 0)

    # Assign values based on conditions
    id_matrix[valid_first] = 1
    id_matrix[valid_second] = 2
    id_matrix[valid_third] = 3

    # Compute area percentages
    total_cells = nrows * ncols  # Total number of grid cells
    first_percentage = np.sum(valid_first) / total_cells
    second_percentage = np.sum(valid_second) / total_cells
    third_percentage = np.sum(valid_third) / total_cells 
    
    lith_fractions = [first_percentage, second_percentage, third_percentage]

    return id_matrix, lith_fractions

# %% Model Run 1

# Set elapsed model time to 0 years
elapsed_time = 0
elevation_prev_step = None
inverted_layer_ids = layer_ids[::-1] 

# Initialize lists for erosion and zircon concentration storage
all_avg_e_rates_a, all_std_e_rates_a = [], []
all_avg_e_rates_b, all_std_e_rates_b = [], []
all_avg_e_rates_c, all_std_e_rates_c = [], []
all_zircon_concentration_a = []
all_zircon_concentration_b = []
all_zircon_concentration_c = []
all_lith_fractions = []

# Set initial timestamp (used to print progress updates)
start_time = time.time()

for i in range(ndt):
    # Update time counter
    elapsed_time = i*dt
    T[i] = elapsed_time 
    
    # Create id_matrix to calculate erosion in the right units
    if i == 0:
        id_matrix_b4 = get_id_matrix(mg, lith.dz)
    
    # Add uplift and update topographic elevation to match
    mg.at_node['bedrock__elevation'][mg.core_nodes] += uplift_per_step
    mg.at_node['topographic__elevation'][:] = (mg.at_node["bedrock__elevation"]
                                                + mg.at_node["soil__depth"]) 
    
    # Skip first iteration (no previous elevation to compute erosion)
    if elevation_prev_step is not None:
        change_in_elevation = elevation_prev_step - mg.at_node['topographic__elevation']
        dz = np.reshape(change_in_elevation, (nrows, ncols))
    
        # Determine uplift rate based on time
        u_dt = np.ones((nrows, ncols)) * uplift_rate * dt  
    
        # Calculate erosion rate
        e_rate = ((u_dt - dz) / dt) * 1e3  # Convert from m/yr to mm/yr
        
        # Create id_matrix to calculate erosion in the right units
        id_matrix, lith_fractions = get_id_matrix(mg, lith.dz)
        
        # Add the new field to the grid
        mg.add_field("id_matrix", id_matrix, at="node", clobber=True)
        
        # Define zones from bedrock property concentration
        node_x_values = mg.node_x[np.arange(0,ncols)]    
        zone_a = np.zeros(n_nodes, dtype=bool)
        zone_b = np.zeros(n_nodes, dtype=bool)
        zone_c = np.zeros(n_nodes, dtype=bool)
               
        zone_a[mg.at_node["id_matrix"]==1] = True
        zone_b[mg.at_node["id_matrix"]==2] = True
        zone_c[mg.at_node["id_matrix"]==3] = True
        
        zones = {'A' : zone_a,
                 'B' : zone_b,
                 'C' : zone_c
                  }

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
        all_lith_fractions.append(lith_fractions)
    
    # Run flow router, SPACE, ConcentrationTracker for SPACE
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
    
    #Layers are advected upwards due to uplift
    lith.dz_advection=uplift_per_step
    lith.run_one_step()

    #Update space K values
    sp.K_br = mg.at_node['K_sp']
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
    #Update C_zircons_bedrock values
    mg.at_node['bedrock_property__concentration_A'] = mg.at_node['C_zircons_bedrock_A']
    mg.at_node['bedrock_property__concentration_B'] = mg.at_node['C_zircons_bedrock_B']
    mg.at_node['bedrock_property__concentration_C'] = mg.at_node['C_zircons_bedrock_C']
    
    # Update previous elevation for next step
    elevation_prev_step = np.copy(mg.at_node['topographic__elevation']) 
    
    # Print progress and plot grids
    if i > 0 and i*dt % 1000 == 0:
        current_time = time.time()
        print_model_progress(start_time,current_time,elapsed_time,total_t,i,ndt,mg)
        # plot_grid_fields(mg,["topographic__elevation"],cmaps)
        # plot_grid_fields(mg,["bedrock_property__concentration_A"],cmaps)
        # plot_grid_fields(mg,["bedrock_property__concentration_B"],cmaps)
        # plot_grid_fields(mg,["bedrock_property__concentration_C"],cmaps)
        
    if i > 0 and i * dt % 10000 == 0:
        time_kyr = int(i * dt / 1000)  # Convert time to kyr
        fpath = f"{path_project}{file_export_name}_{time_kyr}kyr.nc"
        write_netcdf(fpath, mg, names=grid_fields_to_save)

# %% Generate dataframe of specified timeseries after model is complete
timeseries_to_save = {'Time (y)' : T[1:],
                      'Zircon concentration A at outlet (#/kg/m^3)' : all_zircon_concentration_a,
                      'Zircon concentration B at outlet (#/kg/m^3)' : all_zircon_concentration_b,
                      'Zircon concentration C at outlet (#/kg/m^3)' : all_zircon_concentration_c,
                      'Lith fractions (#%)' : all_lith_fractions
                      }
df1 = make_dataframe(timeseries_to_save)

# %% Post-run plotting
plot_grid_fields(mg,grid_fields_to_plot,cmaps)

df1 = df1.reset_index()

plt.figure(figsize=(10, 5))

# Plot each dataset with different colors and labels
plt.plot(df1["Time (y)"], df1["Zircon concentration A at outlet (#/kg/m^3)"], label="Unit A", color="b")
plt.plot(df1["Time (y)"], df1["Zircon concentration B at outlet (#/kg/m^3)"], label="Unit B", color="r")
plt.plot(df1["Time (y)"], df1["Zircon concentration C at outlet (#/kg/m^3)"], label="Unit C", color="g")

# Labels and title
plt.xlabel("Time (years)")
plt.ylabel("Zircon Concentration (#/kg/m³)")
plt.title("Zircon Concentration Over Time for Different Scenarios")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

# %% Export grid fields and timeseries
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
    'Zircon concentration C at outlet (#/kg/m^3)': all_zircon_concentration_c,
    'Lith fractions (#%)' : all_lith_fractions
})

# Export to CSV
export_to_csv(df_erosion, path_project, scenario_name + '_df')

mg.add_field("lith", lith.dz[-1], at="node", clobber=True)
plot_grid_fields(mg,["lith"],cmaps)
















