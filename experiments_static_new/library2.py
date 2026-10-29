# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:43:48 2023

@author: LaurentRoberge
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from landlab import imshow_grid


def check_CFL_condition(dt,ncols,nrows,dx,dy,K_br,K_sed,m_sp):
    "Check C-F-L Condition for maximum timestep length (FastScape)"
    "Δt <= Cmax * (Δx / (K * (A ** m)))"
    
    area = (ncols * dx) * (nrows * dy)
    dt_max = 1 * (dx / (max(K_br,K_sed) * (area ** m_sp)))
    if dt > dt_max:
        raise Exception(
            str("\nTimestep is too large (dt=" + 
                str(dt) + 
                "). \nMaximum stable timestep is dt=" +
                str(dt_max) + 
                " (from C-F-L condition)."
                )
            )
    
    return 

def make_dataframe(timeseries_dict):
    "Save timeseries to dataframe. Must have 'Time (y)' as first key, as it will index by this value."
    df = pd.DataFrame()
    for key in timeseries_dict:
        df[key] = timeseries_dict[key]
    # Set time column as index
    df.set_index('Time (y)', inplace=True)
    return df

def export_to_csv(df,path_project,file_export_name):
    "Export dataframe to .csv file with specified name."
    
    fname_export = str(os.path.join(path_project + "\\" + file_export_name + '.csv'))
    df.to_csv(fname_export, na_rep='NaN', index=True)
    
    return None

def make_provenance_zones(mg,n_nodes,ncols, dx, dy,zone_shape='rectangular'):
    "Choose between 3 rectangular or 3 triangular provenance zones."
    
    node_ids = np.arange(0,n_nodes)
    node_x_values = mg.node_x[np.arange(0,ncols)]
    
    zone_a = np.zeros(n_nodes, dtype=bool)
    zone_b = np.zeros(n_nodes, dtype=bool)
    zone_c = np.zeros(n_nodes, dtype=bool)
    
    if zone_shape == 'rectangular':
        line_boundary_ab = (1/3)*np.max(node_x_values)
        line_boundary_bc = (2/3)*np.max(node_x_values)
               
        zone_a[mg.node_x<line_boundary_ab] = True
        zone_c[mg.node_x>line_boundary_bc] = True
        zone_b[zone_a+zone_c == 0] = True
        
        zones = {'A' : zone_a,
                 'B' : zone_b,
                 'C' : zone_c,
                 }
        
    if zone_shape == 'triangular':
        line_boundary_ab = (3/2)*node_x_values
        line_boundary_bc = (2/3)*node_x_values
        
        for i in range(len(node_x_values)):
            node_x = node_x_values[i]
            node_ids_for_node_x = node_ids[mg.node_x==node_x]
            zone_a[node_ids_for_node_x] = [mg.node_y[node_ids_for_node_x]>line_boundary_ab[i]]
        
        for i in range(len(node_x_values)):
            node_x = node_x_values[i]
            node_ids_for_node_x = node_ids[mg.node_x==node_x]
            zone_c[node_ids_for_node_x] = [mg.node_y[node_ids_for_node_x]<line_boundary_bc[i]]
        
        zone_b[zone_a+zone_c == 0] = True
        
        zones = {'A' : zone_a,
                 'B' : zone_b,
                 'C' : zone_c,
                 }
    
    if zone_shape == 'two_rect_zones':
        line_boundary_ab = (1/2)*np.max(node_x_values)
               
        zone_a[mg.node_x<line_boundary_ab] = True
        zone_b[mg.node_x>line_boundary_ab] = True
        
        zones = {'A' : zone_a,
                 'B' : zone_b,
                 }
        
        
    # CHECK THAT ZONES ARE CORRECTLY SIZED AND PLACED
    # np.sum(zones['A'])
    # np.sum(zones['B'])
    # np.sum(zones['C'])
    # zone_check = np.zeros(n_nodes)
    # zone_check[zones['A']==1]=1
    # zone_check[zones['C']==1]=2
    # imshow_grid(mg,zone_check)
    
    return zones

def print_model_progress(start_time,current_time,elapsed_time,total_t,i,ndt,mg):
    
    runtime_per_step = (current_time - start_time)/i
    runtime_remaining = runtime_per_step * (ndt-i)
    runtime_remaining_mins = round(runtime_remaining/60,2)
   
    print("Years elapsed :   ", elapsed_time/1000, " ka (", round(elapsed_time*100/total_t,2), "%)")
    print("Runtime remaining :   ", runtime_remaining_mins, " min")
    print("Max soil depth :  ", round(np.max(mg.at_node['soil__depth'][mg.core_nodes]),3))
    print("Mean soil depth :  ", round(np.mean(mg.at_node['soil__depth'][mg.core_nodes]),3))
    print("--------------")
    
    return None

def plot_timeseries(df,run_name,col_key_to_plot):
    "Plot timeseries of concentration."
    
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(10,6)
    plt.xlabel('Time (y)', fontsize=18)
    plt.ylabel(col_key_to_plot, fontsize=18)
    plt.plot(df[col_key_to_plot])
    
    ax = plt.gca()
    plt.text(0, -0.1, run_name, 
             horizontalalignment='center', 
             verticalalignment='center',
             transform = ax.transAxes)
    
    plt.show()
        
    return None

def plot_grid_fields(mg,field_names,cmaps):
    "Plot various grid fields in top-down view."

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    for field in field_names:
        
        plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(10,6)
 
        
        if "property__concentration" in field:
            cmap=cmaps['Concentration']
            var_units='-/%m^3%'
            #lims = (0,100)
        elif "elevation" in field:
            cmap=cmaps['Terrain']
            var_units='m'
            lims = (np.max(mg.at_node["topographic__elevation"][mg.core_nodes]),
                    np.min(mg.at_node["topographic__elevation"][mg.core_nodes])
                    )
        elif field == "soil__depth":
            cmap=cmaps['Soil']
            var_units='m'
            lims = (np.max(mg.at_node["soil__depth"][mg.core_nodes]),
                    np.min(mg.at_node["soil__depth"][mg.core_nodes])
                    )        
        else:
            imshow_grid(mg,
                        field,
                        plot_name=field,
                        var_name=field,
                        color_for_closed='pink'
                        )
            plt.show()
            
        imshow_grid(mg,
                    field,
                    plot_name=field,
                    var_name=field,
                    var_units=var_units,
                    #limits=lims,
                    cmap=cmap,
                    color_for_closed='pink',
                    )
        plt.show()
    
    return None