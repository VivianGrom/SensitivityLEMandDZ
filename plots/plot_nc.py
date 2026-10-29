#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:32:47 2025

@author: viviangrom
"""

# %% Change path
import os
path_project = r'/Users/viviangrom/Documents/Archive'
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
from landlab.components import (ConcentrationTrackerForSpace,
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

# fpath4 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Aug8_10_final.nc'
# fpath2 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Aug8_2_final.nc'
fpath1 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Aug8_1_final.nc'
# fpathpt5 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Aug8_pt5_final.nc'
# fpathpt25 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Aug8_pt1_final.nc'

# fpath4 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Jun29_Static10_final.nc'
# fpath2 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Jun29_Static2_final.nc'
# fpath1 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Jun29_Static1_final.nc'
# fpathpt5 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Jun29_Staticpt5_final.nc'
# fpathpt25 = f'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/final_nc_files/Jun29_Staticpt1_final.nc'


# # Read the NetCDF file
# mg4 = read_netcdf(fpath4)

# mg4.set_status_at_node_on_edges(right=NodeStatus.CLOSED,
#                                top=NodeStatus.CLOSED,
#                                left=NodeStatus.CLOSED,
#                                bottom=NodeStatus.CLOSED)
# mg4.status_at_node[2500] = NodeStatus.FIXED_VALUE

# # Now you can access the fields from the grid, such as:
# topo_elevation = mg4.at_node['topographic__elevation']
# soil_depth = mg4.at_node['soil__depth']
# bedrock_elevation = mg4.at_node['bedrock__elevation']
# sediment_concentration_A = mg4.at_node['sediment_property__concentration_A']
# bedrock_concentration_A = mg4.at_node['bedrock_property__concentration_A']
# sediment_concentration_B = mg4.at_node['sediment_property__concentration_B']
# bedrock_concentration_B = mg4.at_node['bedrock_property__concentration_B']
# sediment_concentration_C = mg4.at_node['sediment_property__concentration_C']
# bedrock_concentration_C = mg4.at_node['bedrock_property__concentration_C']

# grid_fields_to_plot = ['topographic__elevation',
#                        'bedrock__elevation',
#                        'soil__depth',
#                        'sediment_property__concentration_A',
#                        'bedrock_property__concentration_A',
#                        'sediment_property__concentration_B',
#                        'bedrock_property__concentration_B',
#                        'sediment_property__concentration_C',
#                        'bedrock_property__concentration_C',
#                         ]

# # Set colour maps for plotting
# cmap_terrain = mpl.cm.get_cmap("terrain").copy()
# cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()
# cmap_C = mpl.cm.get_cmap("Reds").copy()

# cmaps = {'Terrain' : cmap_terrain,
#          'Soil' : cmap_soil,
#          'Concentration' : cmap_C,
#          }
# plot_grid_fields(mg4,grid_fields_to_plot,cmaps)

# # Read the NetCDF file
# mg2 = read_netcdf(fpath2)

# mg2.set_status_at_node_on_edges(right=NodeStatus.CLOSED,
#                                 top=NodeStatus.CLOSED,
#                                 left=NodeStatus.CLOSED,
#                                 bottom=NodeStatus.CLOSED)
# mg2.status_at_node[2500] = NodeStatus.FIXED_VALUE

# # Now you can access the fields from the grid, such as:
# topo_elevation = mg2.at_node['topographic__elevation']
# soil_depth = mg2.at_node['soil__depth']
# bedrock_elevation = mg2.at_node['bedrock__elevation']
# sediment_concentration_A = mg2.at_node['sediment_property__concentration_A']
# bedrock_concentration_A = mg2.at_node['bedrock_property__concentration_A']
# sediment_concentration_B = mg2.at_node['sediment_property__concentration_B']
# bedrock_concentration_B = mg2.at_node['bedrock_property__concentration_B']
# sediment_concentration_C = mg2.at_node['sediment_property__concentration_C']
# bedrock_concentration_C = mg2.at_node['bedrock_property__concentration_C']

# grid_fields_to_plot = ['topographic__elevation',
#                         'bedrock__elevation',
#                         'soil__depth',
#                         'sediment_property__concentration_A',
#                         'bedrock_property__concentration_A',
#                         'sediment_property__concentration_B',
#                         'bedrock_property__concentration_B',
#                         'sediment_property__concentration_C',
#                         'bedrock_property__concentration_C',
#                         ]

# # Set colour maps for plotting
# cmap_terrain = mpl.cm.get_cmap("terrain").copy()
# cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()
# cmap_C = mpl.cm.get_cmap("Reds").copy()

# cmaps = {'Terrain' : cmap_terrain,
#           'Soil' : cmap_soil,
#           'Concentration' : cmap_C,
#           }
# plot_grid_fields(mg2,grid_fields_to_plot,cmaps)

# Read the NetCDF file
mg1 = read_netcdf(fpath1)

mg1.set_status_at_node_on_edges(right=NodeStatus.CLOSED,
                                top=NodeStatus.CLOSED,
                                left=NodeStatus.CLOSED,
                                bottom=NodeStatus.CLOSED)
mg1.status_at_node[2500] = NodeStatus.FIXED_VALUE

# Now you can access the fields from the grid, such as:
topo_elevation = mg1.at_node['topographic__elevation']
soil_depth = mg1.at_node['soil__depth']
bedrock_elevation = mg1.at_node['bedrock__elevation']
sediment_concentration_A = mg1.at_node['sediment_property__concentration_A']
bedrock_concentration_A = mg1.at_node['bedrock_property__concentration_A']
sediment_concentration_B = mg1.at_node['sediment_property__concentration_B']
bedrock_concentration_B = mg1.at_node['bedrock_property__concentration_B']
sediment_concentration_C = mg1.at_node['sediment_property__concentration_C']
bedrock_concentration_C = mg1.at_node['bedrock_property__concentration_C']

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
plot_grid_fields(mg1,grid_fields_to_plot,cmaps)

# # Read the NetCDF file
# mgpt5 = read_netcdf(fpathpt5)

# mgpt5.set_status_at_node_on_edges(right=NodeStatus.CLOSED,
#                                 top=NodeStatus.CLOSED,
#                                 left=NodeStatus.CLOSED,
#                                 bottom=NodeStatus.CLOSED)
# mgpt5.status_at_node[2500] = NodeStatus.FIXED_VALUE

# # Now you can access the fields from the grid, such as:
# topo_elevation = mgpt5.at_node['topographic__elevation']
# soil_depth = mgpt5.at_node['soil__depth']
# bedrock_elevation = mgpt5.at_node['bedrock__elevation']
# sediment_concentration_A = mgpt5.at_node['sediment_property__concentration_A']
# bedrock_concentration_A = mgpt5.at_node['bedrock_property__concentration_A']
# sediment_concentration_B = mgpt5.at_node['sediment_property__concentration_B']
# bedrock_concentration_B = mgpt5.at_node['bedrock_property__concentration_B']
# sediment_concentration_C = mgpt5.at_node['sediment_property__concentration_C']
# bedrock_concentration_C = mgpt5.at_node['bedrock_property__concentration_C']

# grid_fields_to_plot = ['topographic__elevation',
#                         'bedrock__elevation',
#                         'soil__depth',
#                         'sediment_property__concentration_A',
#                         'bedrock_property__concentration_A',
#                         'sediment_property__concentration_B',
#                         'bedrock_property__concentration_B',
#                         'sediment_property__concentration_C',
#                         'bedrock_property__concentration_C',
#                         ]

# # Set colour maps for plotting
# cmap_terrain = mpl.cm.get_cmap("terrain").copy()
# cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()
# cmap_C = mpl.cm.get_cmap("Reds").copy()

# cmaps = {'Terrain' : cmap_terrain,
#           'Soil' : cmap_soil,
#           'Concentration' : cmap_C,
#           }
# plot_grid_fields(mgpt5,grid_fields_to_plot,cmaps)

# # Read the NetCDF file
# mgpt25 = read_netcdf(fpathpt25)

# mgpt25.set_status_at_node_on_edges(right=NodeStatus.CLOSED,
#                                 top=NodeStatus.CLOSED,
#                                 left=NodeStatus.CLOSED,
#                                 bottom=NodeStatus.CLOSED)
# mgpt25.status_at_node[2500] = NodeStatus.FIXED_VALUE

# # Now you can access the fields from the grid, such as:
# topo_elevation = mgpt25.at_node['topographic__elevation']
# soil_depth = mgpt25.at_node['soil__depth']
# bedrock_elevation = mgpt25.at_node['bedrock__elevation']
# sediment_concentration_A = mgpt25.at_node['sediment_property__concentration_A']
# bedrock_concentration_A = mgpt25.at_node['bedrock_property__concentration_A']
# sediment_concentration_B = mgpt25.at_node['sediment_property__concentration_B']
# bedrock_concentration_B = mgpt25.at_node['bedrock_property__concentration_B']
# sediment_concentration_C = mgpt25.at_node['sediment_property__concentration_C']
# bedrock_concentration_C = mgpt25.at_node['bedrock_property__concentration_C']

# grid_fields_to_plot = ['topographic__elevation',
#                         'bedrock__elevation',
#                         'soil__depth',
#                         'sediment_property__concentration_A',
#                         'bedrock_property__concentration_A',
#                         'sediment_property__concentration_B',
#                         'bedrock_property__concentration_B',
#                         'sediment_property__concentration_C',
#                         'bedrock_property__concentration_C',
#                         ]

# # Set colour maps for plotting
# cmap_terrain = mpl.cm.get_cmap("terrain").copy()
# cmap_soil = mpl.cm.get_cmap("YlOrBr").copy()
# cmap_C = mpl.cm.get_cmap("Reds").copy()

# cmaps = {'Terrain' : cmap_terrain,
#           'Soil' : cmap_soil,
#           'Concentration' : cmap_C,
#           }
# plot_grid_fields(mgpt25,grid_fields_to_plot,cmaps)

