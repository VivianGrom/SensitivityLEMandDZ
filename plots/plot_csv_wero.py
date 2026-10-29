#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:57:06 2025

@author: viviangrom
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

# Change path
path_project = r'/Users/viviangrom/Documents/Archive'
os.chdir(path_project)

# Load datasets

# Layers
df10 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_layers/Aug8_10_df.csv', delimiter=",")
df5 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_layers/Aug8_2_df.csv', delimiter=",")
df1 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_layers/Aug8_1_df.csv', delimiter=",")
dfpt5 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_layers/Aug8_pt5_df.csv', delimiter=",")
dfpt1 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_layers/Aug8_pt1_df.csv', delimiter=",")

# ## Static
# df10 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_static/Jun29_Static10_df_clipped.csv', delimiter=",")
# df5 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_static/Jun29_Static2_df_clipped.csv', delimiter=",")
# df1 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_static/Jun29_Static1_df_clipped.csv', delimiter=",")
# dfpt5 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_static/Jun29_Staticpt5_df_clipped.csv', delimiter=",")
# dfpt1 = pd.read_csv(r'/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/experiments_to_stats_static/Jun29_Staticpt1_df_clipped.csv', delimiter=",")


# Function to plot zircon concentrations and erosion rates
def plot_data(df, title):
    
    df["Lith fractions (#%)"] = df["Lith fractions (#%)"].apply(ast.literal_eval)
    df[["Lith A", "Lith B", "Lith C"]] = pd.DataFrame(df["Lith fractions (#%)"].tolist(), index=df.index)
    
    total_area_km2 = 50
    df["Lith A (km2)"] = df["Lith A"] * total_area_km2
    df["Lith B (km2)"] = df["Lith B"] * total_area_km2
    df["Lith C (km2)"] = df["Lith C"] * total_area_km2

        # Set larger font globally
    plt.rcParams.update({
        "font.size": 16,         # base font size
        "axes.titlesize": 14,    # title font
        "axes.labelsize": 16,    # axis labels
        "xtick.labelsize": 18,   # x tick labels
        "ytick.labelsize": 18,   # y tick labels
        "legend.fontsize": 14,   # legend
    })
   
    # fig, ax1 = plt.subplots(figsize=(9, 5))
    
    # # Plot zircon concentrations
    # ax1.set_xlabel("Time (years)")
    # ax1.set_ylabel("Zircon Concentration (#/kg/m³)")
    # ax1.set_ylim(-0.1, 1.1)  # Set limits from 0 to 4
    # ax1.plot(df["Time"], df["Zircon concentration A at outlet (#/kg/m^3)"], label="Conc. A", color="b")
    # ax1.plot(df["Time"], df["Zircon concentration B at outlet (#/kg/m^3)"], label="Conc. B", color="r")
    # ax1.plot(df["Time"], df["Zircon concentration C at outlet (#/kg/m^3)"], label="Conc. C", color="g")
    # # ax1.set_xticks(range(0, int(df["Time"].max()) + 1, 10000))  # Ticks every 100 years
    # ax1.legend(loc="center left")
    # ax1.grid(True)
    # # plt.title(f"{title}")
    # # plt.show()
    
    # # Create secondary y-axis for erosion rates
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Erosion Rate (mm/yr)")
    # ax2.set_ylim(0.9, 2.1)  # Set limits from 0 to 4
    # ax2.plot(df["Time"], df["avg_erosion_rate_A"], linestyle="dashed", label="Erosion A", color="b", alpha=0.6)
    # ax2.plot(df["Time"], df["avg_erosion_rate_B"], linestyle="dashed", label="Erosion B", color="r", alpha=0.6)
    # ax2.plot(df["Time"], df["avg_erosion_rate_C"], linestyle="dashed", label="Erosion C", color="g", alpha=0.6)
    
    
    
    # ax2.legend(loc="center right")
    
    
    
    
    
    # plt.title(f"{title}")
    
    # # Plot just erosion rate
    # ax1.set_xlabel("Time (years)")
    # ax1.set_ylabel("Erosion Rate (mm/yr)")
    # ax1.set_ylim(0.75, 2.25)  # Set limits from 0 to 4
    # ax1.plot(df["Time"], df["avg_erosion_rate_A"], linestyle="dashed", label="Erosion A", color="b")
    # ax1.plot(df["Time"], df["avg_erosion_rate_B"], linestyle="dashed", label="Erosion B", color="r")
    # ax1.plot(df["Time"], df["avg_erosion_rate_C"], linestyle="dashed", label="Erosion C", color="g")
    # ax1.legend(loc="lower left")
    # ax1.grid(True)
    # plt.title(f"{title}")
    
    # Second Figure: Lithological Fractions
    fig2, ax1 = plt.subplots(figsize=(9, 5))
    
    # Plot zircon concentrations
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Zircon Concentration (#/kg/m³)")
    ax1.set_ylim(-0.1, 1.1)  # Set limits from 0 to 4
    ax1.plot(df["Time"], df["Zircon concentration A at outlet (#/kg/m^3)"], label="Unit A", color="b")
    ax1.plot(df["Time"], df["Zircon concentration B at outlet (#/kg/m^3)"], label="Unit B", color="r")
    ax1.plot(df["Time"], df["Zircon concentration C at outlet (#/kg/m^3)"], label="Unit C", color="g")
    # ax1.set_xticks(range(0, int(df["Time"].max()) + 1, 10000))  # Ticks every 100 years
    # ax1.legend(loc="best")
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    # ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Areal Extent (km²)")
    ax2.set_ylim(-5, 55)
    ax2.plot(df["Time"], df["Lith A (km2)"], label="Unit A", linestyle="dotted", color="b")
    ax2.plot(df["Time"], df["Lith B (km2)"], label="Unit B", linestyle="dotted", color="r")
    ax2.plot(df["Time"], df["Lith C (km2)"], label="Unit C", linestyle="dotted", color="g")
    ax2.legend(loc="best")
    # ax1.grid(True)
    
    # plt.title("Areal Extent of Units")
    plt.show()

    # # Third Figure: Erosion Rates
    # fig3, ax1 = plt.subplots(figsize=(10, 5))
    
    # ax1.set_xlabel("Time (years)")
    # ax1.set_ylabel("Erosion Rate (mm/yr)")
    # ax1.set_ylim(0.9, 2.1)
    # ax1.plot(df["Time"], df["avg_erosion_rate_A"], label="ER A", color="b")
    # ax1.plot(df["Time"], df["avg_erosion_rate_B"], label="ER B", color="r")
    # ax1.plot(df["Time"], df["avg_erosion_rate_C"], label="ER C", color="g")
    # ax1.legend(loc="best")
    # ax1.grid(True)
    
    # # plt.title("Erosion Rate (mm/yr)")
    # plt.show()

# Plot all datasets
# plot_data(df4, "Zircon Concentration & Erosion Rates for KA/KB = 10")
# plot_data(df2, "Zircon Concentration & Erosion Rates for KA/KB = 5")
# plot_data(df1, "Zircon Concentration & Erosion Rates for KA/KB = 1")
# plot_data(dfpt5, "Zircon Concentration & Erosion Rates for KA/KB = 0.5")
# plot_data(dfpt25, "Zircon Concentration & Erosion Rates for KA/KB = 0.1")

# plot_data(df10, "Zircon Concentration & Erosion Rates df_10")
# plot_data(df5, "Zircon Concentration & Erosion Rates df_5")
# plot_data(df1, "Zircon Concentration & Erosion Rates df_1")
# plot_data(dfpt5, "Zircon Concentration & Erosion Rates df_pt5")
# plot_data(dfpt1, "Zircon Concentration & Erosion Rates df_pt1")


# Grid and time parameters
nrows = 50 * 0.1
ncols = 100 * 0.1

area = nrows * ncols #km2

print (area)


plot_data(df10, "ZC df_10")
plot_data(df5, "ZC df_2")
plot_data(df1, "ZC df_1")
plot_data(dfpt5, "ZC df_pt5")
plot_data(dfpt1, "ZC df_pt1")

