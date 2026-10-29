#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:03:46 2024

@author: viviangrom
"""
import grainpop_v3 as gp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast

#%% Adam, change here the paths
# Define file paths
df = pd.read_csv(r'path/Aug8_pt1_df_clipped.csv')
experiment_type = '0.1'

# Define the path to the Excel file
existing_excel_file = 'path/Aug9_Simplept1.xlsx'

#%%
# Establish parent populations, example has 2 units
# Simple pop

# Center age of populations in Ma
cap1=[1000]
cap2=[2000]
cap3=[3000]

# Stdev of populations
stdap1=[150]
stdap2=[150]
stdap3=[150]

# Skew of populations
skap1=[0]
skap2=[0]
skap3=[0]

# Percentage represented by each population
pap1=[1]
pap2=[1]
pap3=[1]

# Uncertainty percentage to apply
unc1=0.025
unc2=0.025
unc3=0.025

# Number of total grains
tog1=5000
tog2=5000
tog3=5000

# Set seed numbers for each population - optional
seed1=[1]
seed2=[2]
seed3=[2]

# Establish parent populations, example has 2 units
# Complex pop

# # Center age of populations in Ma
# cap1=[1250,1750]
# cap2=[2000,2500]
# cap3=[2750,3250]
# # Stdev of populations
# stdap1=[150,150]
# stdap2=[150,150]
# stdap3=[150,150]
# # Skew of populations
# skap1=[0,0]
# skap2=[0,0]
# skap3=[0,0]
# # Percentage represented by each population
# pap1=[0.5,0.5]
# pap2=[0.5,0.5]
# pap3=[0.5,0.5]
# # Uncertainty percentage to apply
# unc1=0.025
# unc2=0.025
# unc3=0.03
# # Number of total grains
# tog1=5000
# tog2=5000
# tog3=5000
# # Set seed numbers for each population - optional
# seed1=[1,2]
# seed2=[4,5]
# seed3=[1,2]

# Generate parent populations
pop1=gp.GrainPop(cap1,stdap1,skap1,pap1,unc1,tog1,seeds=seed1)
pop1.pdp_func()
pop2=gp.GrainPop(cap2,stdap2,skap2,pap2,unc2,tog2,seeds=seed2)
pop2.pdp_func()
pop3=gp.GrainPop(cap3,stdap3,skap3,pap3,unc3,tog3,seeds=seed3)
pop3.pdp_func()

########
# Determine fractions of zircons on the basis of total mass and indvidual grains

# Fractions of landscape taken up by individual units
# This split will double as the "baseline" assumption of splits between populations


def run_experiment_with_parameters(zr_masses, zr_radius, zr_radius_std, fractions):
   
    # Generate zircon fractions
    zg = gp.ZirconGeneratorLL(fractions, zr_masses, zr_radius, zr_radius_std)

    # Specify the size of the sampled population in grains
    sample_size = 125
    # Specify the number of iterations
    num_iter = 500

    # Begin random sampling based on the different fractions
    null_mix = gp.Mixer((pop1, pop2, pop3), fractions, sample_size, num_iter)
    [null_ages, null_ages_u, null_pdps] = null_mix.sampler()
    print('Completed null mix')
    mass_mix = gp.Mixer((pop1, pop2, pop3), zg.zirc_mass_fractions, sample_size, num_iter)
    [mass_ages, mass_ages_u, mass_pdps] = mass_mix.sampler()
    print('Completed mass mix')
    num_mix = gp.Mixer((pop1, pop2, pop3), zg.zirc_num_fractions, sample_size, num_iter)
    [num_ages, num_ages_u, num_pdps] = num_mix.sampler()
    print('Completed num mix')

    # Convert pdps to numpy arrays and find mean and standard deviations of pdps
    null_pdps = np.array(null_pdps)
    mass_pdps = np.array(mass_pdps)
    num_pdps = np.array(num_pdps)

    # Generate comparison objects
    com1 = gp.Comparer(np.array(null_ages), np.array(mass_ages), null_pdps, mass_pdps)
    com2 = gp.Comparer(np.array(null_ages), np.array(num_ages), null_pdps, num_pdps)
    com3 = gp.Comparer(np.array(mass_ages), np.array(num_ages), mass_pdps, num_pdps)

    [ks_1, R2_1] = com1.both_compare()
    print('Completed comparison between null and mass')
    [ks_2, R2_2] = com2.both_compare()
    print('Completed comparison between null and num')
    [ks_3, R2_3] = com3.both_compare()
    print('Completed comparison between mass and num')

    # Create Series objects from the lists
    ks_1_0_series = pd.Series(ks_1[0], name='ks_1[0]')
    ks_1_1_series = pd.Series(ks_1[1], name='ks_1[1]')
    ks_2_0_series = pd.Series(ks_2[0], name='ks_2[0]')
    ks_2_1_series = pd.Series(ks_2[1], name='ks_2[1]')
    ks_3_0_series = pd.Series(ks_3[0], name='ks_3[0]')
    ks_3_1_series = pd.Series(ks_3[1], name='ks_3[1]')
    R2_1_0_series = pd.Series(R2_1[0], name='R2_1[0]')
    R2_1_1_series = pd.Series(R2_1[1], name='R2_1[1]')
    R2_2_0_series = pd.Series(R2_2[0], name='R2_2[0]')
    R2_2_1_series = pd.Series(R2_2[1], name='R2_2[1]')
    R2_3_0_series = pd.Series(R2_3[0], name='R2_3[0]')
    R2_3_1_series = pd.Series(R2_3[1], name='R2_3[1]')

    # Create a DataFrame by concatenating the Series
    df = pd.concat([ks_1_0_series, ks_1_1_series, ks_2_0_series, ks_2_1_series,
                    ks_3_0_series, ks_3_1_series, R2_1_0_series, R2_1_1_series,
                    R2_2_0_series, R2_2_1_series, R2_3_0_series, R2_3_1_series], axis=1)
    
    df['KA/KB'] = experiment_type

    # Check if the file exists
    if not os.path.exists(existing_excel_file):
        # If the file doesn't exist, create a new file and save the data
        df.to_excel(existing_excel_file, index=False)
        print(f"New Excel file created: {existing_excel_file}")
    else:
        # If the file exists, load it and append the new data
        existing_df = pd.read_excel(existing_excel_file)
        existing_df = pd.concat([existing_df, df], ignore_index=True)
        existing_df.to_excel(existing_excel_file, index=False)
        print(f"Data appended to existing Excel file: {existing_excel_file}")


#%% Getting KS and R2 values for the concentrations

# Extract the Time values
times = df['Time'].values.tolist()

# Extract Zircon concentration and erosion rate values for Zones A, B, and C
zr_ppm_values_A = df['Zircon concentration A at outlet (#/kg/m^3)'].values.tolist()
zr_ppm_values_B = df['Zircon concentration B at outlet (#/kg/m^3)'].values.tolist()
zr_ppm_values_C = df['Zircon concentration C at outlet (#/kg/m^3)'].values.tolist()
fractions = df['Lith fractions (#%)'].values.tolist()

# Constant parameters
zr_radius = [50, 50, 50]    
zr_radius_std = [15, 15, 15]

# Loop over each combination of Zr ppm and erosion rate values
for zr_ppm_a, zr_ppm_b, zr_ppm_c, fr in zip(
        zr_ppm_values_A, zr_ppm_values_B, zr_ppm_values_C, fractions):
        
    # Set up Zr ppm and erosion rate values for both A and B
    zr_ppm_values = [zr_ppm_a, zr_ppm_b, zr_ppm_c]
    fraction_values = ast.literal_eval(fr)

    # Run your experiment function
    run_experiment_with_parameters(zr_ppm_values, zr_radius, zr_radius_std, fraction_values)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    