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
import detritalpy.detritalFuncs as dFunc

#%% Adam, change here the paths
# Define file paths
df = pd.read_csv(r'path/Aug8_1_df_clipped.csv')
experiment_type = '1'

# Define the path to the Excel file
existing_excel_file = 'path/Aug10_dd_lay_1_s_smzr.xlsx'

# Load data
dataToLoad = [
    'path/INPUT_Rosenblume2021.xlsx',
    'path/ExampleDataset_1.xlsx',
    'path/ExampleDataset_2.xlsx',
    ]

#%%
main_df, main_byid_df, samples_df, analyses_df = dFunc.loadDataExcel(dataToLoad)

sampleList = [(['CMK-1m','CS-23m','FPK-8m','LPK-1m','NM-18m','RRK-11m','SK-20m','TCKKLC','ZG-15m','ZG-186m'],'Kootenai Formation'),
              (['18MT-21','18MT-27'],'Quadrant Formation'),
              (['19MT-3','19MT-8'],'Phosphoria Formation'),
              (['TRD','TRDL-FP'],'Dinwoody Formation'),
              (['POR-1','POR-2','POR-3'],'Point of Rocks Sandstone'),
              (['BUT-5','BUT-4','BUT-3','BUT-2','BUT-1'],'Butano Sandstone')
             ]

ages, errors, numGrains, labels = dFunc.sampleToData(sampleList,
                                                     main_byid_df,
                                                     sigma = '1sigma')

# Define sample list
pop1_samples = ['CMK-1m','CS-23m','FPK-8m','LPK-1m','NM-18m','RRK-11m','SK-20m','TCKKLC','ZG-15m','ZG-186m']
pop2_samples = ['POR-1','POR-2','POR-3']
pop3_samples = ['BUT-5','BUT-4','BUT-3','BUT-2','BUT-1']

# Extract age data from samples
ages1, errors1, numGrains1, labels1 = dFunc.sampleToData(pop1_samples, main_byid_df, sigma='1sigma')
pop1 = gp.GrainPopReal(ages1, errors1, numGrains1, labels1)

ages2, errors2, numGrains2, labels2 = dFunc.sampleToData(pop2_samples, main_byid_df, sigma='1sigma')
pop2 = gp.GrainPopReal(ages2, errors2, numGrains2, labels2)

ages3, errors3, numGrains3, labels3 = dFunc.sampleToData(pop3_samples, main_byid_df, sigma='1sigma')
pop3 = gp.GrainPopReal(ages3, errors3, numGrains3, labels3)

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
zr_radius = [50, 30, 50]    
zr_radius_std = [15, 9, 15]

# Loop over each combination of Zr ppm and erosion rate values
for zr_ppm_a, zr_ppm_b, zr_ppm_c, fr in zip(
        zr_ppm_values_A, zr_ppm_values_B, zr_ppm_values_C, fractions):
        
    # Set up Zr ppm and erosion rate values for both A and B
    zr_ppm_values = [zr_ppm_a, zr_ppm_b, zr_ppm_c]
    fraction_values = ast.literal_eval(fr)

    # Run your experiment function
    run_experiment_with_parameters(zr_ppm_values, zr_radius, zr_radius_std, fraction_values)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    