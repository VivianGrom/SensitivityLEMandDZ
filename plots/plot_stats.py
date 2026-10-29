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

time_init = 1.8*1e6
time_fin = 2.8*1e6
# Define the times for which you want to find the timestep indices
times = (np.linspace(time_init, time_fin, 100)).astype(int)
dt = 100

# %% Plot Stats
# Load the existing Excel file

# UPDATED static synthetic
existing_excel_file = '/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/static_experiments/Aug9_Simple_sta_syn.xlsx'
existing_excel_file_complex = '/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/static_experiments/Aug9_Complex_sta_syn.xlsx'

# # UPDATED static data-derived
# existing_excel_file = '/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/static_experiments/Aug10_Simple_sta_dd.xlsx'
# existing_excel_file_complex = '/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/static_experiments/Aug10_Complex_sta_dd.xlsx'

# # UPDATED static smzr synthetic
# existing_excel_file = '/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/static_experiments/Aug9_Simple_sta_syn_smzr.xlsx'
# existing_excel_file_complex = '/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/static_experiments/Aug9_Complex_sta_syn_smzr.xlsx'

# # UPDATED static smzr data-derived
# existing_excel_file = '/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/static_experiments/Aug10_Simple_sta_dd_smzr.xlsx'
# existing_excel_file_complex = '/Users/viviangrom/Documents/papers/paper1_2_sensitivity_model/codes/sent_to_adam/static_experiments/Aug10_Complex_sta_dd_smzr.xlsx'

# # layer synthetic
# existing_excel_file = '/Users/viviangrom/Documents/Archive/Mar25/Mar24_Simple1.xlsx'
# existing_excel_file_complex = '/Users/viviangrom/Documents/Archive/Mar25/Mar24_Complex1.xlsx'

# # layer data-derived
# existing_excel_file = '/Users/viviangrom/Documents/Archive/Apr10/Apr09_Simple_l.xlsx'
# existing_excel_file_complex = '/Users/viviangrom/Documents/Archive/Apr10/Apr09_Complex_l.xlsx'

# # layer smzr synthetic
# existing_excel_file = '/Users/viviangrom/Documents/Archive/Mar25/Mar24_Simple1_smzr.xlsx'
# existing_excel_file_complex = '/Users/viviangrom/Documents/Archive/Mar25/Mar24_Complex1_smzr.xlsx'

# # layer smzr data-derived
# existing_excel_file = '/Users/viviangrom/Documents/Archive/Apr10/Apr09_Simple_l_smzr.xlsx'
# existing_excel_file_complex = '/Users/viviangrom/Documents/Archive/Apr10/Apr09_Complex_l_smzr.xlsx'

existing_df = pd.read_excel(existing_excel_file)
existing_df_complex = pd.read_excel(existing_excel_file_complex)

# Define a list of labels for the plots
labels = ['Null to Mass', 'Null to Num', 'Mass to Num']

# Define a list of colors for the plots
colors = ['purple', 'orange', 'k']

# Group the data by the 'KA/KB' (experiment type) column
grouped = existing_df.groupby('KA/KB')

# def plot_experiment_results(existing_df, labels, colors, times):
#     # Group the data by the 'KA/KB' (experiment type) column
#     grouped = existing_df.groupby('KA/KB')

#     # Iterate over each experiment type
#     for experiment_type, group_df in grouped:
        
#         # Plot KS Test Statistic
#         plt.figure(figsize=(8, 5))
#         for i, label, color in zip(range(len(labels)), labels, colors):
#             ks_column = f'ks_{i+1}[0]'
#             ks_err_column = f'ks_{i+1}[1]'
            
#             plt.plot(times, group_df[ks_column], c=color, label=label)
#             plt.plot(times, group_df[ks_column], color=color, linestyle='-', marker='')
#             plt.fill_between(times, group_df[ks_column] - group_df[ks_err_column], 
#                              group_df[ks_column] + group_df[ks_err_column], color=color, alpha=0.2)
        
#         plt.ylim([0, 1])
#         plt.xlim(time_init, time_fin)
#         plt.xlabel('Time')
#         plt.ylabel('KS Test Statistic')
#         plt.title(f'KS Test Statistic - {experiment_type}')
#         plt.legend(loc='best')
#         plt.show()
        
#         # Plot PDP Cross Correlation
#         plt.figure(figsize=(8, 5))
#         for i, label, color in zip(range(len(labels)), labels, colors):
#             R2_column = f'R2_{i+1}[0]'
#             R2_err_column = f'R2_{i+1}[1]'
            
#             plt.plot(times, group_df[R2_column], c=color, label=label)
#             plt.plot(times, group_df[R2_column], color=color, linestyle='-', marker='')
#             plt.fill_between(times, group_df[R2_column] - group_df[R2_err_column], 
#                              group_df[R2_column] + group_df[R2_err_column], color=color, alpha=0.2)
        
#         plt.ylim([0, 1])
#         plt.xlim(time_init,time_fin)
#         plt.xlabel('Time')
#         plt.ylabel('PDP Cross Correlation')
#         plt.title(f'PDP Cross Correlation - {experiment_type}')
#         plt.legend(loc='best')
#         plt.show()
        
def plot_experiment_results(existing_df, labels, colors, times):
    # Group the data by experiment type
    grouped = existing_df.groupby('KA/KB')

    for experiment_type, group_df in grouped:
        
        # === KS Test Statistic (inverted to show similarity) ===
        plt.figure(figsize=(3, 2))
        for i, label, color in zip(range(len(labels)), labels, colors):
            ks_column = f'ks_{i+1}[0]'
            ks_err_column = f'ks_{i+1}[1]'
            
            ks_val = 1 - group_df[ks_column]  # invert KS
            ks_err = group_df[ks_err_column]

            plt.plot(times, ks_val, c=color, label=label)
            plt.fill_between(times, ks_val - ks_err, ks_val + ks_err, color=color, alpha=0.2)
        
        plt.ylim([0, 1])
        plt.xlim(time_init, time_fin)
        plt.xlabel('Time (Myr)')
        plt.ylabel('1 – KS')
        plt.title(f'{experiment_type}')
        # plt.legend(loc='best')
        plt.show()
        
        # === PDP Cross-Correlation ===
        plt.figure(figsize=(3, 2))
        for i, label, color in zip(range(len(labels)), labels, colors):
            R2_column = f'R2_{i+1}[0]'
            R2_err_column = f'R2_{i+1}[1]'
            
            plt.plot(times, group_df[R2_column], c=color, label=label)
            plt.fill_between(times, 
                             group_df[R2_column] - group_df[R2_err_column], 
                             group_df[R2_column] + group_df[R2_err_column], 
                             color=color, alpha=0.2)
        
        plt.ylim([0, 1])
        plt.xlim(time_init, time_fin)
        plt.xlabel('Time (Myr)')
        plt.ylabel('PDP Cross Correlation')
        plt.title(f'{experiment_type}')
        # plt.legend(loc='best')
        plt.show()


plot_experiment_results(existing_df, labels, colors, times)
plot_experiment_results(existing_df_complex, labels, colors, times)

# #%% Plot all together

# # Define labels, colors, and lighter colors for plotting
# labels = ['ks_1', 'ks_2', 'ks_3', 'R2_1', 'R2_2', 'R2_3']
# label_2 = ['Null & Mass', 'Null & Num', 'Mass & Num', 'Null & Mass', 'Null & Num', 'Mass & Num']
# colors = ['purple', 'orange', 'black', 'purple', 'orange', 'black']
# lighter_colors = ['mediumpurple', 'peachpuff', 'gray', 'mediumpurple', 'peachpuff', 'gray']

# # Group the data by 'KA/KB' for both datasets
# grouped = existing_df.groupby('KA/KB')
# grouped_complex = existing_df_complex.groupby('KA/KB')

# # Store results for each column and dataset
# ka_kb_values = sorted(existing_df['KA/KB'].unique())
# ka_kb_values_complex = sorted(existing_df_complex['KA/KB'].unique())

# # Initialize dictionaries to hold mean, std, max, and min values for each label
# mean_values = {label: [] for label in labels}
# std_values = {label: [] for label in labels}
# max_values = {label: [] for label in labels}
# min_values = {label: [] for label in labels}

# mean_values_complex = {label: [] for label in labels}
# std_values_complex = {label: [] for label in labels}
# max_values_complex = {label: [] for label in labels}
# min_values_complex = {label: [] for label in labels}

# # Calculate statistics for each label in the main dataset
# for experiment_type, group_df in grouped:
#     for label in labels:
#         mean_values[label].append(group_df[f'{label}[0]'].mean())
#         std_values[label].append(group_df[f'{label}[1]'].mean())
#         max_values[label].append(group_df[f'{label}[0]'].max())
#         min_values[label].append(group_df[f'{label}[0]'].min())

# # Calculate statistics for each label in the complex dataset
# for experiment_type, group_df_complex in grouped_complex:
#     for label in labels:
#         mean_values_complex[label].append(group_df_complex[f'{label}[0]'].mean())
#         std_values_complex[label].append(group_df_complex[f'{label}[1]'].mean())
#         max_values_complex[label].append(group_df_complex[f'{label}[0]'].max())
#         min_values_complex[label].append(group_df_complex[f'{label}[0]'].min())

# # Plotting results separately for Simple and Complex
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows for separation
# plt.suptitle('Experiment: Static Contact', fontsize=14)

# # Simple Dataset - KS Test
# axes[0, 0].set_ylim([0, 1])
# for i, label in enumerate(labels[:3]):
#     axes[0, 0].errorbar(ka_kb_values, mean_values[label], yerr=std_values[label], 
#                         marker='o', linestyle='', color=lighter_colors[i], capsize=5, label=f'Simple {label_2[i]} ± STD')
#     axes[0, 0].fill_between(ka_kb_values, min_values[label], max_values[label], 
#                             color=lighter_colors[i], alpha=0.3)
# axes[0, 0].set_xlabel('KA/KB')
# axes[0, 0].set_ylabel('KS Test Statistic')
# axes[0, 0].set_title('Simple KS Test Statistics')
# axes[0, 0].legend()

# # Complex Dataset - KS Test
# axes[0, 1].set_ylim([0, 1])
# for i, label in enumerate(labels[:3]):
#     axes[0, 1].errorbar(ka_kb_values_complex, mean_values_complex[label], yerr=std_values_complex[label], 
#                         marker='o', linestyle='', color=colors[i], capsize=5, label=f'Complex {label_2[i]} ± STD')
#     axes[0, 1].fill_between(ka_kb_values_complex, min_values_complex[label], max_values_complex[label], 
#                             color=colors[i], alpha=0.3)
# axes[0, 1].set_xlabel('KA/KB')
# axes[0, 1].set_ylabel('KS Test Statistic')
# axes[0, 1].set_title('Complex KS Test Statistics')
# axes[0, 1].legend()

# # Simple Dataset - PDP Cross Correlation
# axes[1, 0].set_ylim([0, 1])
# for i, label in enumerate(labels[3:]):
#     axes[1, 0].errorbar(ka_kb_values, mean_values[label], yerr=std_values[label], 
#                         marker='o', linestyle='', color=lighter_colors[i+3], capsize=5, label=f'Simple {label_2[i+3]} ± STD')
#     axes[1, 0].fill_between(ka_kb_values, min_values[label], max_values[label], 
#                             color=lighter_colors[i+3], alpha=0.3)
# axes[1, 0].set_xlabel('KA/KB')
# axes[1, 0].set_ylabel('PDP Cross Correlation')
# axes[1, 0].set_title('Simple PDP Cross Correlation')
# axes[1, 0].legend()

# # Complex Dataset - PDP Cross Correlation
# axes[1, 1].set_ylim([0, 1])
# for i, label in enumerate(labels[3:]):
#     axes[1, 1].errorbar(ka_kb_values_complex, mean_values_complex[label], yerr=std_values_complex[label], 
#                         marker='o', linestyle='', color=colors[i+3], capsize=5, label=f'Complex {label_2[i+3]} ± 1 STD')
#     axes[1, 1].fill_between(ka_kb_values_complex, min_values_complex[label], max_values_complex[label], 
#                             color=colors[i+3], alpha=0.3)
# axes[1, 1].set_xlabel('KA/KB')
# axes[1, 1].set_ylabel('PDP Cross Correlation')
# axes[1, 1].set_title('Complex PDP Cross Correlation')
# axes[1, 1].legend(loc='lower left')

# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

# #%% Equally spaced values in x
# # Define labels, colors, and lighter colors for plotting
# labels = ['ks_1', 'ks_2', 'ks_3', 'R2_1', 'R2_2', 'R2_3']
# label_2 = ['Null & Mass', 'Null & Num', 'Mass & Num', 'Null & Mass', 'Null & Num', 'Mass & Num']
# colors = ['purple', 'orange', 'black', 'purple', 'orange', 'black']
# lighter_colors = ['mediumpurple', 'peachpuff', 'gray', 'mediumpurple', 'peachpuff', 'gray']

# # Group the data by 'KA/KB' for both datasets
# grouped = existing_df.groupby('KA/KB')
# grouped_complex = existing_df_complex.groupby('KA/KB')

# # Get sorted unique KA/KB values
# ka_kb_values = sorted(existing_df['KA/KB'].unique())
# ka_kb_values_complex = sorted(existing_df_complex['KA/KB'].unique())

# # Create evenly spaced indices
# ka_kb_indices = range(len(ka_kb_values))  
# ka_kb_indices_complex = range(len(ka_kb_values_complex))

# # Initialize dictionaries to hold statistics
# mean_values = {label: [] for label in labels}
# std_values = {label: [] for label in labels}
# max_values = {label: [] for label in labels}
# min_values = {label: [] for label in labels}

# mean_values_complex = {label: [] for label in labels}
# std_values_complex = {label: [] for label in labels}
# max_values_complex = {label: [] for label in labels}
# min_values_complex = {label: [] for label in labels}

# # Calculate statistics for the main dataset
# for _, group_df in grouped:
#     for label in labels:
#         mean_values[label].append(group_df[f'{label}[0]'].mean())
#         std_values[label].append(group_df[f'{label}[1]'].mean())
#         max_values[label].append(group_df[f'{label}[0]'].max())
#         min_values[label].append(group_df[f'{label}[0]'].min())

# # Calculate statistics for the complex dataset
# for _, group_df_complex in grouped_complex:
#     for label in labels:
#         mean_values_complex[label].append(group_df_complex[f'{label}[0]'].mean())
#         std_values_complex[label].append(group_df_complex[f'{label}[1]'].mean())
#         max_values_complex[label].append(group_df_complex[f'{label}[0]'].max())
#         min_values_complex[label].append(group_df_complex[f'{label}[0]'].min())

# # Plotting results separately for Simple and Complex
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# plt.suptitle('Experiment: Static Contact', fontsize=14)

# # Simple Dataset - KS Test
# axes[0, 0].set_ylim([0, 1])
# for i, label in enumerate(labels[:3]):
#     axes[0, 0].errorbar(ka_kb_indices, mean_values[label], yerr=std_values[label], 
#                         marker='o', linestyle='', color=lighter_colors[i], capsize=5, label=f'Simple {label_2[i]} ± STD')
#     axes[0, 0].fill_between(ka_kb_indices, min_values[label], max_values[label], 
#                             color=lighter_colors[i], alpha=0.3)
# axes[0, 0].set_xlabel('KA/KB')
# axes[0, 0].set_ylabel('KS Test Statistic')
# axes[0, 0].set_title('Simple KS Test Statistics')
# axes[0, 0].legend()
# axes[0, 0].set_xticks(ka_kb_indices)
# axes[0, 0].set_xticklabels(ka_kb_values)

# # Complex Dataset - KS Test
# axes[0, 1].set_ylim([0, 1])
# for i, label in enumerate(labels[:3]):
#     axes[0, 1].errorbar(ka_kb_indices_complex, mean_values_complex[label], yerr=std_values_complex[label], 
#                         marker='o', linestyle='', color=colors[i], capsize=5, label=f'Complex {label_2[i]} ± STD')
#     axes[0, 1].fill_between(ka_kb_indices_complex, min_values_complex[label], max_values_complex[label], 
#                             color=colors[i], alpha=0.3)
# axes[0, 1].set_xlabel('KA/KB')
# axes[0, 1].set_ylabel('KS Test Statistic')
# axes[0, 1].set_title('Complex KS Test Statistics')
# axes[0, 1].legend()
# axes[0, 1].set_xticks(ka_kb_indices_complex)
# axes[0, 1].set_xticklabels(ka_kb_values_complex)

# # Simple Dataset - PDP Cross Correlation
# axes[1, 0].set_ylim([0, 1])
# for i, label in enumerate(labels[3:]):
#     axes[1, 0].errorbar(ka_kb_indices, mean_values[label], yerr=std_values[label], 
#                         marker='o', linestyle='', color=lighter_colors[i+3], capsize=5, label=f'Simple {label_2[i+3]} ± STD')
#     axes[1, 0].fill_between(ka_kb_indices, min_values[label], max_values[label], 
#                             color=lighter_colors[i+3], alpha=0.3)
# axes[1, 0].set_xlabel('KA/KB')
# axes[1, 0].set_ylabel('PDP Cross Correlation')
# axes[1, 0].set_title('Simple PDP Cross Correlation')
# axes[1, 0].legend()
# axes[1, 0].set_xticks(ka_kb_indices)
# axes[1, 0].set_xticklabels(ka_kb_values)

# # Complex Dataset - PDP Cross Correlation
# axes[1, 1].set_ylim([0, 1])
# for i, label in enumerate(labels[3:]):
#     axes[1, 1].errorbar(ka_kb_indices_complex, mean_values_complex[label], yerr=std_values_complex[label], 
#                         marker='o', linestyle='', color=colors[i+3], capsize=5, label=f'Complex {label_2[i+3]} ± 1 STD')
#     axes[1, 1].fill_between(ka_kb_indices_complex, min_values_complex[label], max_values_complex[label], 
#                             color=colors[i+3], alpha=0.3)
# axes[1, 1].set_xlabel('KA/KB')
# axes[1, 1].set_ylabel('PDP Cross Correlation')
# axes[1, 1].set_title('Complex PDP Cross Correlation')
# axes[1, 1].legend(loc='lower left')
# axes[1, 1].set_xticks(ka_kb_indices_complex)
# axes[1, 1].set_xticklabels(ka_kb_values_complex)

# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

#%% Equally spaced values in x
# Define labels, colors, and lighter colors for plotting
labels = ['ks_1', 'ks_2', 'ks_3', 'R2_1', 'R2_2', 'R2_3']
label_2 = ['Null & Mass', 'Null & Num', 'Mass & Num', 'Null & Mass', 'Null & Num', 'Mass & Num']
colors = ['purple', 'orange', 'black', 'purple', 'orange', 'black']
lighter_colors = ['mediumpurple', 'peachpuff', 'gray', 'mediumpurple', 'peachpuff', 'gray']

# Group the data by 'KA/KB' for both datasets
grouped = existing_df.groupby('KA/KB')
grouped_complex = existing_df_complex.groupby('KA/KB')

# Get sorted unique KA/KB values
ka_kb_values = sorted(existing_df['KA/KB'].unique())
ka_kb_values_complex = sorted(existing_df_complex['KA/KB'].unique())

# Create evenly spaced indices
ka_kb_indices = range(len(ka_kb_values))  
ka_kb_indices_complex = range(len(ka_kb_values_complex))

# Initialize dictionaries to hold statistics
mean_values = {label: [] for label in labels}
std_values = {label: [] for label in labels}
max_values = {label: [] for label in labels}
min_values = {label: [] for label in labels}

mean_values_complex = {label: [] for label in labels}
std_values_complex = {label: [] for label in labels}
max_values_complex = {label: [] for label in labels}
min_values_complex = {label: [] for label in labels}

# Calculate statistics for the main dataset
for _, group_df in grouped:
    for label in labels:
        val = group_df[f'{label}[0]']
        # Invert KS values
        if 'ks' in label:
            val = 1 - val
        mean_values[label].append(val.mean())
        std_values[label].append(group_df[f'{label}[1]'].mean())
        max_values[label].append(val.max())
        min_values[label].append(val.min())

# Calculate statistics for the complex dataset
for _, group_df_complex in grouped_complex:
    for label in labels:
        val = group_df_complex[f'{label}[0]']
        # Invert KS values
        if 'ks' in label:
            val = 1 - val
        mean_values_complex[label].append(val.mean())
        std_values_complex[label].append(group_df_complex[f'{label}[1]'].mean())
        max_values_complex[label].append(val.max())
        min_values_complex[label].append(val.min())

# === Plotting results ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle('Experiment: Static Contact', fontsize=14)

# --- Simple Dataset - Inverted KS Similarity ---
axes[0, 0].set_ylim([0, 1])
for i, label in enumerate(labels[:3]):
    axes[0, 0].errorbar(ka_kb_indices, mean_values[label], yerr=std_values[label],
                        marker='o', linestyle='', color=lighter_colors[i], capsize=5,
                        label=f'Simple {label_2[i]} ± STD')
    axes[0, 0].fill_between(ka_kb_indices, min_values[label], max_values[label],
                            color=lighter_colors[i], alpha=0.3)
axes[0, 0].set_xlabel('KA/KB')
axes[0, 0].set_ylabel('1 – KS Test Statistic (Similarity)')
axes[0, 0].set_title('Simple KS Similarity')
# axes[0, 0].legend()
axes[0, 0].set_xticks(ka_kb_indices)
axes[0, 0].set_xticklabels(ka_kb_values)

# --- Complex Dataset - Inverted KS Similarity ---
axes[0, 1].set_ylim([0, 1])
for i, label in enumerate(labels[:3]):
    axes[0, 1].errorbar(ka_kb_indices_complex, mean_values_complex[label], yerr=std_values_complex[label],
                        marker='o', linestyle='', color=colors[i], capsize=5,
                        label=f'Complex {label_2[i]} ± STD')
    axes[0, 1].fill_between(ka_kb_indices_complex, min_values_complex[label], max_values_complex[label],
                            color=colors[i], alpha=0.3)
axes[0, 1].set_xlabel('KA/KB')
axes[0, 1].set_ylabel('1 – KS Test Statistic (Similarity)')
axes[0, 1].set_title('Complex KS Similarity')
# axes[0, 1].legend()
axes[0, 1].set_xticks(ka_kb_indices_complex)
axes[0, 1].set_xticklabels(ka_kb_values_complex)

# --- Simple Dataset - PDP Cross Correlation ---
axes[1, 0].set_ylim([0, 1])
for i, label in enumerate(labels[3:]):
    axes[1, 0].errorbar(ka_kb_indices, mean_values[label], yerr=std_values[label],
                        marker='o', linestyle='', color=lighter_colors[i+3], capsize=5,
                        label=f'Simple {label_2[i+3]} ± STD')
    axes[1, 0].fill_between(ka_kb_indices, min_values[label], max_values[label],
                            color=lighter_colors[i+3], alpha=0.3)
axes[1, 0].set_xlabel('KA/KB')
axes[1, 0].set_ylabel('PDP Cross Correlation')
axes[1, 0].set_title('Simple PDP Cross Correlation')
# axes[1, 0].legend()
axes[1, 0].set_xticks(ka_kb_indices)
axes[1, 0].set_xticklabels(ka_kb_values)

# --- Complex Dataset - PDP Cross Correlation ---
axes[1, 1].set_ylim([0, 1])
for i, label in enumerate(labels[3:]):
    axes[1, 1].errorbar(ka_kb_indices_complex, mean_values_complex[label], yerr=std_values_complex[label],
                        marker='o', linestyle='', color=colors[i+3], capsize=5,
                        label=f'Complex {label_2[i+3]} ± 1 STD')
    axes[1, 1].fill_between(ka_kb_indices_complex, min_values_complex[label], max_values_complex[label],
                            color=colors[i+3], alpha=0.3)
axes[1, 1].set_xlabel('KA/KB')
axes[1, 1].set_ylabel('PDP Cross Correlation')
axes[1, 1].set_title('Complex PDP Cross Correlation')
# axes[1, 1].legend(loc='lower left')
axes[1, 1].set_xticks(ka_kb_indices_complex)
axes[1, 1].set_xticklabels(ka_kb_values_complex)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


