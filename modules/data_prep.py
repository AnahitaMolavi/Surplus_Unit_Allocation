# -*- coding: utf-8 -*-
"""
reated on Sat Feb 17 16:40:50 2024

@author: anahitamolavi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pyDOE import lhs
from scipy.stats import qmc
from itertools import product
import logging

logging.basicConfig(level=logging.INFO)

def data_transformation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transpose the input DataFrame and convert it to a proper format. Add a column for product ID.

    Parameters:
    - df_raw: DataFrame, input data

    Returns:
    - demand_df: DataFrame, transformed data
    """
    logging.info("Performing data transformation...")
    demand_df = df_raw.transpose()
    headers = demand_df.iloc[0]
    demand_df = pd.DataFrame(demand_df.values[1:], columns = headers)
    demand_df['Product_ID'] = range(1, len(demand_df.index) + 1)
    return demand_df

def exploratory_data_analysis(demand_df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the input DataFrame.

    Parameters:
    - demand_df: DataFrame, input data

    Returns:
    - None
    """
    logging.info("Performing exploratory data analysis...")
    
    # Check the data types and missing values
    print(demand_df.info())
    
    # Summary statistics for numerical columns
    print(demand_df.describe())
    
    # Distribution of Demand
    plt.figure(figsize=(10, 6))
    sns.histplot(demand_df['Demand'], bins=20, kde=True)
    plt.title('Distribution of Demand')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.show()
    
    # Scatter plot of Demand vs. Margin
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Demand', y='Margin', data=demand_df)
    plt.title('Demand vs. Margin')
    plt.xlabel('Demand')
    plt.ylabel('Margin')
    plt.show()
    
    # Box plot of Variance group
    plt.figure(figsize=(8, 6))
    sns.boxplot(y='Variance group', data=demand_df)
    plt.title('Boxplot of Variance group')
    plt.ylabel('Variance group')
    plt.show()

def demand_dist_visualization(var_group_df: pd.DataFrame) -> None:
    """
    Visualize the demand distribution for different variance groups.

    Parameters:
    - var_group_df: DataFrame, input data with variance groups

    Returns:
    - None
    """
    logging.info("Visualizing demand distribution...")
    
    # Get unique groups
    groups = var_group_df['Demand Var Group'].unique()
    
    # Create a counter for subplot position
    plot_count = 0
    
    # Create a figure and axis
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loop over unique groups
    for group in groups:
        # Get distribution parameters
        dist = stats.burr12
        c = var_group_df[var_group_df['Demand Var Group'] == group]['c'].values[0]
        d = var_group_df[var_group_df['Demand Var Group'] == group]['d'].values[0]
        loc = var_group_df[var_group_df['Demand Var Group'] == group]['loc'].values[0]
        scale = var_group_df[var_group_df['Demand Var Group'] == group]['scale'].values[0]
    
        # Generate x values
        x = np.linspace(dist.ppf(0.01, c, d),
                        dist.ppf(0.99, c, d), 100000)
    
        # Plot on current axis
        ax[plot_count // 3, plot_count % 3].plot(x, dist.pdf(x, c, d),
                                                 'r-', lw=5, alpha=0.6, label='burr12 pdf')
        ax[plot_count // 3, plot_count % 3].set_title("PDF for variance group %s" % (group))
    
        # Update subplot position
        plot_count += 1
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def feature_engineering(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the input DataFrame.

    Parameters:
    - demand_df: DataFrame, input data

    Returns:
    - demand_df: DataFrame, transformed data with new features
    """
    logging.info("Performing feature engineering...")

    # Fill missing values in Capacity with a hypothetical large value
    demand_df['Capacity'].fillna(0.50, inplace = True)
    
    # Create a new feature 'Total_Cost' as the sum of Margin and COGS
    demand_df['Revenue'] = demand_df['Margin'] + demand_df['COGS']

    return demand_df

def generate_individual_scenarios(dist_type: str, dist_params: list, sampling_method: str, num_scenarios: int, random_seed: int = 777) -> np.ndarray:
    """
    Generate scenarios from a specified distribution using the specified sampling method.

    Parameters:
    - dist_type: String, the distribution type (e.g., 'burr12').
    - dist_params: List, distribution parameters (e.g., [c, d, loc, scale]).
    - sampling_method: String, the sampling method ('random', 'lhs', or 'qmc').
    - num_scenarios: Integer, the number of scenarios to generate.
    - random_seed: Integer, random seed for reproducibility (default: 777).

    Returns:
    - individual_scenarios: NumPy array, generated scenarios with their associated probabilities.
    """
    logging.info("Generating individual scenarios...")
    
    # Mapping of distribution names to distribution classes
    dist_mapping = {
        'burr12': stats.burr12,
        'gamma': stats.gamma,
        # Add more distributions as needed
    }

    # Create a distribution object
    dist = dist_mapping[dist_type](*dist_params)

    individual_scenarios = []

    if sampling_method == 'random':
        samples = dist.rvs(size=num_scenarios, random_state=random_seed)
        probabilities = dist.pdf(samples)
    elif sampling_method == 'lhs':
        lhs_samples = lhs(1, samples = num_scenarios, criterion = 'maximin')
        samples = dist.ppf(lhs_samples)
        probabilities = dist.pdf(samples)
    elif sampling_method == 'qmc':
        qmc_gen = qmc.Sobol(1)
        qmc_samples = qmc_gen.random(num_scenarios)
        samples = dist.ppf(qmc_samples)
        probabilities = dist.pdf(samples)
    else:
        raise ValueError("Invalid sampling method. Choose from 'random', 'lhs', or 'qmc'.")
    
    # Normalize probabilities to ensure they sum to 1
    probabilities /= np.sum(probabilities)

    individual_scenarios.append((samples, probabilities))
    
    return individual_scenarios


def generate_joint_scenarios(var_group_df: pd.DataFrame, num_scenarios_per_group: int, random_seed: int, sampling_method: str) -> pd.DataFrame:
    """
    Generate joint scenarios from a DataFrame of variable groups.

    Parameters:
    - var_group_df: DataFrame, containing variable groups with columns: 'Demand Var Group', 'distribution', 'c', 'd', 'loc', 'scale'.
    - num_scenarios_per_group: Integer, number of scenarios per group.
    - random_seed: Integer, random seed for reproducibility.
    - sampling_method: String, the sampling method ('random', 'lhs', or 'qmc').

    Returns:
    - joint_scenarios_df: DataFrame, containing Scenario_ID, demand_var_group columns, and joint_probability.
    """
    logging.info("Generating joint scenarios...")
    
    individual_scenarios_dict = {}

    for group in var_group_df['Demand Var Group'].unique():
        dist_type = var_group_df.loc[var_group_df['Demand Var Group'] == group, 'distribution'].values[0]
        dist_params = [
            var_group_df.loc[var_group_df['Demand Var Group'] == group, 'c'].values[0],
            var_group_df.loc[var_group_df['Demand Var Group'] == group, 'd'].values[0],
            var_group_df.loc[var_group_df['Demand Var Group'] == group, 'loc'].values[0],
            var_group_df.loc[var_group_df['Demand Var Group'] == group, 'scale'].values[0]
        ]

        individual_scenarios = generate_individual_scenarios(
            dist_type, dist_params, sampling_method, num_scenarios_per_group, random_seed
        )
        individual_scenarios_dict[group] = individual_scenarios


    num_groups = len(individual_scenarios_dict)

    # Generate all combinations of scenarios for each group
    scenarios_combinations = product(*[individual_scenarios_dict[group][0][0] for group in individual_scenarios_dict])

    # Generate all combinations of probabilities for each group
    probabilities_combinations = list(product(*[individual_scenarios_dict[group][0][1] for group in individual_scenarios_dict]))

    # Calculate joint probabilities
    joint_probabilities = [np.prod(prob_product) for prob_product in probabilities_combinations]
    
    # Generate Scenario_ID
    #num_scenarios = len(next(iter(individual_scenarios_dict.values()))[0])
    scenario_ids = range(1, num_scenarios_per_group ** num_groups + 1)

    # Initialize DataFrame
    joint_scenarios_df = pd.DataFrame(columns=["Scenario_ID"] + [f"demand_var_group{i}" for i in range(num_groups)] + ["joint_probability"])

    # Populate DataFrame
    for scenario_id, scenario_combination, joint_probability in zip(scenario_ids, scenarios_combinations, joint_probabilities):
        scenario_data = {"Scenario_ID": scenario_id}
        for i, scenario_value in enumerate(scenario_combination):
            scenario_data[f"demand_var_group{i}"] = scenario_value[0]
        scenario_data["joint_probability"] = joint_probability
        joint_scenarios_df = joint_scenarios_df.append(scenario_data, ignore_index=True)

    # Normalize probabilities to ensure they sum to 1
    joint_scenarios_df['joint_probability'] /= joint_scenarios_df['joint_probability'].sum()
    
    return joint_scenarios_df


def reduce_scenarios(joint_scenarios_df, number_of_scenarios):
    reduced_df = pd.DataFrame()
    
    return reduced_df