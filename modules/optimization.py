#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:40:50 2024

@author: anahitamolavi
"""


import logging
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from typing import Dict, Any, Union


#--------------------Deterministic Model-------------------#

def summarize_results_deterministic(results: Any, instance: Any, opt_data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """
    Summarize the optimization results into a DataFrame.

    Parameters:
    - results = Pyomo results object.
    - instance: Pyomo instance object.
    - opt_data: pd.DataFrame

    Returns:
    - results_df: Pandas DataFrame containing optimal values and objective value.
    - profit: Objective function value
    """
    
    results_dict = {}
    
    if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:   
        # Extract variable values
        for product in opt_data['Product_ID'].to_numpy().tolist():
            results_dict[product] = {
                'Surplus': pyo.value(instance.surplus[product])
            }
        
        
        profit = pyo.value(instance.obj)
        
        # Create DataFrame
        results_df = pd.DataFrame(results_dict).T
    
        return results_df, profit
    else:
        logging.warning("Solver terminated with status: %s", results.solver.status)
        return None, 0

def run_deterministic_optimization(opt_data: pd.DataFrame, macroTargetPercentage: float, solver_name: str = 'glpk') -> pd.DataFrame:
    """
    Run the surplus unit allocation optimization.

    Parameters:
    - opt_data: DataFrame containing optimization input data.
    - macroTargetPercentage: Float, macro target percentage for surplus allocation.
    - solver_name: String, name of the solver to use (default: 'glpk').

    Returns:
    - results_df: Pandas DataFrame containing optimal values and objective value.
    """
    OptModel = pyo.AbstractModel()
    
    # Define Indices
    OptModel.demand_group = pyo.Set()
    OptModel.products = pyo.Set()
    OptModel.scenarios = pyo.Set()
    
    
    #Define Variables
    OptModel.surplus = pyo.Var(OptModel.products, within = pyo.NonNegativeIntegers, bounds = (0, 100000))
    
    #Define Parameters
    OptModel.target_demand = pyo.Param(OptModel.products, within = pyo.NonNegativeReals)
    OptModel.margin = pyo.Param(OptModel.products, within = pyo.Reals)
    OptModel.COGS = pyo.Param(OptModel.products, within = pyo.Reals)
    OptModel.capacity = pyo.Param(OptModel.products, within = pyo.NonNegativeReals, default = float('inf'))
    OptModel.macroTargetPercentage = pyo.Param(within = pyo.NonNegativeReals)
    
    
    #Initialize Parameters
    products_list = opt_data['Product_ID'].to_numpy().tolist()
    
    target_demand_dict = pd.Series(opt_data.Demand.astype(float).values, index = [opt_data.Product_ID.values]).to_dict() 
    margin_dict = pd.Series(opt_data.Margin.astype(float).values, index = [opt_data.Product_ID.values]).to_dict() 
    capacity_dict = pd.Series(opt_data.Capacity.astype(float).values, index = [opt_data.Product_ID.values]).to_dict() 
    
    data = {
        
        None: {
            'products': {None: products_list},
            'target_demand': target_demand_dict,
            'margin': margin_dict,
            'capacity': capacity_dict,
            'macroTargetPercentage': {None: macroTargetPercentage},
            
            }
        }
    
    
    # Define constraints
    def capacity_constraint_rule(OptModel, i):
        return OptModel.surplus[i] <= OptModel.capacity[i] * OptModel.target_demand[i]  
    OptModel.CapacityConstraint = pyo.Constraint(OptModel.products, rule = capacity_constraint_rule)
    
    def total_surplus_constraint_rule(OptModel):
        return sum(OptModel.surplus[i] for i in OptModel.products) <= (1 + OptModel.macroTargetPercentage) * sum(OptModel.target_demand[i] for i in OptModel.products)   
    OptModel.TotalSurplusConstraint = pyo.Constraint(rule = total_surplus_constraint_rule)
    
    # Define objective function
    def objective_rule(OptModel):
        return sum(OptModel.margin[i] * (OptModel.target_demand[i] + OptModel.surplus[i]) for i in OptModel.products)
    OptModel.obj = pyo.Objective(rule = objective_rule, sense = pyo.maximize)
          
    # Create an instance of the optimization
    instance = OptModel.create_instance(data)
    logging.info('Solve Started.')
    
    # Specify the solver
    solver = SolverFactory(solver_name)

    # Solve the model
    results = solver.solve(instance, tee = True)
    
    results_df, profit = summarize_results_deterministic(results, instance, opt_data)
    
    return results_df, profit if results_df is not None else pd.DataFrame(), 0  # Ensure non-empty DataFrame is returned



#--------------------Stochastic Model-------------------#


def summarize_results_stochastic(results: Any, instance: Any, opt_data: pd.DataFrame, joint_scenarios_df: pd.DataFrame, target_demand_df: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """
    Summarize the stochastic optimization results into a DataFrame.

    Parameters:
    - results: Pyomo results object.
    - instance: Pyomo instance object.
    - opt_data: DataFrame containing optimization input data.
    - joint_scenarios_df: DataFrame containing scenarios.
    - target_demand_df: DataFrame containing target demands.

    Returns:
    - results_df: Pandas DataFrame containing optimal values and objective value.
    - profit: Objective function value.
    """
    
    results_dict = {}
    
    if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:   
        # Extract variable values
        for product in opt_data['Product_ID'].to_numpy().tolist():
            for scenario in joint_scenarios_df['Scenario_ID'].to_numpy().tolist():
                results_dict[(scenario, product)] = {
                    'Surplus': pyo.value(instance.surplus[product, scenario])
                }
        
        # Extract objective value
        profit = pyo.value(instance.obj)
        
        # Create DataFrame
        results_df = pd.DataFrame(results_dict).T
        results_df.index.names = ['Scenario_ID', 'Product_ID']
        results_df = results_df.rename(columns={0: 'Surplus'})
    
        return results_df, profit
    else:
        logging.warning("Solver terminated with status: %s", results.solver.status)
        return None, 0

    
    
def run_stochastic_optimization(opt_data: pd.DataFrame, joint_scenarios_df: pd.DataFrame, target_demand_df: pd.DataFrame, macroTargetPercentage: float, solver_name: str = 'glpk') -> pd.DataFrame:
    """
    Run the surplus unit allocation optimization.

    Parameters:
    - opt_data: DataFrame containing optimization input data.
    - joint_scenarios_df: DataFrame containing scenarios
    - macroTargetPercentage: Float, macro target percentage for surplus allocation.
    - solver_name: String, name of the solver to use (default: 'glpk').

    Returns:
    - results_df: Pandas DataFrame containing optimal values and objective value.
    """
    OptModel = pyo.AbstractModel()
    
    # Define Indices
    OptModel.demand_group = pyo.Set()
    OptModel.products = pyo.Set()
    OptModel.scenarios = pyo.Set()
    
    
    #Define Variables
    OptModel.surplus = pyo.Var(OptModel.products, OptModel.scenarios, within = pyo.NonNegativeReals, bounds = (0, 100000))
    
    #Define Parameters
    OptModel.target_demand = pyo.Param(OptModel.products, OptModel.scenarios, within = pyo.NonNegativeReals, initialize = 0)
    OptModel.margin = pyo.Param(OptModel.products, within = pyo.Reals)
    OptModel.COGS = pyo.Param(OptModel.products, within = pyo.Reals)
    OptModel.capacity = pyo.Param(OptModel.products, within = pyo.NonNegativeReals, default = float('inf'))
    OptModel.macroTargetPercentage = pyo.Param(within = pyo.NonNegativeReals)
    
    OptModel.probabilities = pyo.Param(OptModel.scenarios)
    OptModel.predictedDemand = pyo.Param(OptModel.products, within = pyo.NonNegativeReals, initialize = 0)
    
    #Initialize Parameters
    products_list = opt_data['Product_ID'].to_numpy().tolist()
    scenarios_list = joint_scenarios_df['Scenario_ID'].to_numpy().tolist()
    
    predictedDemand_dict = pd.Series(opt_data.Demand.astype(float).values, index = [opt_data.Product_ID.values]).to_dict()
    target_demand_dict = pd.Series(target_demand_df.target_demand.astype(float).values, index = [target_demand_df.Product_ID.values, target_demand_df.Scenario_ID.values]).to_dict() 
    margin_dict = pd.Series(opt_data.Margin.astype(float).values, index = [opt_data.Product_ID.values]).to_dict() 
    capacity_dict = pd.Series(opt_data.Capacity.astype(float).values, index = [opt_data.Product_ID.values]).to_dict()
    probability_dict = pd.Series(joint_scenarios_df.joint_probability.astype(float).values, index = [joint_scenarios_df.Scenario_ID.values]).to_dict()
    
    data = {
        
        None: {
            'products': {None: products_list},
            'scenarios': {None: scenarios_list},
            'predictedDemand': predictedDemand_dict,
            'target_demand': target_demand_dict,
            'margin': margin_dict,
            'capacity': capacity_dict,
            'macroTargetPercentage': {None: macroTargetPercentage},
            'probabilities': probability_dict
            
            }
        }
    
    
    # Define constraints
    def capacity_constraint_rule(OptModel, i, w):
        return OptModel.surplus[i,w] <= OptModel.capacity[i] * OptModel.target_demand[i,w] 
    OptModel.CapacityConstraint = pyo.Constraint(OptModel.products, OptModel.scenarios, rule = capacity_constraint_rule)
    
    def total_surplus_constraint_rule(OptModel, w):
        return sum(OptModel.surplus[i,w] for i in OptModel.products) <= (1 + OptModel.macroTargetPercentage) * sum(OptModel.target_demand[i,w] for i in OptModel.products)   
    OptModel.TotalSurplusConstraint = pyo.Constraint(OptModel.scenarios, rule = total_surplus_constraint_rule)
  
    # Define objective function
    def objective_rule(OptModel):
        return sum(OptModel.probabilities[w] * OptModel.margin[i] * OptModel.surplus[i,w] for i in OptModel.products for w in OptModel.scenarios)
    OptModel.obj = pyo.Objective(rule = objective_rule, sense = pyo.maximize)
          
    # Create an instance of the optimization
    instance = OptModel.create_instance(data)
    logging.info('Solve Started.')
    
    # Specify the solver
    solver = SolverFactory(solver_name)

    # Solve the model
    results = solver.solve(instance, tee = True)
    
    results_df, profit = summarize_results_stochastic(results, instance, opt_data, joint_scenarios_df, target_demand_df)
    
    return results_df, profit if results_df is not None else pd.DataFrame(), 0  # Ensure non-empty DataFrame is returned
