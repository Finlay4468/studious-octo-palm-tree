# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:05:59 2025

@author: FinlaySinclair
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_portfolio = 1000000
annual_contribution = 20000
years_until_retirement = 10
years_in_retirement = 30
total_years = years_until_retirement + years_in_retirement
mean_return = 0.07
std_dev_return = 0.15
num_simulations = 10000
degrees_of_freedom=5
initial_spending_rate=0.04
minimum_spending=30000
maximum_spending=60000
inflation_rate=0.025



# Simulation
portfolio_paths = np.zeros((num_simulations, total_years))
portfolio_paths[:,0] = initial_portfolio

for i in range(num_simulations):
    current_spending = initial_portfolio * initial_spending_rate
    inflated_min_spend = minimum_spending
    t_dist_returns = np.random.standard_t(df=degrees_of_freedom, size=total_years)
    scaled_returns = t_dist_returns * std_dev_return
    random_returns = scaled_returns + mean_return
    for year in range(1, total_years):
        inflated_min_spend = inflated_min_spend * (1 + inflation_rate)
        if year < years_until_retirement:
            portfolio_paths[i, year] = portfolio_paths[i, year-1] * (1 + random_returns[year]) + annual_contribution
        else:
            desired_spending = portfolio_paths[i,year-1] * initial_spending_rate
            current_spending = max(inflated_min_spend, desired_spending)
            if maximum_spending:
                current_spending = min(current_spending, maximum_spending)
                
            portfolio_paths[i, year] = portfolio_paths[i, year-1] * (1 + random_returns[year]) - current_spending
            if portfolio_paths[i, year] < 0:
                portfolio_paths[i, year] = 0

# Analysis
final_values = portfolio_paths[:, -1]
success_probability = np.mean(final_values >  0) * 100
print(f"Probability of success: {success_probability}%")

# Visualisation
plt.figure(figsize=(10,6))
for i in range(100):
    plt.plot(portfolio_paths[i], color="blue", alpha=0.1)
plt.axvline(x=years_until_retirement, color='red', linestyle='--', label='Retirement Starts')
plt.title("Monte Carlo Simulation of Retirement Portfolio")
plt.xlabel("Years")
plt.ylabel("Portfolio Value (£)")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,6))
plt.hist(final_values[final_values > 0], bins=50, edgecolor='black', alpha=0.7, color='green')
plt.title("Distribution of Final Portfolio Values (Successes)")
plt.xlabel("Final Portfolio Value (£)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


