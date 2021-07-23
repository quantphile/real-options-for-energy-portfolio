import numpy as np
import pandas as pd
import scipy
import ellipsoid
import uv_stats
import underlyings
import uv_stats
import matplotlib.pyplot as plt

def extract_eigenvectors(data_sets, period):
    data_sample = data_sets[period][['ELEC', 'NG', 'CO2']].dropna().to_numpy()
    ET = ellipsoid.EllipsoidTool()
    U, center, radii, rotation = ET.getMinVolEllipse(data_sample, .01)
    return data_sample, U

def transform_ts(data_sample, eigenvectors):
    data_sample_mean = data_sample.mean(axis=0)
    data_sample = data_sample - np.tile(data_sample_mean, (len(data_sample), 1))
    sample_project = data_sample.dot(eigenvectors.T)
    # uv_stats.plot_time_series(pd.DataFrame(sample_project, columns = ['PC1','PC2','PC3']))
    return sample_project, data_sample_mean

def pc1_model_fitting(pc1):
    price = underlyings.generic_price()
    mean, volatility = price.price_params(pc1, 'gbm')
    print('PC1 Model Parameters')
    print(f'mean level: {mean}')
    print(f'volatility: {volatility}')
    return mean, volatility

def cash_flow_simulation(mean, volatility, eigenvectors, sample_mean, tech_option):
    price = underlyings.generic_price()
    prices_simulation_results = price.price_simulation(mean, volatility, eigenvectors, sample_mean, 'gbm')

    if tech_option == 'OP1':
        # data for cash flow calculations
        total_capex = 824450000.00
        total_fixed_opex = 15405000.00
        partial_variable_opex = 11095979.90 # including gas turbine and water-related costs
        partial_revenues = 72270000.00 # including water-related revenues
        annual_gt_production = 3184698
        gt_plant_capacity_factor = 0.661
        annual_energy_demand = 3000000
    elif tech_option == 'OP2':
        # data for cash flow calculations
        total_capex = 1390035000.00
        total_fixed_opex = 28511500.00
        partial_variable_opex = 12859662.86 # including gas turbine and water-related costs
        partial_revenues = 72270000.00 # including water-related revenues
        annual_gt_production = 1563397.2
        gt_plant_capacity_factor = 0.661
        annual_energy_demand = 3000000
    elif tech_option == 'OP3':
        # data for cash flow calculations
        total_capex = 1816090000.00
        total_fixed_opex = 25314000.00
        partial_variable_opex = 21434587.79 # including gas turbine and water-related costs
        partial_revenues = 72270000.00 # including water-related revenues
        annual_gt_production = 3160888.32
        gt_plant_capacity_factor = 0.5638
        annual_energy_demand = 3000000
    elif tech_option == 'OP4':
        # data for cash flow calculations
        total_capex = 1891275000.00
        total_fixed_opex = 33536500.00
        partial_variable_opex = 18102793.89 # including gas turbine and water-related costs
        partial_revenues = 72270000.00 # including water-related revenues
        annual_gt_production = 1580444.16
        gt_plant_capacity_factor = 0.5638
        annual_energy_demand = 3000000
    elif tech_option == 'OP5':
        # data for cash flow calculations
        total_capex = 2657225000.00
        total_fixed_opex = 58280400.00
        partial_variable_opex = 8873000 # including gas turbine and water-related costs
        partial_revenues = 72270000.00 # including water-related revenues
        annual_gt_production = 0
        gt_plant_capacity_factor = 1
        annual_energy_demand = 3000000
    else:
        pass

    profit_sim = []

    for prices_simulation_result in prices_simulation_results:
        # uv_stats.plot_time_series(prices_simulation_result)
        total_variable_opex = partial_variable_opex + (np.array(prices_simulation_result['NG']) * annual_gt_production / gt_plant_capacity_factor)
        total_opex = total_fixed_opex + total_variable_opex
        total_revenues = partial_revenues + (np.array(prices_simulation_result['ELEC']) * annual_energy_demand)
        co2_tax = ((1000 * annual_gt_production * 181 / 1000000) * np.array(prices_simulation_result['CO2']))
        profit = total_revenues - total_opex - co2_tax
        profit_sim.append(list(profit))

    np.savetxt('profit.csv', np.array(profit_sim).T, delimiter=',')

    # plt.bar(np.array(range(len(profit))), profit)
    # plt.xlabel('Time steps (annually)')
    # plt.ylabel('Profit/Loss')
    # plt.title('Cash Flow Series')
    # plt.show()