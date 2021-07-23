import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
import matplotlib.pyplot as plt

class generic_price:
    def __init__(self):
        pass
    
    def price_params(self, data, model_type):
        if model_type == 'gbm': # generalised brownian motion
            plt.plot(data)
            plt.xlabel('Time steps (weekly)')
            plt.ylabel('S')
            plt.title('PC1 plot')
            plt.show()

            plt.bar(np.array(range(len(data))), np.array(pd.Series(data).pct_change()))
            plt.xlabel('Time steps (weekly)')
            plt.ylabel('dS / $S_{t}$')
            plt.title('Price Delta')
            plt.show()

            price_change = np.array(pd.Series(data).pct_change())

            return [price_change[~np.isnan(price_change)].mean(), price_change[~np.isnan(price_change)].std()]
        
        else:
            pass

    def  price_simulation(self, mean, volatility, eigenvectors, sample_mean, model_type):
        if model_type == 'gbm': # generalised brownian motion
            # model parameters
            # So    :   initial level
            So = 0.01
            # dt    :   time increment
            dt = 1
            # N     :   number of time points in the prediction time horizon
            N = 25
            # t     :   array for time points in the prediction time horizon  [1, 2, 3, ..., N]
            t = np.arange(N) + 1
            # mu    :   mean of historical data
            mu = mean
            # sigma :   standard deviation of historical data
            sigma = volatility
            
            prices_simulation_results = []

            electricity_prices_sim = []
            natural_gas_prices_sim = []
            co2_prices_sim = [] 

            # PC1 -> prices (backtransform) + MC simulation
            for i in range(100):
                # W     :   array for Brownian path
                W = np.random.normal(loc=mu, scale=sigma, size=N)
                # dW     :   array for Brownian increments
                dW = np.concatenate((np.array([W[0]]), np.diff(W)))
                # dS = mu*dt + sigma*dW
                pc1_path = np.concatenate((np.array([0]), np.multiply(dW, t) + So))

                # backtransform to get ELEC, NG, CO2 prices
                pc1_path_backtransform = pc1_path.reshape(26,1).dot(eigenvectors[0].reshape(1,3)) + sample_mean

                # correction for negative values
                pc1_path_backtransform[pc1_path_backtransform < 0] = 0

                electricity_prices_sim.append(list(pc1_path_backtransform[:, 0]))
                natural_gas_prices_sim.append(list(pc1_path_backtransform[:, 1]))
                co2_prices_sim.append(list(pc1_path_backtransform[:, 2]))
                

                # pack the simulation results in a list of dataframes
                pc1_path_backtransform_zero_padding = np.hstack((pc1_path_backtransform, np.zeros(len(pc1_path_backtransform)).reshape(len(pc1_path_backtransform), 1)))
                prices_simulation_results.append(pd.DataFrame(pc1_path_backtransform_zero_padding, columns=['ELEC', 'NG', 'CO2', 'DUMMY']))
                
            np.savetxt('elec_sim.csv', np.array(electricity_prices_sim).T, delimiter=',')
            np.savetxt('ng_sim.csv', np.array(natural_gas_prices_sim).T, delimiter=',')
            np.savetxt('co2_sim.csv', np.array(co2_prices_sim).T, delimiter=',')
            
            return prices_simulation_results
            
        else:
            pass