import numpy as np
import itertools
import matplotlib.pyplot as plt

def plot_time_series(data_frame):
    # plots of the original time series using matplotlib
    plot_cols = data_frame.columns.to_list()[0:-1]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    data_frame[plot_cols].plot(subplots=True, ax=axes)

    for ax, col in zip(axes, plot_cols):
    # ax.axhline(0, color='k', linestyle='-', linewidth=1)
        ax.set_title(f'{col} prices')
        ax.set_ylabel('Unit Price ($/MWh)')
        ax.set_xlabel('Date')
        ax.legend(loc='upper left', fontsize=11, frameon=False)
    
    plt.tight_layout()
    plt.savefig('destination_path.eps', format='eps', dpi=1200)
    plt.show()

def get_basic_uv_stats(data):
    sample_size = len(data)
    sample_mean = data.mean()
    sample_std = data.std()
    return [sample_size, sample_mean, sample_std]

def interval_estimate_1p(data_sets, statistic):
    # initialise the plotting grid for one-population interval estimations
    fig, axs = plt.subplots(1, 3, tight_layout=True)
    vcolours = ['blue', 'green', 'red']

    # iterate through the datasets
    for key in data_sets.keys():
        # iterate through the columns for each dataset
        for col in data_sets[key].columns[0:-1]:
            # get the necessary sample statistics for interval estimation
            stats = get_basic_uv_stats(data_sets[key][col])

            if statistic == 'mean':
                fig.suptitle(f'95% confidence intervals for the population mean')
                
                # estimate population mean interval for 95% confidence
                z_alpha_by_2 = 1.96 # from z table
                lb = stats[1] - (stats[2]/np.sqrt(stats[0])) * z_alpha_by_2
                ub = stats[1] + (stats[2]/np.sqrt(stats[0])) * z_alpha_by_2
                ci_centre = (lb + ub) / 2

                # print 95% CI estimate to terminal
                print(f'\n*** For the case: {key}, {col} ***')
                print(f'95% confidence interval for the population mean: [{lb}, {ub}]')

            elif statistic == 'variance':
                fig.suptitle('90% confidence intervals for the population standard deviation')
                                    
                # estimate population variance interval for 90% confidence
                chi_sq_1_minus_alpha_by_2 =  17.71 # from chi-square table for dof = 29
                chi_sq_alpha_by_2 = 42.56 # from chi-square table for dof = 29
                lb = np.sqrt(29 * stats[2]**2 / chi_sq_alpha_by_2)
                ub = np.sqrt(29 * stats[2]**2 / chi_sq_1_minus_alpha_by_2)
                ci_centre = (lb + ub) / 2

                # print 90% CI estimate to terminal
                print(f'\n*** For the case: {key}, {col} ***')
                print(f'90% confidence interval for the population standard deviation: [{lb}, {ub}]')
            
            else:
                return 0

            # index for dataset    
            irow = list(data_sets.keys()).index(key)
            # index for dataset columns/variables            
            icol = data_sets[key].columns[0:-1].get_loc(col)
            
            # confidence interval plot 
            axs[icol].errorbar(irow + 1, ci_centre, yerr=(ub - lb), color = vcolours[icol], capsize=5, capthick=3)
            axs[icol].set_title(f'{data_sets[key].columns[0:-1][icol]} Price')
            axs[icol].set_xlabel('Period')
            axs[icol].set_xticks([1, 2, 3, 4])

    plt.show()

def interval_estimate_2p(data_sets, statistic):
    # initialise the plotting grid for two-population interval estimations
    fig, axs = plt.subplots(3, 1, tight_layout=True)
    vcolours = ['blue', 'green', 'red']

    combs_list = list(itertools.combinations(data_sets.keys(), 2))
    data_cols = data_sets['Period 1'].columns[0:-1]

    for col in data_cols:
        for comb in combs_list:
            # get the necessary sample statistics for two-population interval estimation, in time
            pop1_stats = get_basic_uv_stats(data_sets[comb[0]][col])
            pop2_stats = get_basic_uv_stats(data_sets[comb[1]][col])
            # assuming the variances are similar over time, compute pooled sample variance
            s_pooled_squared = ((pop1_stats[0] - 1) * pop1_stats[2]**2 + (pop2_stats[0] - 1) * pop2_stats[2]**2) / (pop1_stats[0] + pop2_stats[0] - 2)

            if statistic == 'difference of means':
                fig.suptitle(f'95% confidence intervals for the "difference of means"')

                # estimate 'difference of means' interval for 95% confidence
                z_alpha_by_2 = 1.96 # from z table
                lb = (pop2_stats[1] - pop1_stats[1]) -  z_alpha_by_2 * np.sqrt((s_pooled_squared / pop1_stats[0]) + (s_pooled_squared / pop2_stats[0]))
                ub = (pop2_stats[1] - pop1_stats[1]) +  z_alpha_by_2 * np.sqrt((s_pooled_squared / pop1_stats[0]) + (s_pooled_squared / pop2_stats[0]))
                ci_centre = (lb + ub) / 2

                # print 95% CI estimate to terminal
                print(f'\n*** For the case: {col}, {comb} ***')
                print(f'95% confidence interval for the "difference of means": [{lb}, {ub}]')

            elif statistic == 'ratio of variances':
                fig.suptitle('90% confidence for the "ratio of variances"')
                                    
                # estimate 'ratio of variances' with 90% confidence
                F_alpha_by_2 = 	1.62 # from F table for dof = 30, 29
                ci_centre = (pop1_stats[2]**2 / pop2_stats[2]**2) / F_alpha_by_2
                lb = ub = 0

                # print 90% CI estimate to terminal
                print(f'\n*** For the case: {col}, {comb} ***')
                print(f'90% confidence for the "ratio of variances": {ci_centre}')

            else:
                return 0
            
            # index for dataset columns/variables            
            irow = data_sets['Period 1'].columns[0:-1].get_loc(col)
            data_cols = data_sets['Period 1'].columns[0:-1]
            # index for combinations    
            icol = combs_list.index(comb)

            # confidence interval plot
            axs[irow].errorbar(icol + 1, ci_centre, yerr=(ub - lb), color = vcolours[irow], capsize=5, capthick=3)
            axs[irow].set_title(f'{data_cols[irow]} Price')
            axs[irow].set_xlabel('Period-on-Period')
            axs[irow].set_xticklabels(['', 'P2 ~ P1', 'P3 ~ P1', 'P4 ~ P1', 'P3 ~ P2', 'P4 ~ P2', 'P4 ~ P3'])
    
    plt.show()

def hypothesis_test():
    pass

def anova():
    pass