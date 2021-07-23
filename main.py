import sys
import argparse, textwrap
import data_utils
import uv_stats
import mv_stats
import ts_analysis

def main():
    # initialise argument parser object
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', type=str, nargs='*', \
                        help='csv data source file')
    parser.add_argument('--option', type=int, \
                        help=textwrap.dedent('''\
                                                1: univariate stats
                                                2: multivariate stats
                                                3: time series analysis,
                                                   Monte Carlo simulation &
                                                   real options valuation
                                            '''))
    
    # from underlyings import co2_price, ng_price
    # parser.add_argument('--co2_pricing', type=str, default='gbm', \
    #                     help='gbm: geometric brownian motion')
    # parser.add_argument('--ng_pricing', type=str, default='mu-rev', \
    #                     help='mu-rev: mean-reversion | jump-d: jump diffusion')

    # read the argument values
    args = parser.parse_args()

    # read and split multi-period data
    data_frame = data_utils.read_data(args.data[0])
    data_frame = data_utils.set_index_as_date(data_frame) # make 'DATE' column, the index column
    data_sets = data_utils.split_data(data_frame)
    
    # read and process functionality for -
    # 1. univariate statistics
    if args.option == 1:
        # plot time series
        uv_stats.plot_time_series(data_frame)
        
        # one-population statistics
        uv_stats.interval_estimate_1p(data_sets, 'mean')
        # uv_stats.interval_estimate_1p(data_sets, 'variance')

        # two-population statistics
        uv_stats.interval_estimate_2p(data_sets, 'difference of means')
        # uv_stats.interval_estimate_2p(data_sets, 'ratio of variances')

    # 2. multivariate statistics
    elif args.option == 2:
        # plot 2D and 3D scatter plots
        mv_stats.plot_2d_scatter_plots(data_frame)
        mv_stats.plot_3d_scatter_plots(data_frame, data_sets)

    # 3. time series analysis, Monte Carlo simulation (3x) and real options valuation  
    elif args.option == 3:
        data, ev = ts_analysis.extract_eigenvectors(data_sets, 'Period 4') # analysis per period
        sample_project, sample_mean = ts_analysis.transform_ts(data, ev)
        mean, volatility = ts_analysis.pc1_model_fitting(sample_project[:, 0])
        ts_analysis.cash_flow_simulation(mean, volatility, ev, sample_mean, 'OP1')

    # invalid argument handler
    else:
        sys.stderr.write("Invalid argument for '--option': try with 1, 2 or 3")

if __name__ == '__main__':
    main()