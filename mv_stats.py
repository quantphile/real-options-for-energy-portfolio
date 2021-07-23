import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import ellipsoid

def plot_2d_scatter_plots(data_frame):
    # plot the scatter plot matrix using seaborn
    sns.set_theme(style="ticks")
    sns.pairplot(data_frame, hue='PERIOD')
    plt.show()

def confidence_region_estimate_1p(period, data_set, ax, pcolours, irow):
    data_sample = data_set[['ELEC', 'NG', 'CO2']].dropna().to_numpy()

    # sample_mean = data_sample.mean(axis=0)
    # sample_covariance = np.cov(data_sample.T)

    ET = ellipsoid.EllipsoidTool()
    U, center, radii, rotation = ET.getMinVolEllipse(data_sample, .01)
    print(f'eigenvectors/principal components for {period}: {U}')
    ET.plotEllipsoid(center, radii, rotation, ax)

def plot_3d_scatter_plots(data_frame, data_sets):
    pcolours = ['blue', 'orange', 'green', 'red']
    
    # separated scatter plots, based on periods
    fig = plt.figure()

    for key in data_sets.keys():
        irow = list(data_sets.keys()).index(key)
        
        ax = fig.add_subplot(2, 2, (irow + 1), projection='3d')
        ax.scatter(data_sets[key]['ELEC'], data_sets[key]['NG'], data_sets[key]['CO2'], alpha=1, s=25, c=pcolours[irow])
        ax.set_title(f'95% confidence ellipsoid for mean vector in {key}')
        ax.set_xlabel('ELEC')
        ax.set_ylabel('NG')
        ax.set_zlabel('CO2')
        ax.set_xlim(20, 100)
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 40)
        confidence_region_estimate_1p(key, data_sets[key], ax, pcolours, irow)    
    
    plt.tight_layout()
    plt.show()