import numpy as np
import matplotlib.pyplot as plt
from bayesee.generation import *

def radial_average_plot(x):
    center = (np.array(x.shape)-1) / 2
    euclidean_distance = euclidean_distance_exponential(x.shape, center, 1)
    array_unique_distance = np.unique(np.round(euclidean_distance))
    array_radial_average = np.zeros_like(array_unique_distance)
    for index_unique_distance, unique_distance in enumerate(array_unique_distance):
        array_radial_average[index_unique_distance] = np.mean(x[np.round(euclidean_distance) == unique_distance])

    fig, ax = plt.subplots()

    ax.scatter(array_unique_distance, array_radial_average, s=100)
        
    ax.set(xlabel='Eucledian distance', ylabel='Radial average')
    return array_unique_distance, array_radial_average
