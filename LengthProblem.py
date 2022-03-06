from turtle import distance
import pandas as pd
from random import uniform

def spaced_points(x0, x1, epsilon):
    '''
    Returns a set of randomly distrubuted points such that no two points 
    are not within some minimum distance epsilon from each other.
    '''
    delta = x1 - x0
    min = 1
    max = (delta * min) / (9 * epsilon)
    distance_list = []
    for i in range(10):
        distance_list.append(uniform(min, max))
    distance_series = pd.Series(distance_list)
    t = distance_series.sum()
    distance_series = distance_series * (delta/t)
    return distance_series

distances = spaced_points(0, 10, 2)
point_list = []
for i in range (len(distances)):
    point_list.append(distances.head(i).sum())
point_list.append(10)
print(len(point_list))
for i in range(len(point_list)-1):
    print(f"this far apart: {point_list[i+1] - point_list[i]}")

    

