from turtle import distance
import pandas as pd
from random import uniform

def spaced_x_vals(x0, x1, epsilon, paramList=[]):
    '''
    spaced generation for x values
    '''
    distance_series = spaced_distances(x0, x1, epsilon, paramList=[])
    point_list = []
    for i in range (len(distance_series)):
        point_list.append(distance_series.head(i).sum())
    
    point_list.append(x1)
    return point_list

def spaced_distances(x0, x1, epsilon, paramList=[]):
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

def spaced_y_vals(y0, y1, epsilon, paramList=[]):
    points = spaced_distances(y0, y1, epsilon, paramList=[])
    points = list(points)
    for i in range(len(points)):
        if i >= 5:
            points[i] = -points[i]
    points[1] = -points[1]
    points[-2] = -points[-2]
    distance_series = pd.Series(points)
    point_list = []
    for i in range (len(distance_series)):
        point_list.append(distance_series.head(i).sum())
    point_list.append(y0)
    return point_list


def test1():
    print("get x points")
    point_list = spaced_x_vals(0, 10, 0.2)
    print(point_list)
    print(f"length: {len(point_list)}")
    for i in range(len(point_list)-1):
        print(f"this far apart: {point_list[i+1] - point_list[i]}")

def test2():
    print("get y points")
    point_list = spaced_y_vals(0, 10, 0.2)
    print(point_list)
    print(f"length: {len(point_list)}")
    for i in range(len(point_list)-1):
        print(f"this far apart: {point_list[i+1] - point_list[i]}")

def test3():
    print("test spaced distances")
    print(spaced_distances(0, 10, 0.2))
if __name__ == "__main__":
    test1()
    test2()
    test3()
