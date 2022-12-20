from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from math import pi, sqrt
from random import uniform, seed, randint
from fibbonacciTimesFibbonacciSubstitution import *
from write_to_svg import *
from scipy.optimize import minimize

"""
https://pomax.github.io/bezierinfo/
"""


def bezierQuad(t0, m, t1):  # potentially use rational bezier to give addtional degrees of variation
    '''returns a bezier curve with 2 control points'''
    t = np.arange(0, 1.01, 0.01)
    t2 = t * t
    mt = 1 - t
    mt2 = mt * mt
    x = (t0[0] * mt2) + (m[0] * 2 * mt * t) + (t1[0] * t2)
    y = (t0[1] * mt2) + (m[1] * 2 * mt * t) + (t1[1] * t2)
    return x, y


def bezierCubic(t0, m, n, t1):
    '''returns a bezier curve with 3 control points'''
    t = np.arange(0, 1.01, 0.01)
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    x = (t0[0] * mt3) + (3 * m[0] * mt2 * t) + (3 * n[0] * mt * t2) + (t1[0] * t3)
    y = (t0[1] * mt3) + (3 * m[1] * mt2 * t) + (3 * n[1] * mt * t2) + (t1[1] * t3)
    return x, y


def xmid(pointa, pointb):
    '''returns the x component midpoint of two points'''
    return pointb[0] - (pointb[0] - pointa[0]) / 2


def ymid(pointa, pointb):
    '''returns the y component midpoint of two points'''
    return pointb[1] - (pointb[1] - pointa[1]) / 2


def rotate(x, y, theta):
    '''rotates a point about the origin by theta'''
    xcopy = x
    ycopy = y
    x = (xcopy * np.cos(theta)) - (ycopy * np.sin(theta))  # rotate to theta
    y = (xcopy * np.sin(theta)) + (ycopy * np.cos(theta))
    return x, y

    """
    guidelines for region selection:

    x: t0<a<b<d<c<e<g<f<h<i<t1,
    1 < g-c < 2 #to keep nub from being too small
    

    #y: t0 = t1, 
    e > {d ,f } > {c, g, a, i} > {t0, t1, b, h}
    e shouldn't be too high, because it will collide with other puzzle edges. 
    
    """

def rotate_points(point_list, theta):
    '''
    rotates a list of points about the origin by theta'''
    new_list = []
    for point in point_list:
        new_list.append(rotate(point[0], point[1], theta))
    return new_list

def translate_points(point_list, x, y):
    '''translates point list by x, y'''
    new_list = []
    for point in point_list:
        new_list.append((point[0] + x, point[1] + y))
    return new_list

def linear_transform(x, a, b):
    '''linearly transforms a value x from 0 to 1, to a value from a to b'''
    return ((b - a) * x ) + a

def random_puzzle(t0, t1, inseed):
    return

def puzzleCurve(t0, t1, parameters=[], inseed=1, flipTabs=False, svg_file = None, size = 700, matplotlib_plot = False):
    '''

    puzzleCurve is a function that takes two points, t0 and t1, and generates a puzzle piece with a tab at t0 and a nub at t1.


    :param t0: tuple of x, y coordinates of the first point
    :param t1: tuple of x, y coordinates of the second point
    t0, t1: the endpoints of the curve
    inseed: the seed for the random number generator
    parameters: a list of parameters to use for the curve
    flipTabs: whether or not to flip the tabs
    svg_file: the svg file to write to
    '''
    seed(inseed)
    if flipTabs:
        if randint(0, 1):
            temp = t0
            t0 = t1
            t1 = temp
    t0init = t0

    # find angle of line to origin
    # by quadrant

    deltax = t1[0] - t0[0]
    deltay = t1[1] - t0[1]

    if (deltax > 0) and (deltay > 0):  # Q1
        theta = np.arctan(deltay / deltax)

    elif (deltax < 0) and (deltay > 0):  # Q2
        theta = pi - np.arctan(-1 * deltay / deltax)

    elif (deltax < 0) and (deltay < 0):  # Q3
        theta = pi + np.arctan(deltay / deltax)

    elif (deltax > 0) and (deltay < 0):  # Q4
        theta = (2 * pi) - np.arctan(-1 * deltay / deltax)

    elif (deltax > 0) and (deltay == 0):  # Q1
        theta = 0

    elif (deltax < 0) and (deltay == 0):  # Q1
        theta = pi

    elif (deltax == 0) and (deltay > 0):  # Q1
        theta = pi / 2

    elif (deltax == 0) and (deltay < 0):  # Q1
        theta = 3 * pi / 2

    elif (deltax == 0) and (deltay == 0):  # Q1
        theta = 0

    # construct curve of line length
    distance = sqrt((t1[0] - t0[0]) ** 2 + (t1[1] - t0[1]) ** 2)
    t0 = (0, 0)
    t1 = (distance, 0)
    t0x = t0[0]
    t0y = t0[1]
    t1x = t1[0]
    t1y = t1[1]

    puzzle = []  # holds selection of bezier curves

    # pointsx = []
    # pointsx.append(t0x)
    # pointsx.append(t1x)
    # for i in range(9):
    #     pointsx.append(uniform(t0x, t1x))
    # pointsx.sort()

    # pointsy = []
    # pointsy.append(t0y)
    # pointsy.append(t1y)
    # for i in range(9):
    #     pointsy.append(uniform(0, (t1x - t0x)/3))
    # pointsy.sort()

    seed(inseed)


# assigning as parameters

    # assign x values for critical points
    pointsx = []

    pointsx.append(t0x)
    pointsx.append(linear_transform(parameters[0], (distance * 0.15), distance * 0.2))  # a 5
    pointsx.append(linear_transform(parameters[1], (distance * 0.25), distance * 0.35))  # b 15
    pointsx.append(linear_transform(parameters[2], (distance * 0.4), distance * 0.42))  # d 5

    pointsx.append(linear_transform(parameters[3], (distance * 0.42), distance * 0.45))  # c 5
    pointsx.append(linear_transform(parameters[4], (distance * 0.45), distance * 0.55))  # e 10
    pointsx.append(linear_transform(parameters[5], (distance * 0.55), distance * 0.58))  # g 5

    pointsx.append(linear_transform(parameters[6], (distance * 0.58), distance * 0.6))  # f 5
    pointsx.append(linear_transform(parameters[7], (distance * 0.65), distance * 0.75))  # h 15
    pointsx.append(linear_transform(parameters[8], (distance * 0.8), distance * 0.85))  # i 5
    
    pointsx.append(t1x)

    # assign y values for critical points
    pointsy = []
    maxheight = distance * 0.20
    pointsy.append(t0y)
    pointsy.append(t1y)
    pointsy.append(linear_transform(parameters[9], maxheight * -0.10, maxheight * 0.05))  # b
    pointsy.append(linear_transform(parameters[10], maxheight * -0.10, maxheight * 0.05))  # h
    pointsy.append(linear_transform(parameters[11], maxheight * 0.1, maxheight * 0.2))  # i
    pointsy.append(linear_transform(parameters[12], maxheight * 0.1, maxheight * 0.2))  # a
    pointsy.append(linear_transform(parameters[13], maxheight * 0.2, maxheight * 0.4))  # c
    pointsy.append(linear_transform(parameters[14], maxheight * 0.2, maxheight * 0.4))  # g
    pointsy.append(linear_transform(parameters[15], maxheight * 0.50, maxheight * 0.80))  # d
    pointsy.append(linear_transform(parameters[16], maxheight * 0.50, maxheight * 0.80))  # f
    pointsy.append(linear_transform(parameters[17], maxheight * 0.9, maxheight * 1))  # e

    a = (pointsx[1], pointsy[4])
    b = (pointsx[2], pointsy[2])
    d = (pointsx[3], pointsy[9])
    c = (pointsx[4], pointsy[6])
    e = (pointsx[5], pointsy[10])
    g = (pointsx[6], pointsy[7])
    f = (pointsx[7], pointsy[8])
    h = (pointsx[8], pointsy[3])
    i = (pointsx[9], pointsy[5])

    # generate bezier curves by region

    minimum_angle = pi / 10
    minimum_slope = np.tan(minimum_angle)

    # region (t0, a)
    # mx, my are variable, but must be along line y=x, and mx, my < a[1]
    points_t0_a = [t0, (a[1] / (2 * minimum_slope), a[1] / 2), (a[1] / minimum_slope, a[1]), a]
    points_t0_a = rotate_points(points_t0_a, theta)
    points_t0_a = translate_points(points_t0_a, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_t0_a, svg_file, size=size)
    x, y = bezierCubic(
        points_t0_a[0], points_t0_a[1], points_t0_a[2], points_t0_a[3]
    )
    puzzle.append((x, y))
    # plt.plot([0, 10], [0, 7.26])

    # region (a, b)
    points_a_b = [a, (xmid(a, b), a[1]), (xmid(a, b), b[1]), b]
    points_a_b = rotate_points(points_a_b, theta)
    points_a_b = translate_points(points_a_b, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_a_b, svg_file, size=size)
    x, y = bezierCubic(
        points_a_b[0], points_a_b[1], points_a_b[2], points_a_b[3]
    )  # mx , nx are variable
    puzzle.append((x, y))

    # region (b, c)
    points_b_c = [b, (c[0], b[1]), c]
    points_b_c = rotate_points(points_b_c, theta)
    points_b_c = translate_points(points_b_c, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_b_c, svg_file, size=size)
    x, y = bezierQuad(points_b_c[0], points_b_c[1], points_b_c[2])
    puzzle.append((x, y))

    # region (c, d)
    points_c_d = [c, (c[0], ymid(c, d)), (d[0], ymid(c, d)), d]
    points_c_d = rotate_points(points_c_d, theta)
    points_c_d = translate_points(points_c_d, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_c_d, svg_file, size=size)
    x, y = bezierCubic(
        points_c_d[0], points_c_d[1], points_c_d[2], points_c_d[3]
    )  # my , ny are variable
    puzzle.append((x, y))

    # region (d, e)
    points_d_e = [d, (d[0], e[1]), e]
    points_d_e = rotate_points(points_d_e, theta)
    points_d_e = translate_points(points_d_e, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_d_e, svg_file, size=size)
    x, y = bezierQuad(points_d_e[0], points_d_e[1], points_d_e[2])
    puzzle.append((x, y))

    # region (e, f)
    points_e_f = [e, (f[0], e[1]), f]
    points_e_f = rotate_points(points_e_f, theta)
    points_e_f = translate_points(points_e_f, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_e_f, svg_file, size=size)
    x, y = bezierQuad(points_e_f[0], points_e_f[1], points_e_f[2])
    puzzle.append((x, y))

    # region (f, g)
    points_f_g = [f, (f[0], ymid(f, g)), (g[0], ymid(f, g)), g]
    points_f_g = rotate_points(points_f_g, theta)
    points_f_g = translate_points(points_f_g, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_f_g, svg_file, size=size)
    x, y = bezierCubic(
        points_f_g[0], points_f_g[1], points_f_g[2], points_f_g[3]
    )  # my , ny are variable
    puzzle.append((x, y))

    # region (g, h)
    points_g_h = [g, (g[0], h[1]), h]
    points_g_h = rotate_points(points_g_h, theta)
    points_g_h = translate_points(points_g_h, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_g_h, svg_file, size=size)
    x, y = bezierQuad(points_g_h[0], points_g_h[1], points_g_h[2])
    puzzle.append((x, y))

    # region (h, i)
    points_h_i = [h, (xmid(h, i), h[1]), (xmid(h, i), i[1]), i]
    points_h_i = rotate_points(points_h_i, theta)
    points_h_i = translate_points(points_h_i, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_h_i, svg_file, size=size)
    x, y = bezierCubic(
        points_h_i[0], points_h_i[1], points_h_i[2], points_h_i[3]
    )  # mx , nx are variable
    puzzle.append((x, y))

    # region (i, t1)
    m = (t1x - (i[1] / minimum_slope), i[1])
    n = (t1x - (i[1] / (2 * minimum_slope)), i[1] / 2)
    points_i_t1 = [i, m, n, t1]
    points_i_t1 = rotate_points(points_i_t1, theta)
    points_i_t1 = translate_points(points_i_t1, t0init[0], t0init[1])
    if svg_file != None:
        draw_curve(points_i_t1, svg_file, size=size)
    x, y = bezierCubic(
        points_i_t1[0], points_i_t1[1], points_i_t1[2], points_i_t1[3]
    )  # mx my are variable (but must be diagonal) and below i[1]
    puzzle.append((x, y))
    # plt.plot([10,0], [0, 7.26])

    if matplotlib_plot:
        for region in puzzle:   
            plt.plot(region[0], region[1])


def curveGen(lineset, paramset, flipTabs=True, svg_file = None, size = 700, mat_plot_lib = False):

    seed = 0
    for i in range(len(lineset)):
        puzzleCurve(lineset[i][0], lineset[i][1], paramset[i], seed, flipTabs=flipTabs, svg_file=svg_file, size= size)
        seed += 1

    if mat_plot_lib:
        plt.rcParams["figure.figsize"] = (9, 9)
        plt.axis("equal")


def recGrid(width, height, scaling=0.6, translate=100):
    '''Generates a list of lines for a rectangular grid'''
    lines = []
    border = []
    # veritcal lines
    for x in range(width + 1):
        for y in range(height):
            temp = [(x, y), ((x), y - 1)]
            if x == 0 or x == width:
                border.append(temp)
            else:
                lines.append(temp)
    # horizontal lines
    for x in range(width):
        for y in range(height + 1):
            temp = [(x, y - 1), ((x + 1), y - 1)]
            shuffle(temp)

            if y == 0 or y == height:
                border.append(temp)
            else:
                lines.append(temp)
    for line in lines:
        line[0] = ((line[0][0] - translate) * scaling, (line[0][1] - translate) * scaling)
        line[1] = ((line[1][0] - translate) * scaling, (line[1][1] - translate) * scaling)
        
    for line in border:
        line[0] = ((line[0][0] - translate) * scaling, (line[0][1] - translate) * scaling)
        line[1] = ((line[1][0] - translate) * scaling, (line[1][1] - translate) * scaling)

    return lines, border

def trigrid(width, height, scaling=0.8, translate=5):
    border = []
    lines = []
    init_triangle = [(0, 0), (1, 0), (0.5, 0.5 * sqrt(3))]
    for x in range(width):
        for y in range(height):
            if y % 2 == 0:
                temp = [(init_triangle[0][0] + x, init_triangle[0][1] + y*(0.5 * sqrt(3))), (init_triangle[1][0] + x, init_triangle[1][1] + y*(0.5 * sqrt(3))), (init_triangle[2][0] + x, init_triangle[2][1] + y*(0.5 * sqrt(3)))]

                lines.append((temp[0], temp[1]))
                lines.append((temp[1], temp[2]))
                lines.append((temp[2], temp[0]))

            else:
                temp = [(init_triangle[0][0] + x + 0.5, init_triangle[0][1] + y*(0.5 * sqrt(3))), (init_triangle[1][0] + x + 0.5, init_triangle[1][1] + y*(0.5 * sqrt(3))), (init_triangle[2][0] + x + 0.5, init_triangle[2][1] + y*(0.5 * sqrt(3)))]

                lines.append((temp[0], temp[1]))
                lines.append((temp[1], temp[2]))
                lines.append((temp[2], temp[0]))


    line_lists = []
    for line in lines:
        line_lists.append(list(line))

    for line in line_lists:
        line[0] = ((line[0][0] - translate) * scaling, (line[0][1] - translate) * scaling)
        line[1] = ((line[1][0] - translate) * scaling, (line[1][1] - translate) * scaling)
        
    border_lists = []
    for line in border:
        border_lists.append(list(line))

    for line in border_lists:
        line[0] = ((line[0][0] - translate) * scaling, (line[0][1] - translate) * scaling)
        line[1] = ((line[1][0] - translate) * scaling, (line[1][1] - translate) * scaling)

    return line_lists, border_lists


def makePuzzle(radius, svg_filename, size=1500):

    lines = penroseLines(6, maxradius=radius)
    # drawFromLines(lines)
    

    file = initialize_svg(svg_filename, size=size)

    lines = list(lines)
    maxradius = radius - (PHI)
    border = []
    maxradiussqrd = maxradius*maxradius
    counter = len(lines)
    index = 0
    while (counter > index):
        line = lines[index] 
        if(
            ((line[0][0]**2 + line[0][1]**2) > maxradiussqrd) and
            ((line[1][0]**2 + line[1][1]**2) > maxradiussqrd)
            ):
            border.append(lines.pop(index))
            counter -=1

        else:
            index += 1

    curveGen(lines, flipTabs=True, svg_file=file, size=size)
    for line in border:
        draw_line(line[0], line[1], file, size=size)
    finalize_svg(file)
    drawFromLines(border)

    plt.show()



def minkowski_difference(params):
    '''
    returns a list of vectors which represents the set of differences between each vector
    '''
    vectors = []
    for i in range(len(params)):
        for j in range(len(params)):
            if i != j:
                vectors.append(params[i] - params[j])
    return vectors

def euclidean_set_difference(params):
    '''
    This function takes a list of vectors and returns a single value representing the 
    "uniqueness" of the set of vectors as determined by the sum of the norms of the minkowski difference.
    '''
    differences = []
    for i in range(len(params)):
        for j in range(i+1, len(params)):
            differences.append(np.linalg.norm(params[i] - params[j]))
    return pd.Series(differences)

def manhattan_set_difference(params):
    '''
    This function takes a list of vectors and returns a single value representing the 
    "uniqueness" of the set of vectors as determined by the sum of the norms of the minkowski difference.
    '''
    differences = []
    for i in range(len(params)):
        for j in range(i+1, len(params)):
            differences.append(np.linalg.norm(params[i] - params[j], ord=1))
    return pd.Series(differences)

def hamming_set_difference(params):
    '''
    This function returns a list of vectors which represent the hamming difference between 
    each vector in the set.
    '''
    differences = []
    for i in range(len(params)):
        for j in range(i+1, len(params)):
            differences.append(np.sum(params[i] ^ params[j]))
    return pd.Series(differences)


def bitwise_distribution(num_edges):
    '''
    This function takes a number of edges and returns a list of lists of vectors representing
    some good ways to describe puzzle curves so that they are quite different.
    '''

    # sep = (2**18-1)//num_edges # this is the number of possible vectors
    sep = (2^18) - 1
    params = []
    for i in range(num_edges):
        binrep = np.binary_repr(i*sep, width=18)
        # print(binrep)
        params.append(np.array(list(binrep), dtype=int))

    return params

def random_distribution(num_edges):
    '''
    This function takes a number of edges and returns a list of lists of vectors representing
    some good ways to describe puzzle curves so that they are quite different.
    '''

    params = []
    for i in range(num_edges):
        vector = []
        for j in range(18):
            vector.append(uniform(0,1))
        params.append(np.array(vector))

    return params

def random_unit_distribution(num_edges):
    '''
    function takes a number of edges and returns a list of vectors which have unit components.
    '''
    params = []
    for i in range(num_edges):
        vector = []
        for j in range(18):
            vector.append(randint(0,1))
        params.append(np.array(vector))

    return params

def array_hamming_distance(array, n):
    # print((array))
    # print("len array: ", len(array))
    total_distance = 0
    for i in range(0, (len(array)), n):
        # print("i: ", i)
        for j in range(i+n, (len(array)), n):
            # print("j: ", j)
            # total_distance += np.sum(array[i:i+n] ^ array[j:j+n])
            vector1 = array[i:i+n]
            vector2 = array[j:j+n]
            for k in range(n):
                if vector1[k] != vector2[k]:
                    total_distance += 1
            # total_distance += 1
            # print("vector1, vector2: ", array[i:i+n], array[j:j+n])
            # print("added distance: ", np.sum(array[i:i+n] ^ array[j:j+n]))
# 
    # print(total_distance)
    # print()
    return -total_distance


def test0():
    print(array_hamming_distance(np.array([0, 0, 0, 1]), 2))
    print(array_hamming_distance(np.array([0, 0, 0, 1, 0, 0]), 2))
    print(array_hamming_distance(np.array([0,0,0, 0,1,1]), 3))

    print(array_hamming_distance(np.array([0,0, 0,1, 1,0]), 2))
    
    x0 = np.array([0, 0, 0, 1, 1, 0])
    # bounds = [(0,1)]*6
    # res = minimize(array_hamming_distance, x0, args=(2,), method='CG', options={'disp': True}, iter)
    
    # print(res.x)





# def test3():
#     print("start")
#     point_list = [(0, 1)]
#     print(rotate(0, 1, pi))
#     print(rotate_points(point_list, pi))
#     print("try")

# test3()

# def test4():
#     line = [[(0, 0), (10, 0)]]
#     filename = "fivetry.svg"
#     file = initialize_svg(filename)
#     curveGen(line, flipTabs=False, svg_file=file)
#     finalize_svg(file)

# test4()

def test5():
    
    # parameters = [0] * 18
    # print(f"Puzzle Parameters {parameters}")
    # puzzleCurve((0, 0), (1, 0) , 1, parameters)

    # parameters = [1] * 18
    # print(f"Puzzle Parameters {parameters}")
    # puzzleCurve((0, 0), (1, 0) , 1, parameters)

    # parameters = [0] * 18
    # print(f"Puzzle Parameters {parameters}")
    # puzzleCurve((0, 0), (1, 0) , 1, parameters)

    # parameters = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # print(f"Puzzle Parameters {parameters}")
    # puzzleCurve((0, 0), (1, 0) , 1, parameters)

    parameters = [0.5] * 18
    print(f"Puzzle Parameters {parameters}")
    puzzleCurve((0, 0), (1, 0) , 1, parameters)
    plt.show()

# test5()

def test6():
    parameters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(f"Puzzle Parameters {parameters}")
    puzzleCurve((0, 0), (1, 0) , 1, parameters)
    for i in range(0, 18):
        parameters[i] = 1
        print(f"Puzzle Parameters {parameters}")
        puzzleCurve((0, 0), (1, 0) , 1, parameters)
        parameters[i] = 0
    plt.show()

# test6()


def test7():
    parameters = [0.5] * 18

    parameters = np.array(parameters)
    print(f"Puzzle Parameters {parameters}")
    puzzleCurve((0, 0), (1, 0), parameters)
    plt.show()

# test7()
def test8():
    penlines = penroseLines(2, maxradius=17)
    svg_file = open("penrose.svg", "w")
    curveGen(penlines, bitwise_distribution(len(penlines)), flipTabs=True, svg_file=svg_file, size=700)
    svg_file.close()
    plt.show()

def test9():
    # print(f"Bitwise distribution: {uniqueness_metric(bitwise_distribution(257))}")
    # print(f"Random distribution: {uniqueness_metric(random_distribution(257))}")
    print(f"Average Bitwise distance: {uniqueness_metric(bitwise_distribution(256))/(256 * 128)}")
    print(f"Average Random distance: {uniqueness_metric(random_distribution(256))/(256 * 128)}")
    print(f"minimum Bitwise distance: {uniqueness_metric2(bitwise_distribution(256))}")
    print(f"minimum Random distance : {uniqueness_metric2(random_distribution(256))}")



def test11():
    size = 255
    # differences = euclidean_set_difference(random_distribution(size))
    # print(f"Euclidean stats: {differences.describe()}")
    # fig = px.histogram(differences, nbins=100, title="Euclidean distance between random distribution")
    # fig.show()

    # differences = euclidean_set_difference(bitwise_distribution(size))
    # print(f"Euclidean stats: {differences.describe()}")
    # fig = px.histogram(differences, nbins=100, title="Euclidean distance between Bitwise Distribution")
    # fig.show()

    # differences = hamming_set_difference(bitwise_distribution(size))
    # print(f"Euclidean stats: {differences.describe()}")
    # fig = px.histogram(differences, nbins=100, title="Hamming distance between Bitwise Distribution")
    # fig.show()

    differences = hamming_set_difference(random_unit_distribution(size))
    print(f"Euclidean stats: {differences.describe()}")
    fig = px.histogram(differences, nbins=100, title="Hamming distance between Bitwise Distribution")
    fig.show()


if __name__ == "__main__":
    # test0()
    figlines, border = trigrid(5, 5, 1, 0)
    drawFromLines(figlines)
    drawFromLines(border)
    