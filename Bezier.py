from typing import List
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt
from random import uniform, seed, randint
from fibbonacciTimesFibbonacciSubstitution import *

'''
https://pomax.github.io/bezierinfo/
'''
def bezierQuad(t0, m, t1): #potentially use rational bezier to give addtional degrees of variation
    t = np.arange(0, 1.01, 0.01)
    t2 = t * t
    mt = 1 - t
    mt2 = mt * mt
    x = (t0[0]*mt2) + (m[0]*2*mt*t) + (t1[0]*t2)
    y = (t0[1]*mt2) + (m[1]*2*mt*t) + (t1[1]*t2)
    return x, y 


def bezierCubic(t0, m, n, t1):
    t = np.arange(0, 1.01, 0.01)
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    x = (t0[0]*mt3) + (3*m[0]*mt2*t) + (3*n[0]*mt*t2) + (t1[0]*t3)
    y = (t0[1]*mt3) + (3*m[1]*mt2*t) + (3*n[1]*mt*t2) + (t1[1]*t3)
    return x, y 


def xmid(pointa, pointb):
    return (pointb[0] - (pointb[0] - pointa[0])/2)

def ymid(pointa, pointb):
    return (pointb[1] - (pointb[1] - pointa[1])/2)


def rotate(x, y, theta):
    xcopy = x
    ycopy = y
    x = (xcopy * np.cos(theta)) - (ycopy * np.sin(theta)) # rotate to theta
    y = (xcopy * np.sin(theta)) + (ycopy * np.cos(theta))
    return x, y


    '''
    guidelines for region selection:

    x: t0<a<b<d<c<e<g<f<h<i<t1,
    1 < g-c < 2 #to keep nub from being too small
    

    #y: t0 = t1, 
    e > {d ,f } > {c, g, a, i} > {t0, t1, b, h}
    e shouldn't be too high, because it will collide with other puzzle edges. 
    
    '''

def puzzleCurve(t0, t1, inseed = 1, parameters = [], flipTabs = False):
    seed(inseed)
    if flipTabs:
        if randint(0, 1):
            temp = t0
            t0 = t1
            t1 = temp
    t0init = t0

    #find angle of line to origin
    #by quadrant
    
    deltax = t1[0] - t0[0]
    deltay = t1[1] - t0[1]

    if ((deltax>0) and (deltay>0)): #Q1
        theta = np.arctan(deltay/deltax)
    
    elif ((deltax<0) and (deltay>0)): #Q2
        theta = pi - np.arctan(-1*deltay/deltax)

    elif ((deltax<0) and (deltay<0)): #Q3
        theta = pi + np.arctan(deltay/deltax)

    elif ((deltax>0) and (deltay<0)): #Q4
        theta = (2*pi) - np.arctan(-1*deltay/deltax)

    elif ((deltax>0) and (deltay==0)): #Q1
        theta = 0

    elif ((deltax<0) and (deltay==0)): #Q1
        theta = pi

    elif ((deltax==0) and (deltay>0)): #Q1
        theta = pi/2

    elif ((deltax==0) and (deltay<0)): #Q1
        theta = 3*pi/2

    elif ((deltax==0) and (deltay==0)): #Q1
        theta = 0 


    #construct curve of line length
    distance = (sqrt((t1[0] - t0[0])**2 + (t1[1] - t0[1])**2))
    t0 = (0, 0)
    t1 = (distance, 0)
    t0x = t0[0]
    t0y = t0[1]
    t1x = t1[0]
    t1y = t1[1]

    puzzle = [] #holds selection of bezier curves

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

    #assign x values for critical points
    pointsx = []
    pointsx.append(t0x)
    pointsx.append(uniform((distance * 0.1), distance * 0.15)) #a 5
    pointsx.append(uniform((distance * 0.20), distance * 0.35)) #b 15
    pointsx.append(uniform((distance * 0.4), distance * 0.42)) #d 5

    pointsx.append(uniform((distance * 0.42), distance * 0.45)) #c 5
    pointsx.append(uniform((distance * 0.45), distance * 0.55)) #e 10
    pointsx.append(uniform((distance * 0.55), distance * 0.58)) #g 5


    pointsx.append(uniform((distance * 0.58), distance * 0.6)) #f 5
    pointsx.append(uniform((distance * 0.65), distance * 0.8)) #h 15
    pointsx.append(uniform((distance * 0.85), distance * 0.9)) #i 5
    pointsx.append(t1x)

    #assign y values for critical points
    pointsy = []
    maxheight = distance * 0.20
    pointsy.append(t0y)
    pointsy.append(t1y)
    pointsy.append(uniform(maxheight * -0.10, maxheight * 0.05)) #b
    pointsy.append(uniform(maxheight * -0.10, maxheight * 0.05)) #h
    pointsy.append(uniform((maxheight * 0.1), maxheight * 0.2)) #i
    pointsy.append(uniform((maxheight * 0.1), maxheight * 0.2)) #a
    pointsy.append(uniform((maxheight * 0.2), maxheight * 0.4)) #c
    pointsy.append(uniform((maxheight * 0.2), maxheight * 0.4)) #g
    pointsy.append(uniform((maxheight * 0.50), maxheight * 0.80)) #d
    pointsy.append(uniform((maxheight * 0.50), maxheight * 0.80))#f
    pointsy.append(uniform((maxheight * 0.9), maxheight * 1)) #e

    a = (pointsx[1], pointsy[4])
    b = (pointsx[2], pointsy[2])
    d = (pointsx[3], pointsy[9])
    c = (pointsx[4], pointsy[6])
    e = (pointsx[5], pointsy[10])
    g = (pointsx[6], pointsy[7])
    f = (pointsx[7], pointsy[8])
    h = (pointsx[8], pointsy[3])
    i = (pointsx[9], pointsy[5])

    #generate bezier curves by region

    #region (t0, a)
    minimum_angle = pi/10
    minimum_slope = np.tan(minimum_angle)
    # mx, my are variable, but must be along line y=x, and mx, my < a[1] 
    x, y = bezierCubic(t0, (a[1]/(2*minimum_slope), a[1]/2)  ,  (a[1]/minimum_slope , a[1]),   a) 
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))
    # plt.plot([0, 10], [0, 7.26])

    #region (a, b)
    x, y = bezierCubic(a, (xmid(a,b), a[1]), (xmid(a,b), b[1]), b ) #mx , nx are variable
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    #region (b, c)
    x, y = bezierQuad(b,(c[0],b[1]), c)
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    #region (c, d)
    x, y = bezierCubic(c, (c[0], ymid(c,d)), (d[0], ymid(c,d)), d)   #my , ny are variable
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    #region (d, e)
    x, y = bezierQuad(d,(d[0],e[1]), e)
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    #region (e, f)
    x, y = bezierQuad(e,(f[0],e[1]), f)
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    #region (f, g)
    x, y = bezierCubic(f, (f[0], ymid(f,g)), (g[0], ymid(f,g)), g)   #my , ny are variable
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    #region (g, h)
    x, y = bezierQuad(g,(g[0],h[1]), h)
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    #region (h, i)
    x, y = bezierCubic(h, (xmid(h, i), h[1]), (xmid(h,i), i[1]), i)   #mx , nx are variable
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    #region (i, t1)
    
    m = (t1x - (i[1]/minimum_slope), i[1])
    n = (t1x - (i[1]/(2*minimum_slope)), i[1]/2)

    x, y = bezierCubic(i, m, n, t1) #mx my are variable (but must be diagonal) and below i[1]
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    # plt.plot([10,0], [0, 7.26])

    for region in puzzle:
        plt.plot(region[0], region[1])



def curveGen(lineset, flipTabs = True):

    seed = 0
    for line in lineset:
        puzzleCurve(line[0], line[1], seed, flipTabs = flipTabs)
        seed += 1

    plt.rcParams["figure.figsize"] = (9, 9)
    plt.axis('equal')
    # plt.grid()



def recGrid(width, height):
    lines = []
    #veritcal lines
    for x in range(width+1):
        for y in range(height):
            temp = [(x, y), ((x), y-1)]
            lines.append(temp)
    #horizontal lines
    for x in range(width):
        for y in range(height+1):
            temp = [(x, y - 1), ((x+1), y - 1)]
            shuffle(temp)
            lines.append(temp)
    return lines


# lines = [[(0,0), (0,1)], [(0,1), (1, 1)], [(1,1), (1,0)], [(1,0), (0, 0)], [(0,0),(-1, 0)], [(-1, 1), (-1, 0)],[ (0, 1), (-1, 1)]]
# lines = [[(1,0), (0, 0)]]
# lines = [[(0,0), (10, 0)]]
# lines = recGrid(6, 4)
# curveGen(lines)

def test1():
    lines = penroseLines(5, maxradius=17)
    # drawFromLines(lines)
    curveGen(lines, flipTabs=True)

# test1()


def test2():
    # lines = [[(0,1), (1,0)], [(1,0), (0, -1)], [(0,-1), (-1,0)], [(-1,0), (0, 1)]]
    lines = [[(0,0), (10, 0)], [(0,0), (10, 0)], [(0,0), (10, 0)], [(0,0), (10, 0)] ,[(0,0), (10, 0)] ,[(0,0), (10, 0)]]

    # lines = [[(0,0), (0,1)], [(0,1), (1, 1)], [(1,1), (1,0)], [(1,0), (0, 0)], [(0,0),(-1, 0)], [(-1, 1), (-1, 0)],[ (0, 1), (-1, 1)]]
    # lines = [[(1,0), (0, 0)]]
    # lines = [[(0,0), (10, 0)]]
    # lines = recGrid(6, 4)
    curveGen(lines, flipTabs=False)

def makePuzzle(radius):

    lines = penroseLines(6, maxradius=radius)
    # drawFromLines(lines)
    
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

    curveGen(lines, flipTabs=True)
    drawFromLines(border)
    plt.show()



makePuzzle(17)
# test2()



# t0 = (0, 0)
# t1 = (1, 1)
# x, y = bezierQuad(t0, (t1[1]/3, t1[1]) ,   t1) 
# a = np.arange(0, 1, 0.01)
# b = np.empty(100)
# b.fill(1)
# plt.plot(x, y, color="blue")
# plt.plot(a, b, linestyle="dashed", color="orange")

# plt.plot(0, 0, marker="o", color="orange")
# plt.plot(t1[1]/3, t1[1], marker="o", color="blue")
# plt.plot(1, 1, marker="o", color="orange")

# x, y = bezierQuad(t0, (t1[1]/2, t1[1]) ,   t1) 
# plt.plot(x, y, color="red")
# plt.plot(t1[1]/2, t1[1], marker="o", color="red")


# x, y = bezierQuad(t0, (t1[1]/8, t1[1]) ,   t1) 
# plt.plot(x, y, color="green")
# plt.plot(t1[1]/8, t1[1], marker="o", color="green")

# plt.show()
