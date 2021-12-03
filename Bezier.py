from typing import List
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt
from random import uniform, seed


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

def puzzleCurve(t0, t1, inseed = 1, parameters = []):

    t0init = t0
    t1init = t1
    if ((t1[0] - t0[0]) == 0 ): 
        if (t0[1] < t1[1]): theta = pi/2
        else: theta = 3*pi/2
    else: theta = np.arctan((t1[1] - t0[1])/(t1[0] - t0[0]))
    if (theta == 0):
        if(t0[0] > t1[0]):
            theta = pi

    print("theta", theta)
    distance = (sqrt((t1[0] - t0[0])**2 + (t1[1] - t0[1])**2))
    print("distance", distance)
    t0 = (0, 0)
    t1 = (distance, 0)
    
    t0x = t0[0]
    t0y = t0[1]
    t1x = t1[0]
    t1y = t1[1]
    t = np.arange(0, 1.01, 0.01) 
    puzzle = []


    #determine regions:
    #this will be where a lot of the variation will be
    '''
    guidelines for region selection:

    x: t0<a<b<c<d<e<f<g<h<i<t1,
    1 < g-c < 2 #to keep nub from being too small
    

    #y: t0 = t1, 
    e > {d ,f } > {c, g, a, i} > {t0, t1, b, h}
    e shouldn't be too high, because it will collide with other puzzle edges. 
    
    '''
    seed(inseed)

    pointsx = []
    pointsx.append(t0x)
    pointsx.append(t1x)
    for i in range(9):
        pointsx.append(uniform(t0x, t1x))
    pointsx.sort()

    
    pointsy = []
    pointsy.append(t0y)
    pointsy.append(t1y)
    for i in range(9):
        pointsy.append(uniform(0, (t1x - t0x)/3))
    pointsy.sort()
    

    a = (1,1)
    b = (3,0)
    c = (4,1.5)
    d = (3.75,3)
    e = (5,4.5)
    f = (6.25,3)
    g = (6,1.5)
    h = (7,0)
    i = (9,1)
    
    # a = (pointsx[1], pointsy[4])
    # b = (pointsx[2], pointsy[2])
    # d = (pointsx[3], pointsy[9])
    # c = (pointsx[4], pointsy[6])
    # e = (pointsx[5], pointsy[10])
    # g = (pointsx[6], pointsy[7])
    # f = (pointsx[7], pointsy[8])
    # h = (pointsx[8], pointsy[3])
    # i = (pointsx[9], pointsy[5])

    #region (t0, a)
    x, y = bezierQuad(t0,(( (a[0]-t0[0])/2 , a[1] )), a) #mx is variable
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

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
    x, y = bezierQuad(i,((xmid(i, t1),(i[1]))), t1) #mx is variable
    x, y = rotate(x, y, theta)
    x = x + t0init[0]
    y = y + t0init[1]
    puzzle.append((x, y))

    print("initial point: ")
    print("(" + str(puzzle[0][0][0])+ ", ", end="")
    print(str(puzzle[0][1][0])+ ")")

    print("final point: ")
    print("(" + str(puzzle[-1][0][-1])+ ", ", end="")
    print(str(puzzle[-1][1][-1]) + ")")

    plt.rcParams["figure.figsize"] = (9, 9)

    for region in puzzle:
        plt.plot(region[0], region[1])
    
    # plt.plot([t0init[0], t1init[0]], [t0init[1], t1init[1]])
    # plt.plot(e[0], e[1], marker='o')
    plt.axis('equal')
    plt.grid()




# for i in range(1):
#     puzzleCurve((6, 9), (3, 4), i)


def puzzleMaker(lineset):
    seed = 0 
    for line in lineset:
        puzzleCurve(line[0], line[1], seed)
        seed += 1
    plt.show()


lines = [[(0,0), (0,1)], [(0,1), (1, 1)], [(1,1), (1,0)], [(1,0), (0, 0)]]
# lines = [[(1,0), (0, 0)]]
puzzleMaker(lines)

