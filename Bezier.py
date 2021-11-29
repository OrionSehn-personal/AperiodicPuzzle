from typing import List
import matplotlib.pyplot as plt
import numpy as np
from math import pi, e
# t0 = 0
# tx = 1
# t = np.arange(t0, tx , 0.01)

# #r = (e**t) #replace this with your r equation

# p1 = (110,150)
# p2 = (25, 190)
# p3 = (210, 250)
# p4 = (210, 30)

# x = (p1[0]*((1-t)**3)) + (p2[0]*3*((1-t)**2))*t + (p3[0] * 3 *((1-t))*t**2) + (p4[0]*t**3) 
# y = (p1[1]*((1-t)**3)) + (p2[1]*3*((1-t)**2))*t + (p3[1] * 3 *((1-t))*t**2) + (p4[1]*t**3)


# print("initial point: ")
# print("(" + str(x[0])+ ", ", end="")
# print(str(y[0])+ ")")

# print("final point: ")
# print("(" + str(x[-1])+ ", ", end="")
# print(str(y[-1])+ ")")


# plt.plot(x, y)
# plt.grid()
# plt.show()



#print("done")






def puzzleCurve(t0, t1):
    t0x = t0[0]
    t0y = t0[1]
    t1x = t1[0]
    t1y = t1[1]
    t = np.arange(0, 1.01, 0.01) 
    puzzle = []


    #determine regions:
    #this will be where a lot of the variation will be

    a = (1,1)
    b = (3,0)
    c = (4,2)
    d = (3.5,4)
    e = (5,5)
    f = (6.5,4)
    g = (6,2)
    h = (7,0)
    i = (9,1)
    

    #region (t0, a)
    x, y = bezierQuad(t0,((0.5,(a[1]))), a) #mx is variable
    puzzle.append((x, y))

    #region (a, b)
    x, y = bezierCubic(a, (2, a[1]), (2, b[1]), b)   #mx , nx are variable
    puzzle.append((x, y))

    #region (b, c)
    x, y = bezierQuad(b,(c[0],b[1]), c)
    puzzle.append((x, y))

    #region (c, d)
    x, y = bezierCubic(c, (c[0], 3.75), (d[0], 3.75), d)   #my , ny are variable
    puzzle.append((x, y))

    #region (d, e)
    x, y = bezierQuad(d,(d[0],e[1]), e)
    puzzle.append((x, y))

    #region (e, f)
    x, y = bezierQuad(e,(f[0],e[1]), f)
    puzzle.append((x, y))

    #region (f, g)
    x, y = bezierCubic(f, (f[0], 3.75), (g[0], 3.75), g)   #my , ny are variable
    puzzle.append((x, y))

    #region (g, h)
    x, y = bezierQuad(g,(g[0],h[1]), h)
    puzzle.append((x, y))

    #region (h, i)
    x, y = bezierCubic(h, (8, h[1]), (8, i[1]), i)   #mx , nx are variable
    puzzle.append((x, y))

    #region (i, t1)
    x, y = bezierQuad(i,((9.5,(i[1]))), t1) #mx is variable
    puzzle.append((x, y))


    # p1 = (110,150)
    # p2 = (25, 190)
    # p3 = (210, 250)
    # p4 = (210, 30)

    
    # a, b = bezierQuad(p1, p2, p4)
    # x, y = bezierCubic(p1, p2, p3, p4)

    # puzzle.append((a, b))




    print("initial point: ")
    print("(" + str(x[0])+ ", ", end="")
    print(str(y[0])+ ")")

    print("final point: ")
    print("(" + str(x[-1])+ ", ", end="")
    print(str(y[-1])+ ")")

    for region in puzzle:
        plt.plot(region[0], region[1])
    plt.grid()
    plt.show()

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






puzzleCurve((0, 0), (10, 0))