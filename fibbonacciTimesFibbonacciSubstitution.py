from typing import List
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt, sin, cos
from random import uniform, seed, shuffle

PHI = (1 + (5)**0.5)/2

def drawFromLines(lines):
    for line in lines:
        x = np.array([line[0][0], line[1][0]])
        y = np.array([line[0][1], line[1][1]])
        plt.plot(x, y)
    plt.rcParams["figure.figsize"] = (9, 9)
    plt.axis('equal')
    plt.grid()
    plt.show()


class Tile:
    point1 = (0,0)
    point2 = (PHI * cos(pi/5), sin(2*pi/2))
    point3 = ((PHI * cos(pi/5) + cos(2*pi/5)), 0)
    point4 = (PHI * cos(pi/5), -sin(2*pi/2))
    subtype = 1 #0, 1, 2 

    def __init__(self, point1 = (0,0) , point2 =  (PHI * cos(pi/5), sin(2*pi/5)),
     point3=((PHI * cos(pi/5) + cos(2*pi/5)), 0), point4=(PHI * cos(pi/5), -sin(2*pi/5)),
      subtype = 1):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.point4 = point4
        self.subtype = subtype
    


    def substitute(self):
        newtiles = []

        if (self.subtype == 0):
            scalar1 = PHI/(1+PHI)
            scalar2 = 1 /(1+PHI)
            leftvector = ((self.point2[0] - self.point1[0]), (self.point2[1] - self.point1[1]))
            midvector = ((self.point1[0] - self.point3[0]), (self.point1[1] - self.point3[1]))#possible swap
            rightvector = ((self.point4[0] - self.point1[0]), (self.point4[1] - self.point1[1]))
            rightpoint = (self.point1[0]+(rightvector[0]*scalar1), self.point1[1] + (rightvector[1]*scalar1))
            leftpoint = ((self.point1[0]+(leftvector[0]*scalar1), self.point1[1] + (leftvector[1]*scalar1)))

            #first tile
            tile1 = Tile(self.point1, leftpoint, self.point3, rightpoint, 1)
            newtiles.append(tile1)
            #second tile
            
            botleftpoint = ((self.point2[0]+midvector[0]), (self.point2[1]+ midvector[1]))
            tile2 = Tile(self.point2, self.point3, leftpoint, botleftpoint, 0)
            newtiles.append(tile2)
            
            botrightpoint = ((self.point4[0]+midvector[0]), (self.point4[1]+ midvector[1]))
            tile3 = Tile(self.point4, botrightpoint, rightpoint, self.point3, 0)
            newtiles.append(tile3)

        if (self.subtype == 1):
            scalar1 = PHI/(1+PHI)
            scalar2 = 1 /(1+PHI)
            leftvector = ((self.point2[0] - self.point1[0]), (self.point2[1] - self.point1[1]))
            midvector = ((self.point3[0] - self.point1[0]), (self.point3[1] - self.point1[1]))
            rightvector = ((self.point4[0] - self.point1[0]), (self.point4[1] - self.point1[1]))
            midpoint = (self.point1[0]+(midvector[0]*scalar1), self.point1[1] + (midvector[1]*scalar1))
            rightpoint = (self.point1[0]+(rightvector[0]*scalar2), self.point1[1] + (rightvector[1]*scalar2))
            leftpoint = ((self.point1[0]+(leftvector[0]*scalar2), self.point1[1] + (leftvector[1]*scalar2)))

            #first tile
            tile1 = Tile(self.point2, self.point3, midpoint, leftpoint, 1)
            newtiles.append(tile1)
            #second tile
            tile2 = Tile(self.point4, rightpoint, midpoint, self.point3, 1)
            newtiles.append(tile2)
            
            #third tile
            toprightvector = ((self.point3[0] - self.point4[0]), (self.point3[1] - self.point4[1]))
            botleftpoint = ((self.point1[0]+toprightvector[0]), (self.point1[1]+ toprightvector[1]))
            tile3 = Tile(self.point1, botleftpoint, leftpoint, midpoint, 0)
            newtiles.append(tile3)

            #fourth tile
            topleftvector = ((self.point3[0] - self.point2[0]), (self.point3[1] - self.point2[1]))
            botrightpoint = ((self.point1[0]+topleftvector[0]), (self.point1[1]+ topleftvector[1]))
            tile4 = Tile(self.point1, midpoint, rightpoint, botrightpoint, 0)
            newtiles.append(tile4)

        return newtiles

    def inflate(self, lamnda):
        self.point1 = ((lamnda * self.point1[0]) , (lamnda * self.point1[1]))
        self.point2 = ((lamnda * self.point2[0]) , (lamnda * self.point2[1]))
        self.point3 = ((lamnda * self.point3[0]) , (lamnda * self.point3[1]))
        self.point4 = ((lamnda * self.point4[0]) , (lamnda * self.point4[1]))


    def getLines(self):
        lines = []
        lines.append((self.point1, self.point2))
        lines.append((self.point2, self.point3))
        lines.append((self.point3, self.point4))
        lines.append((self.point4, self.point1))
        return lines

def penroseLines(iterations = 5):
    startTile = Tile()
    tileList = [startTile]

    for i in range(iterations):
        newlist = []
        for tile in tileList:
            newlist.extend(tile.substitute())
        for tile in newlist:
            tile.inflate((sqrt(5)/2)+(1/2))
        tileList = newlist

    lineList = []
    for tile in tileList:
        lineList.extend(tile.getLines())
    lineSet = set(lineList)
    return lineSet

# drawFromLines(penroseLines())