# import numpy as np
import math
# import matplotlib.pyplot as plt

def P(arr):
    return [[arr[2*i],arr[2*i+1]] for i in range(int(len(arr)/2))]

def add(v,w):
# assumes same length array
    return [v[i] + w[i] for i in range(len(v))]

def scale(k,v):
# assumes same length array
    return [k*v[i] for i in range(len(v))]
    
def matmul(M,v):
#Wiorks fir a 2x2 matrix and 2x1 vector
    return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]

def sign(x):
    if x>0:
        return 1
    if x == 0:
          return 0
    return -1

def bezout(a, b):
    s1 = sign(a)
    s2 = sign(b)
    a = abs(a)
    b = abs(b)
    """
    Returns a list `result` of size 3 where:
    Referring to the equation ax + by = gcd(a, b)
        result[0] is gcd(a, b)
        result[1] is x
        result[2] is y
    """
    s = 0; old_s = 1
    t = 1; old_t = 0
    r = b; old_r = a

    while r != 0:
        quotient = old_r//r # In Python, // operator performs integer or floored division
        # This is a pythonic way to swap numbers
        # See the same part in C++ implementation below to know more
        old_r, r = r, old_r - quotient*r
        old_s, s = s, old_s - quotient*s
        old_t, t = t, old_t - quotient*t
    return [old_r, s1*old_s, s2*old_t]

def find(cone):
    # x1 = math.ceil(cone[0][0]); y1 = math.ceil(cone[0][1]); x2 = math.ceil(cone[1][0]); y2 = math.ceil(cone[1][1])
    x1 = cone[0][0]; y1 = cone[0][1]; x2 = cone[1][0]; y2 = cone[1][1]
    # x1 = int(x1/math.gcd(x1,y1)); y1 = int(y1/math.gcd(x1,y1))
    # x2 = int(x2/math.gcd(x2,y2)); y2 = int(y2/math.gcd(x2,y2))
    
    T = [[y1,-x1],[bezout(x1,y1)[1],bezout(x1,y1)[2]]]
    w = [x2,y2]
    Tw = matmul(T,w)
    sign = 1
    if Tw[0] == 0:
        return None
    if Tw[0] < 0:
        sign = -1;
    k = math.ceil(sign*Tw[1]/Tw[0])
    # invT = np.linalg.inv(T)
    invT = [[bezout(x1,y1)[2],x1],[-bezout(x1,y1)[1], y1]]
    temp = [sign,k]
    return matmul(invT,temp)
    

#####
# NOTE: ASSUMES CONVEX CONE ?
#####


# Assumes they are not opposite?
# also prints endpoints
def subdivision(cone):
    start = cone[0]; end = cone[1]
    print("")
    print("Between {},{}".format(start,end))
    fc = end
    ans = [start]
    while (find(cone)[0] != end[0] or find(cone)[1] != end[1]):
        fc = find(cone)
        
        #If want to print the Vector
        Vector(fc[0],fc[1])
        
        # if fc[0] != 0:
        #     print("{}, {}".format(fc,(u[0]+v[0])/fc[0]))
        # else:
        # print(fc)
        ans.append(fc)
        cone = [fc,cone[1]]
    ans.append(end)
    print(len(ans)-2)
    return ans
    
## Useful for printing, and returns number of resolution vectors
def psubdivision(cone):
    ans = subdivision(cone)
    i = 1; size = len(ans)
    number_of_red = 0
    while i < size-1:
        prev = ans[(i-1)%size]; curr = ans[i]; nextt = ans[(i+1)%size]
        Vector(curr[0],curr[1]).color = 'red'
        number_of_red += 1
        if curr[0] != 0:
            print("{},{}".format(curr,(prev[0]+nextt[0])/curr[0]))
        else:
            print("{},{}".format(curr,(prev[1]+nextt[1])/curr[1]))
        i += 1
    print("")
    
    return number_of_red

def fanc(polygon, str):
    i = 0; size = len(polygon)
    vectors = []
    print("Fan:")
    while i < size:
        u = polygon[(i-1)%size]; v = polygon[i]; w = polygon[(i+1)%size]
        a = w[0]-v[0]; b = w[1]-v[1]; d = math.gcd(a,b)
        x = u[0]-v[0]; y = u[1]-v[1]
        # if d == 0:
        #     print(a,b)
        v = [a/d, b/d]

        dot = x*v[1] - y*v[0]
        sign = 1
        if dot < 0:
            sign = -1
        ans = [sign*v[1], -sign*v[0]]
        vectors.append(ans)
        print(ans)
        s = "({},{})".format(ans[0],ans[1])
        Vector(ans[0],ans[1]).color = str
        i += 1
    return vectors


def fan(polygon):
    return fanc(polygon, 'black')
    
#Subdivides the whole fan
def sub(fan):
    i = 0; n = len(fan)
    arr = []
    while i < n:
        # Vector(fan[i % n][0],fan[i % n][1])
        sub_array = subdivision([fan[i % n],fan[(i+1) % n]])
        for elem in sub_array[:-1]:
            arr.append(elem)
        i += 1
    return arr

fan = [[1,0],[-7,19],[-1,0],[0,-1]]
print(sub(fan))
  
#Subdivides the whole fan
def psub(fan):
    i = 0; n = len(fan)
    while i < n:
        Vector(fan[i % n][0],fan[i % n][1])
        psubdivision([fan[i % n],fan[(i+1) % n]])
        i += 1
    
# k = 2
# c = 6
# d = 10 - c
# w = 4
# z = 3
# fan = [[0,1],[d,1],[w - d*z, -z],[-c,1 + k*c]]
# psub(fan)

#fan= [[0,1],[-3,-2],[3,1],[12,5]]
#psub(fan)

# fan = [[1, 0], [-134, 217], [-21, 34], [51, -83]]
# psub(fan)

# fan = [[0,1],[23,10],[-91,-40]]
# psub(fan)

# fan = [[0,1],[5,-2]]
# psub(fan)

# k=0
# fan = [[0,1],[-3,1],[-7-3*(7*k+1),7*k+1],[3,-2]]
# psub(fan)

# fan = [[0,1],[2,-1],[-3,-4],[-1,3]]
# psub(fan)

# fan = [[0,1],[3,-2],[11,-13],[-10,1],[-3,1]]
# psub(fan)

# fan = [[0,1],[10,-7],[-11,2],[-3,1]]
# psub(fan)

# y = 6
# fan = [[0,1],[-9,1],[-1-9*y,y]]
# psub(fan)

# # Case: s = 0, (a,b) = (0,7)
# l = 3
# fan = [[-9-8*l,1+8*l],[-1,1],[0,1]]
# psub(fan)

# Case: s = -1, (a,b) = (0,6)
# l = 2
# fan = [[-8-7*l,1+7*l],[-1,1],[0,1]]
# psub(fan)

# fan = [[1,0],[7,-15]]
# psub(fan)

# fan = [[1,0],[13,-15]]
# psub(fan)
    
# x=-34; y=-39; z=41; w=47
# fan = [[-41,47],[1,0],[47,-55]]   
# psub(fan)
    
# fan = [[0,1],[217,-183]]
# psub(fan)

##########
# P and Maximal resolution playground
##########
# fan = [[1,0],[-7,19]]
# psub(fan)
# Line(Point(1,0),Point(-7,19))
# u1=(0,1), u2=(−1,4), u3=(−2,7), u4=(−1,3), u5=(−5,14), u6=(−4,11)
# Vector(0,1,color = 'blue')
# Vector(1,0)
# Vector(0,1, color = 'blue')
# Vector(-1,4,color = 'blue')
# Vector(-2,7,color = 'blue')
# Vector(-1,3,color = 'blue')
# Vector(-5,14,color = 'blue')
# Vector(-4,11,color = 'blue')
# Vector(-7,19)

# Line(Point(1,0),Point(-1,4))
# Line(Point(-1,4),Point(-7,19))

# Vector(19,7)
# Vector(1,1)
# Vector(2,1)
# Vector(5,2)
# Vector(8,3)

# Draw vectors  
def draw_vec(vectors,str):
    for vec in vectors:
        Vector(vec[0],vec[1]).color = str
    
def poly_from_fan(fan):
    first = [0,0]
    second = [0,0]
    i = 0
    while i < len(fan):
        v = fan[i]; w = fan[(i+1) % len(fan)]
        a = v[0]; b = v[1] # so v = [a,b]
        sign = 1; # v_perp = [-b, a]
        
        dot = -b * w[0] + a * w[1]
        if dot < 0:
            sign = -1
        v_perp = scale(sign,[-b,a])
        
        first = second
        second = add(second, v_perp)
        Segment(Point(first[0],first[1]), Point(second[0],second[1]))
        i += 1
    
    
def fan_split(polygon):
    vectors = fan(polygon)
    i = 0; size = len(vectors)
    ans = []
    while i < size:
        v = vectors[i]; w = vectors[(i+1)%size]
        psubdivision([v,w])
        i += 1
        
def vec_split(vectors):
    for vec in vectors:
        Vector(vec[0],vec[1])
    num_of_vectors = len(vectors);
    i = 0; size = len(vectors)
    ans = []
    while i < size:
        v = vectors[i]; w = vectors[(i+1)%size]
        num_of_vectors += psubdivision([v,w])
        i += 1
    print("NUMBER OF VECTORS = {}".format(num_of_vectors))


def fanc_split(polygon,str):
    vectors = fan(polygon,str)
    i = 0; size = len(vectors)
    while i < size:
        v = vectors[i]; w = vectors[(i+1)%size]
        subdivision([v,w])
        i += 1
        
def coefficients(vec):
    n = len(vec)
    v0 = [1,0]; v1 = [0,1]
    ans = [v0,v1]
    for i in range(n):
        next = (i+1) % n
        ans.append(add(scale(vec[next],ans[i+1]),scale(-1,ans[i])))
    return ans
    # M = Matrix([add(ans[n],[-1,0]),add(ans[n+1],[0,-1])])
    # return M
    
def fan_from_surface(vec, list_of_black):
    n = len(vec)
    v0 = [1,0]; v1 = [0,1]
    recursive_list = [v0,v1]
    for i in range(n):
        next = (i+1) % n
        next_vec = add(scale(vec[next],recursive_list[i+1]),scale(-1,recursive_list[i]))
        recursive_list.append(next_vec)
        
        ### If want to draw all
        # Vector(ans[i][0],ans[i][1])
        
        if i in list_of_black:
            Vector(recursive_list[i][0],recursive_list[i][1])
            print(recursive_list[i])
    return recursive_list[:n]


# vec = [[0,1],[1,0],[4,-3],[-2,1]]
# psub(vec)

# vec = [[-1,2],[1,0],[1,-1],[5,-9],[-7,12]]
# psub(vec)

# fan = [[0,1],[4,-3],[-2,1],[-8,5],[-3,2],[-1,1]]
# psub(fan)
        
# fan = [[1,0],[5,-4],[-1,2],[-3,8],[-1,3],[0,1]]
# psub(fan)
        
# vec = [1, 6, 2, 1, 2, 3, 1, 2, 2, 2, 2, 2, 2, 2]
# list_of_black = [0,1,3,6]
# v = fan_from_surface(vec,list_of_black)
# vec_split(v)


########

# cone = [[0,1],[-1,-2]]
# Vector(0,1)
# Vector(-1,-2)
# psubdivision(cone)

#### draw_from_coefficients 
# vec = [1, 2, 3, 1, 2, 2, 3, 2, 1, 3, 2, 2]
# list_of_black = [0,3,8,9]
# fan_from_surface(vec, list_of_black)

# vec = [[1,0],[-1,2],[-3,5],[2,-7]]
# vec_split(vec)
    
# z2_generator = np.eye(2).astype(int)
# b1 = np.array([1,0])
# b2 = np.array([0,1])
# basis_vectors = [b1, b2]

# ldown = np.array([-15, -15])
# rup = np.array([15, 15])

# fig, ax = plt.subplots()
# points = plotLattice(ax, basis_vectors, ldown, rup, 'blue', 3, 0.25)

################

# plt.arrow(0,0,2,2, head_width=0.05)

# cone = [[1,1],[13,-46]]
# print(cone[0])
# subdivision(cone)
# print(cone[1])

################

## (q) example 7

# Polygon([Point(0,0),Point(-3,3),Point(43,16),Point(3,0)])
# polygon = [[3,0],[0,3],[46,16],[6,0]]
# fan(polygon)
# fan_split(polygon)
# arr = fan(polygon)
# # subdivision([arr[0],arr[1]])
# # Vector(2,-7)

# one_param = [[0,0],[3,1]]
# fanc(one_param,'blue')

## Remarks: 2 coinicde, 2 between (both adjacent) both colliding with resolution vectors
## #NP = 4

################

# # (q) example 4

# Polygon([Point(0,0),Point(-1,1),Point(37,28),Point(2,0)])
# polygon = [[0,0],[-1,1],[37,28],[2,0]]
# # Polygon([Point(0,0),Point(2,0),Point(37,28),Point(-1,1)])
# # polygon = [[0,0],[2,0],[37,28],[-1,1]]
# fan(polygon)
# # # fan_split(polygon)

# # # print("Newton Polygon of R2")

# Polygon([Point(7,5),Point(2,1),Point(0,0)])
# small = [[7,5],[2,1],[0,0]]
# fanc(small,'blue')

## Remarks: 2 coinicde, 2 between (both adjacent)
## #NP = 3

################

# # Special (13,14) example

# Polygon([Point(0,0),Point(4,0),Point(5,1),Point(15,36)])
# poly = [ [0 , 0] , [4 , 0] , [5 , 1] , [15 , 36] ]
# fan(poly)
# fan_split(poly)

# # print("Newton Polygon of R1")

# Polygon([Point(0,0),Point(1,0),Point(3,7),Point(2,5)])
# np1 = [[0,0],[1,0],[3,7],[2,5]]
# fanc(np1,'blue')

# print("Newton Polygon of R2")

# Polygon([Point(1,0),Point(3,0),Point(4,1),Point(5,4),Point(11,25),Point(1,1)])
# np2 = [[1,0],[3,0],[4,1],[5,4],[11,25],[1,1]]
# fanc(np2,'blue')

##Code test
# Vector(-7,2)
# subdivision([[-7,2],[12,-5]])
# Vector(12,-5)

## Remarks: 6 sides newton Polygon of R2, even R1 has quadrilateral
## 4 coincide, 2 in region that are not adjacent

################

# # Random Example


# Polygon([Point(0,0),Point(5,0),Point(11,14),Point(11,15)])
# poly = [[0,0],[5,0],[11,14],[11,15]]
# fan(poly)
# fan_split(poly)

# Polygon([Point(0,0),Point(2,0),Point(5,7),Point(2,3)])
# NP1 = [[0,0],[2,0],[5,7],[2,3]]
# fan(NP1)
# fanc(NP1,'green')

# NP2 = [[0,0],[1,1]]
# fanc(NP2,'blue')

################

# # Random Example 2

# Polygon([Point(0,0),Point(5,0),Point(9,15),Point(8,14)])
# poly = [[0,0],[5,0],[8,14],[9,15]]
# fan(poly)

# small = [[0,0],[2,0],[3,3],[4,7]]
# fanc(small,'red')

## Remarks: only 1 coincides

################

# # Random Example 3

# Polygon([Point(10,0),Point(0,20),Point(27,5),Point(12,0)])
# poly = [ [10 , 0] , [0 , 20 ], [27 , 5] , [12 , 0] ]
# fan_split(poly)
# r1 = [[0,0],[1,0]]
# r2 = [ [8, 0], [9, 0], [21, 4], [20, 5], [2, 15], [0, 16] ]

# fan(poly)
# fanc(r1,'blue')

################

# # Both 1 parameter subgroups

# Polygon([Point(0,0),Point(1,0),Point(14,32),Point(6,14)])
# poly = [[0,0],[1,0],[14,32],[6,14]]
# fan_split(poly)
# r1 = [[0,0],[1,2]]
# r2 = [[0,0],[2,5]]
# fanc(r1,'blue')

################

# # Another Triangle

# Polygon([Point(0,0),Point(1,0),Point(21,8),Point(8,5)])
# poly = [ [0 , 0] , [1 , 0] , [21 , 8] , [8,5]]
# fan(poly)
# fan_split(poly)
# r1 = [[0,0],[2,1]]
# r2 = [[0,0],[5,2],[1,1]]

# fanc(r2,'blue')

################

# Another Triangle 2

# Polygon([Point(30,0),Point(36,0),Point(3,22),Point(0,13)])
# poly = [ [30, 0] , [36 , 0] , [3 , 22] , [0 , 13]]
# fan(poly)
# fan_split(poly)
# r1 = [[2,0],[0,1]]
# r2 = [[3,0],[0,2],[0,1]]

# fanc(r1,'blue')

##############

# Both 1 param

# Polygon([Point(0,0),Point(1,0),Point(20,14),Point(7,5)])

# poly = [ [0 , 0] , [1 , 0] , [20 , 14], [7 , 5] ]
# r1 = [[0,0], [3,2]]
# r2 = [[0,0], [4,3]]


##############

# Genus 2 Example

# Polygon([Point(0,0),Point(5,2),Point(7,3),Point(3,8),Point(1,5)])
# poly = P([0,0,5,2,7,3,3,8,1,3])
# fan(poly)
# fan_split(poly)

# e1 = [[0,0],[0,1]]
# e2 = [[0,0],[1,2],[2,1]]
# r1 = [[0,0],[1,0],[3,1],[1,3]]

# R = e2

# fanc(R,'blue')

##############

# # More examples to test -1 black sides conjecture

# Polygon([Point(0,0),Point(5,3),Point(4,4),Point(2,5),Point(1,5)])
# p = [[0,0],[5,3],[4,4],[2,5],[1,5]]
# fan(p)
# fan_split(p)

# Polygon([Point(4,2),Point(5,5),Point(0,4),Point(2,1),Point(3,0)])
# p = [[4,2],[5,5],[0,4],[2,1],[3,0]]

# Polygon([Point(6,6),Point(7,5),Point(4,0),Point(2,3),Point(0,7),Point(1,7)])
# [6,6],[7,5],[4,0],[2,3],[0,7],[1,7]]

# Polygon([Point(0,0),Point(5,2),Point(7,3),Point(3,8),Point(1,3)])

# Polygon([Point(8,2),Point(5,4),Point(0,7),Point(0,6),Point(1,3),Point(3,0),Point(6,1)])

# Polygon([Point(0,0),Point(3,1),Point(5,2),Point(3,3),Point(1,3)])

############

# Tring to find a Polygon with 12 fan vectors

# cone = [[1,8],[1,-1]]
# subdivision(cone)

#The Polygon is given by

# Polygon([Point(0,0),Point(1,0),Point(1,10),Point(-8,1)])
# p = [[0,0],[1,0],[1,10],[-8,1]]
# fan(p)
# fan_split(p)

#### 12: Example 2

# vec = [[0,1],[1,-1],[2,5]]
# vec_split(vec)

# vec = [[0,1],[1,-1]]
# vec_split(vec)
# poly_from_fan(vec)
# draw_vec(vec)

#The polygon is

# Polygon([Point(0,0),Point(-12,9),Point(-19,2),Point(-14,0)])
# p = [[0,0],[-12,9],[-19,2],[-14,0]]
# fan(p)

# int_p = [[1,2],[3,1],[17,1],[16,2],[8,8],[7,8]]
# draw_vec(int_p,'blue')

#### 12: Example 3?

# vec = [[-2,-1],[2,-1],[2,1],[2,5]]
# draw_vec(vec,'blue')
# vec_split(vec)

# Polygon([Point(0,0),Point(1,-2),Point(11,-6),Point(3,6)])
# p = [[0,0],[1,-2],[11,-6],[3,6]]
# fan(p)

# vec = [[11,15],[2,1],[7,-11],[-3,-2]]
# draw_vec(vec,'black')
# vec_split(vec)

# cone  = [[11,15],[5,4]]
# vec_split(cone)

# cone = [[1,0],[7,9],[0,1]]
# draw_vec(cone,'black')
# vec_split(cone)

#### Halphen 1

# p = P([0,0,1,0,6,1,8,2,7,5,5,8,1,2])
# fan(p)
# fan_split(p)

# r = P([0,0,1,0,3,1,2,3])
# fanc(r,'blue')

# print(p)

##############

# Always coincide for interior?

# p = P([0,0,4,0,10,7,3,8])
# fan_split(p)

# r = P([2,3,3,1,7,5,7,7,4,7])
# fanc(r,'blue')

# Polygon([Point(0,0),Point(4,0),Point(6,5),Point(3,4)])
# p = P([0,0,4,0,6,5,3,4])
# fan(p)
# fan_split(p)

# p = P([0,0,39,18,45,21,0,16])
# fan(p)
# fan_split(p)
