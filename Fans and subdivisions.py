# import numpy as np
import math
# import matplotlib.pyplot as plt

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

def subdivision(cone):
    start = cone[0]; end = cone[1]
    print("")
    print("Between {},{}".format(start,end))
    fc = end
    ans = [start]
    while (find(cone)[0] != end[0] or find(cone)[1] != end[1]):
        fc = find(cone)
        # if fc[0] != 0:
        #     print("{}, {}".format(fc,(u[0]+v[0])/fc[0]))
        # else:
        # print(fc)
        ans.append(fc)
        cone = [fc,cone[1]]
    ans.append(end)
    
    return ans
    
def psubdivision(cone):
    ans = subdivision(cone)
    i = 1; size = len(ans)
    while i < size-1:
        prev = ans[(i-1)%size]; curr = ans[i]; nextt = ans[(i+1)%size]
        Vector(curr[0],curr[1]).color = 'red'
        if curr[0] != 0:
            print("{},{}".format(curr,(prev[0]+nextt[0])/curr[0]))
        else:
            print("{},{}".format(curr,(prev[1]+nextt[1])/curr[1]))
        i += 1
    print("")

def fanc(polygon, str):
    i = 0; size = len(polygon)
    vectors = []
    print("Fan:")
    while i < size:
        u = polygon[(i-1)%size]; v = polygon[i]; w = polygon[(i+1)%size]
        a = w[0]-v[0]; b = w[1]-v[1]; d = math.gcd(a,b)
        x = u[0]-v[0]; y = u[1]-v[1]
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

def fan_split(polygon):
    vectors = fan(polygon)
    i = 0; size = len(vectors)
    ans = []
    while i < size:
        v = vectors[i]; w = vectors[(i+1)%size]
        psubdivision([v,w])
        i += 1

def fanc_split(polygon,str):
    vectors = fan(polygon,str)
    i = 0; size = len(vectors)
    while i < size:
        v = vectors[i]; w = vectors[(i+1)%size]
        subdivision([v,w])
        i += 1
        
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
Polygon([Point(0,0),Point(2,0),Point(37,28),Point(-1,1)])
polygon = [[0,0],[2,0],[37,28],[-1,1]]
fan(polygon)
fan_split(polygon)

# print("Newton Polygon of R2")

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

# # Polygon([Point(0,0),Point(1,0),Point(3,7),Point(2,5)])
# # np1 = [[0,0],[1,0],[3,7],[2,5]]
# # fanc(np1,'blue')

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

# # NP1 = [[0,0],[2,0],[5,7],[2,3]]
# # fanc(NP1,'green')

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



