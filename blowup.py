import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter
import turtle

def add(v,w):
# assumes same length array
    return [v[i] + w[i] for i in range(len(v))]

def scale(k,v):
# assumes same length array
    return [k*v[i] for i in range(len(v))]

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
    Tw = np.matmul(T,w)
    sign = 1
    if Tw[0] == 0:
        return None
    if Tw[0] < 0:
        sign = -1

    if Tw[1] == 0:
        return w
    
    invT = [[bezout(x1,y1)[2],x1],[-bezout(x1,y1)[1], y1]] 
    # We are dealing with a convex cone so TW[0] ! -1

    # if Tw[1] < 0:
    #     return np.matmul(invT,[sign,0]) 
    # if Tw[1] > 0:
    #     k = math.ceil(sign*Tw[1]/Tw[0])
    #     # invT = np.linalg.inv(T)
    #     temp = [sign,k]
    #     return np.matmul(invT,temp)

    k = math.ceil(sign*Tw[1]/Tw[0])
    # invT = np.linalg.inv(T)
    temp = [sign,k]
    return np.matmul(invT,temp) 
    
# Creates subdivision of a cone to resolve the singularity
# Also takes input an R fan to test for collisions ?? maybe
# Last argument: in_R used to tell if it is in R or not
def subdivision(cone, arr):
    start = cone[0]; end = cone[1]
    print("")
    print("Between {},{}".format(start,end)) # Prints "Between"
    fc = end
    ans = [start] # ans list starts with u (first cone vertex)
    while (find(cone)[0] != end[0] or find(cone)[1] != end[1]):
        fc = find(cone)

        in_R = 0
        for k in arr:
            if fc[0] == k[0] and fc[1] == k[1]:
                in_R = 1

        ans.append([int(fc[0]),int(fc[1]),in_R])
        cone = [fc,end]
    ans.append(end) # ans list starts with v (second cone vertex)

    i = 1; size = len(ans)

# Now we add the self intersection numbers. Note x means C^2 = -x
# The array sizes would be 3 for these resolution vectors, helping us identify if this is an 
# exceptional divisior or a boundary one 
    while i < size-1:
        prev = ans[(i-1)%size]; curr = ans[i]; nextt = ans[(i+1)%size]
        if curr[0] != 0:
            self_intersection = (int)((prev[0]+nextt[0])/curr[0] )
            print("{},{}".format(curr,self_intersection))
            ans[i] = [int(curr[0]), int(curr[1]), self_intersection, curr[2]]
        else:
            self_intersection = (int)((prev[1]+nextt[1])/curr[1])
            print("{},{}".format(curr,self_intersection))
            ans[i] = [int(curr[0]), int(curr[1]), self_intersection, curr[2]]
        i += 1
    print("") #Empty line
    
    return ans[1:len(ans)-1] #Returns the array of *exceptional vectors* not the boundary ones

### More algorithmic and recursive way to resolve; has a few bugs

# def find_mu_lam(cone):
#     v = cone[0]; 
#     if bezout(v[0],v[1])[0] != 1:
#         raise Exception("Coordinates must be coprime!")
#     # assuming that v has coprime coordinates, we set v = f
# # f 
#     f = [v[0],v[1]]
#     w = cone[1]; x = w[0]; y = w[1]
#     k = bezout(f[0],f[1])[1]; l = bezout(f[0],f[1])[2]
#     dot = x*l + y*k
#     if dot < 0:
#         sign = -1
# # e, lam, mu
#     e = [sign*l, sign*k]
#     lam = w[0]*f[1]-w[1]*f[0]
#     mu = (w[0]*e[1]-w[1]*e[0])
#     # Vector(f[0],f[1])
#     # Vector(e[0],e[1])
#     # Vector(w[0],w[1])
#     return [f,e,lam,mu]
    
# v = [2,-1]
# w = [11,-7]
# # print(find_mu_lam([v,w]))

# def resolve(cone):
#     u = cone[0]; v = cone[1]
#     # ea = []; fa = []; mua = [-1]; lama = []; sigma = []
#     r = find_mu_lam(cone)
#     fa = [[0,1]]; ea = [[1,0]]; lama = [r[2]]; mua = [r[3]]; ka = [math.ceil(r[2]/r[3])]; ua = [u]
#     i = 0
#     print(u)
#     # i = 0; sigma.append(cone); ea.append(0)
#     while mua[i] != 0:
#         fa.append(ea[i])
#         ea.append(add(scale(-1,fa[i]),scale(ka[i],ea[i])))
#         lama.append(mua[i])
#         mua.append(ka[i]*mua[i]-lama[i])
#         ua.append(ea[i])
#         if mua[i+1] != 0:
#             ka.append(math.ceil(lama[i+1]/mua[i+1]))
#         print("{},{}".format(ua[i+1],ka[i]))
            
#         # Vector(ua[i+1][0],ua[i+1][1],color = 'red')

#         i += 1
#     ua.append(ea[i])
#     print(v)

### Creates the normal fan of a polygon
### NOTE: ORDER OF VERTICES MATTERS
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

        ans = [int(sign*v[1]), int(-sign*v[0])]
        vectors.append(ans)
        print(ans)
        s = "({},{})".format(ans[0],ans[1])
#IF WANT TO PLOT
        # plt.arrow(0,0,ans[0],ans[1],width=0.05,head_width=0.2,color = str) 
        i += 1
    return vectors

### Creates the normal fan of a polygon
### NOTE: ORDER OF VERTICES MATTERS
### Take array of R as input too
def fanc_with_arr(polygon, str, arr):
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
        temp = [int(sign*v[1]), int(-sign*v[0])]

        in_R = 0
        if temp in arr:
            in_R = 1
        ans = [temp[0], temp[1], in_R]

        vectors.append(ans)
        print(ans)
        s = "({},{})".format(ans[0],ans[1])
#IF WANT TO PLOT
        # plt.arrow(0,0,ans[0],ans[1],width=0.05,head_width=0.2,color = str) 
        i += 1
    return vectors

## Creates a black colored fan of a polygon

def fan(polygon):
    return fanc(polygon, 'black')

def fan_split(polygon, R_arr):
    vectors = fanc_with_arr(polygon, 'black', R_arr)
    i = 0; size = len(vectors)
    list = []
    while i < size:
        v = vectors[i]; w = vectors[(i+1)%size]
        list.append([int(v[0]),int(v[1]), v[2]])
        print("")
        # print("Between {},{}".format(v, w))
        arr = subdivision([v,w], R_arr)
        # arr.append("Start") #marks where each resolution starts
        for elem in arr:
            list.append(elem)
        # arr.append("End")
        print("")
        i += 1
    return list

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

# polygon = [[1,0],[0,1],[13,8],[3,0]]
# fan_split(polygon)

# plt.show()

# print(find([[-7,2],[12,-5]]))

# NOTE: ORDER OF VERTICES MATTERS!
def draw(poly, R_arr):
    tt = turtle.Turtle()
    drawing_area = turtle.Screen()
    tt.screen.screensize(2000,1500)
    list = fan_split(poly, R_arr)
    n = len(list)
    for a in list:
        if len(a) == 3:
            tt.color('black')
            tt.width(4)
            tt.backward(20)
            tt.forward(20)
    #Create dot in center
            if a[2] == 1:
                tt.forward(50)
                tt.dot(10)
                tt.forward(50)
            else:
                tt.forward(100)    
        if len(a) == 4:
            tt.width(1)
            tt.color('blue')
            tt.backward(20)

            # Use this for continuous lines
            tt.forward(20)
            tt.forward(50)
            if a[3] == 1:
                tt.dot(10)
            tt.write(a[2], font=("Arial", 20, "bold"), align = "right")
            tt.forward(50)

            ## Use this for a gap in the line (to write text)
            # tt.forward(40)
            # tt.penup()
            # tt.forward(10)
            # tt.write(a[2], font=("Arial", 20, "bold"), align = "right")
            # tt.forward(10)
            # tt.pendown()
            # tt.forward(40)
        tt.forward(20)
        tt.backward(20)
        tt.left(360/n)
    turtle.done()

def full_auto(poly,r_poly):
    r_fan = fanc(r_poly,'blue')
    draw(poly,r_fan)

################

## (q) example 7

# poly = [[3,0],[0,3],[46,16],[6,0]]
# r1 = [[0,0],[3,1]]
# r2 = [[1, 0], [11, 4], [7, 3], [0, 1]]

# full_auto(poly,r2)

################

# # (q) example 4

# Polygon([Point(0,0),Point(-1,1),Point(37,28),Point(2,0)])
# poly = [[0,0],[-1,1],[37,28],[2,0]]

# r1 = [[0,0],[1,1]]
# r2 = [[7,5],[2,1],[0,0]]
# # r1_fan = fanc(r1_poly,'blue')
# full_auto(poly, r1)

# print("Newton Polygon of R2")

# Polygon([Point(7,5),Point(2,1),Point(0,0)])
# r2_poly = [[7,5],[2,1],[0,0]]
# r2_fan = fanc(r2_poly,'blue')
# draw(poly, r2_fan)


################

# # Special (13,14) example

# Polygon([Point(0,0),Point(4,0),Point(5,1),Point(15,36)])
# poly = [ [0 , 0] , [4 , 0] , [5 , 1] , [15 , 36] ]

# print("Newton Polygon of R1")

# np1 = [[0,0],[1,0],[3,7],[2,5]]
# np1_fan = fanc(np1,'blue')
# print('')

# print("Main fan")
# # fan_split(poly,R_fan)
# draw(poly, np1_fan)

# print("Newton Polygon of R2")

# np2 = [[1,0],[3,0],[4,1],[5,4],[11,25],[1,1]]
# np2_fan = fanc(np2,'blue')
# print('')

# print("Main fan")
# # fan_split(poly,R_fan)
# draw(poly, np2_fan)

# fanc(np2,'blue')

################

# # Random Example

# poly = [[0,0],[5,0],[11,14],[11,15]]

# R1_poly = [[0,0],[1,1]]
# print("R fan:")
# R1_fan = fanc(R1_poly,'black')
# print('')

# print("Main fan")
# # fan_split(poly,R_fan)
# draw(poly, R1_fan)

# R2_poly = [[0,0],[2,0],[5,7],[2,3]]
# print("R fan:")
# R2_fan = fanc(R2_poly,'black')
# print('')

# print("Main fan")
# # fan_split(poly,R_fan)
# draw(poly, R2_fan)


################

# # One curve covers all -k sides

# poly = [ [10 , 0] , [0 , 20 ], [27 , 5] , [12 , 0] ]
# r1 = [[0,0],[1,0]]
# r2 = [ [8, 0], [9, 0], [21, 4], [20, 5], [2, 15], [0, 16] ]

# # full_auto(poly,r1)
# full_auto(poly,r2)
