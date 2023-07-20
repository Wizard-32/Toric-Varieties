import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter
import turtle
from sympy import Matrix
import array as arr

# Might be slow
def is_square(apositiveint):
  x = apositiveint // 2
  seen = set([x])
  while x * x != apositiveint:
    x = (x + (apositiveint // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True

def add(v,w):
# assumes same length array
    return [v[i] + w[i] for i in range(len(v))]

def add_array(arr, size):
# assumes same "size" length arrays
    if len(arr) == 0:
        return [0 for i in range(size)]
    return add(arr[0],add_array(arr[1:],size))

def scale(k,v):
# assumes same length array
    return [k*v[i] for i in range(len(v))]

def vecToMatrix(vec):
    n = len(vec)
    arr = [[0 for i in range(n)] for j in range(n)]
    for k in range(n):
        arr[k][(k-1) % n] = 1
        arr[k][k] = -vec[k]
        arr[k][(k+1) % n] = 1
    return arr

# 13-14 example (counterclockwise, starting with -1,-2 black edges)
# vec = [3,1,2,2,3,1,2,2,3,2,1,3,2]
# vec = [1, 2, 3, 1, 2, 2, 3, 2, 1, 3, 2, 2]
# arr = vecToMatrix(vec)
# arr = [[1,1],[0,0]]
# A = Matrix(arr)
# print(A.nullspace())

def coefficients(vec):
    n = len(vec)
    v0 = [1,0]; v1 = [0,1]
    ans = [v0,v1]
    for i in range(n):
        next = (i+1) % n
        ans.append(add(scale(vec[next],ans[i+1]),scale(-1,ans[i])))
    # return ans
    M = Matrix([add(ans[n],[-1,0]),add(ans[n+1],[0,-1])])
    return M

def coefficientsCheck(vec, v0, v1):
    result = coefficients(vec)
    for v in result:
        print(add(scale(v[0], v0),scale(v[1],v1)))
        print()


def fan_from_surface(vec, list_of_black):
    n = len(vec)
    v0 = [1,0]; v1 = [0,1]
    recursive_list = [v0,v1]
    ans = []
    for i in range(n):
        next = (i+1) % n
        next_vec = add(scale(vec[next],recursive_list[i+1]),scale(-1,recursive_list[i]))
        recursive_list.append(next_vec)

        if i in list_of_black:
            ans.append(recursive_list[i])
    return ans

def poly_fan(vectors):
    n = len(vectors)
    ans = []
    for i in range(n):
        v = vectors[i]; w = vectors[(i+1)%n]
        sign = 1
        dot = v[1]*w[0] - v[0]*w[1]
        if dot > 0: #TODO: check once
            sign = -1
        perp = scale(sign,[v[1],-v[0]])
        ans.append(perp)
    return ans


def solution_from_fan(fan):
    arr = []
    arr.append([v[0] for v in fan])
    arr.append([v[1] for v in fan])

    scale_factor = arr[0][3]*arr[1][0]
    a = scale(scale_factor,[1,0]); b = scale(scale_factor,[0,1])
    c = scale(-1/arr[0][3],add_array([scale(arr[0][1],a),scale(arr[0][2],b)],2))
    d = scale(-1/arr[1][0], add_array([scale(arr[1][1],a),scale(arr[1][2],b),scale(arr[1][3],c)],2))

    c = [int(c[0]),int(c[1])]; d = [int(d[0]),int(d[1])]
    # a = scale(scale_factor,[1,0]); b = scale(scale_factor,[0,1])
    ans = [a,b,c,d]
    return ans
    
def m_Volume(fan,solution):
    m = add_array([k for k in solution],2)
    a = Matrix(solution[0]); b = Matrix(solution[1]); c = Matrix(solution[2])
    # v1 = [0,0]
    # v2 = [scale(fan[0][1], a),scale(fan[1][1],a)]
    # v3 = [scale(fan[0][2], b),scale(fan[1][2],b)]

    Mab = Matrix([[fan[1][0],fan[2][0]],[fan[1][1],fan[2][1]]]) 
    Mac = Matrix([[fan[1][0],fan[3][0]],[fan[1][1],fan[3][1]]]) 
    Mbc = Matrix([[fan[2][0],fan[3][0]],[fan[2][1],fan[3][1]]]) 

    # print(b)
    Volume_temp = Mab.det()*(a*b.transpose()) + Mac.det()*(a*c.transpose()) + Mbc.det()*(b*c.transpose())
    Volume = [Volume_temp[0],Volume_temp[1]+Volume_temp[2],Volume_temp[3]]
    return [m,Volume,solution]

def final_check(vectors,fan, m_vol):
    m = m_vol[0]; vol = m_vol[1]; sol = m_vol[2]
    m_mat = Matrix(m)
    m_square_temp = m_mat*m_mat.transpose() 
    m_square = [m_square_temp[0], m_square_temp[1]+m_square_temp[2],m_square_temp[3]]

    m_square_minus_vol = add(m_square, scale(-1,vol))

    det = m_square_minus_vol[1]**2-4*m_square_minus_vol[0]*m_square_minus_vol[2] 

    # If negative determinant
    if(det < 0):
        return

    if is_square(det):
       sqroot = int(math.sqrt(det))
       root_temp = [-m_square_minus_vol[1]+sqroot,2*m_square_minus_vol[0]]
       gcd = math.gcd(root_temp[0],root_temp[1])
       root = [int(root_temp[0]/gcd), int(root_temp[1]/gcd)]
       a = sol[0]; b = sol[1]; c = sol[2]; d = sol[3]
       a = a[0] * root[0]; b = b[1] * root[1]
       c = c[0] * root[0] + c[1] * root[1]
       d = d[0] * root[0] + d[1] * root[1]
       if a < 0 or b < 0 or c < 0 or d < 0:
        #    print('NO')
        #    print("Equation {}a^2+({})ab+({})b^2 has a root a/b = {}, but c or d negative".format(m_square_minus_vol[0],m_square_minus_vol[1],m_square_minus_vol[2],root))
           return
       print("Yess!!")
       print([a,b,c,d])
       print("Equation {}a^2+({})ab+({})b^2 has a root a/b = {}".format(m_square_minus_vol[0],m_square_minus_vol[1],m_square_minus_vol[2],root))
       print("Vertices:")
       A = [0,0]
       B = add(A, scale(a,[fan[1][0],fan[1][1]]))
       C = add(B, scale(b,[fan[2][0],fan[2][1]]))
       D = add(C, scale(c,[fan[3][0],fan[3][1]]))
       vertices = [A,B,C,D]
       print(vertices)
       return 1
    else:
        return
    

def vec_check(vec,list_of_black):
    vectors = fan_from_surface(vec, list_of_black)
    # print(vectors)
    fan = poly_fan(vectors) 
    # print(fan)
    solution = solution_from_fan(fan)
    # print(solution)
    m_vol = m_Volume(fan,solution)
    # print(m_vol)
    ans = final_check(vectors,fan, m_vol)
    if ans == 1:
        print("List of black:")
        print(list_of_black)
        print("Vectors:")
        print(vectors)
        print("Toric Fan:")
        print(fan)
        print("Solution")
        print(solution)

# M = Matrix([1,2])
# N = Matrix([0,1])
# print(M * N.transpose())

# vec = [1, 2, 3, 1, 2, 2, 3, 2, 1, 3, 2, 2]

# list = list_of_black[2]
# list = [0, 2, 3, 8]
# vec_check(vec,list)
# vectors = fan_from_surface(vec, list_of_black)
# # print(fan)
# fan = poly_fan(vectors)
# solution = solution_from_fan(fan)
# # print(solution)
# m_vol = m_Volume(fan,solution)
# final_check(m_vol)

# print(coefficients([2,1,3,1]))
# print(coefficients(vec))
# coefficientsCheck(vec,[1,0],[0,1])
# arr = [[0 for i in range(12)],[0 for i in range(12)]]

# arr has positive entries for simplication
def contract(arr):
    ans = arr.copy()
    while 1 in ans:
        i = ans.index(1)
        next = (i+1) % len(ans); prev = (i-1) % len(ans)
        ans[next] -= 1
        if prev != next:
            ans[(i-1) % len(ans)] -= 1
        del ans[i]

    return scale(-1, ans)

# arr1 = [1,2,2,2,2,2,4,2,1,3,2,2,2,1,2,6]
# contract(arr1)

# arr2 = [1,2,2,3,1,1,2,2,3,2,2,2]
# contract(arr2)

# arr3 = [1,1,2,2,3,3,2,2,2,2,2,2]
# # print(len(arr3))
# contract(arr3)

# a = [2,2,2,2,2,2,2,2,2,2,2,2]

##### Check A212

# def check(arr, i1, i2, j, k):
#     arr[i1] = 1; arr[i2] = 1
#     arr[j] = 3; arr[k] = 3; 
#     carr = contract(arr)
#     if carr == [1,1,1]:
#         print(arr)

# for i1 in range(11):
#     for i2 in range(i1+1, 12):
#         for j in range(11):
#             if j != i1 and j != i2:
#                 for k in range(j+1, 12):
#                     if k != i1 and k != i2:
#                         # temp = a
#                         check([2,2,2,2,2,2,2,2,2,2,2,2], i1, i2, j, k)

##### Check A214

def check214(arr, i1, i2, i3, i4, j, k):
    arr[i1] = 1; arr[i2] = 1; arr[i3] = 1; arr[i4] = 3
    arr[j] = 3; arr[k] = 3; 
    carr = contract(arr)
    if carr == [1,1,1]:
        # print(arr)
        list_of_black = []
        for count in range(3):
            temp = []
            which_three = count
            added_three = -1
            for i in range(len(arr)):
                if arr[i] == 1:
                    temp.append(i)
                if arr[i] == 3:
                    added_three += 1
                    if added_three == which_three:
                        temp.append(i)
            list_of_black.append(temp)
        # print(list_of_black)
        for list in list_of_black:
            vec_check(arr,list)

# print(list_of_black)


# for i1 in range(9):
i1 = 0 
for i2 in range(i1+1, 10):
    for i3 in range(i2+1, 11):
        for i4 in range(i3+1, 12):
            for j in range(11):
                if j != i1 and j != i2 and j != i3 and j != i4:
                    for k in range(j+1, 12):
                        if k != i1 and k != i2 and k!= i3 and k!= i4:
                            # temp = a
                            check214([2,2,2,2,2,2,2,2,2,2,2,2], i1, i2, i3, i4, j, k)
                
# # a = [0,1,1,3,3,1,2,5,1,3,1,3]
# # print(contract(a))2