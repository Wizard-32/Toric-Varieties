import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter
import turtle
from sympy import Matrix
import array as arr
import itertools # For cartesian product of for loops

def gcd(a,b):
    if a > 0 and b > 0:
        return math.gcd(a,b)
    if a < 0 and b < 0:
        return -math.gcd(a,b)

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
    
    if recursive_list[n] != [1,0] or recursive_list[n+1] != [0,1]:
        print("Something is wrong in the fan")
        print("Recursive list")
        print(recursive_list)

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
    # print(arr)

    scale_factor = arr[0][3]*arr[1][0]
    a = scale(scale_factor,[1,0]); b = scale(scale_factor,[0,1])
    c = scale(-1/arr[0][3],add_array([scale(arr[0][1],a),scale(arr[0][2],b)],2))
    d = add_array([scale(arr[1][1],a),scale(arr[1][2],b),scale(arr[1][3],c)],2)

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

    Volume_temp = Mab.det()*(a*b.transpose()) + Mac.det()*(a*c.transpose()) + Mbc.det()*(b*c.transpose())
    Volume = [Volume_temp[0],Volume_temp[1]+Volume_temp[2],Volume_temp[3]]
    # print(Volume)

    return [m,Volume,solution]

def final_check(vectors,fan, m_vol):
    m = m_vol[0]; vol = m_vol[1]; sol = m_vol[2]
    m_mat = Matrix(m)
    m_square_temp = m_mat*m_mat.transpose() 
    m_square = [m_square_temp[0], m_square_temp[1]+m_square_temp[2],m_square_temp[3]]
    m_square_minus_vol = add(m_square, scale(-1,vol))

    A = m_square_minus_vol[0]; B = m_square_minus_vol[1]; C = m_square_minus_vol[2]
    disc = B**2-4*A*C

    # print("Equation {}a^2+({})ab+({})b^2, disc = {}".format(A,B,C,disc))
        
    # If negative determinant
    if(disc < 0):
        return

    if is_square(disc):
        sqroot = int(math.sqrt(disc))
    # quadratic is Aa^2+Bab+Cb^2=0, discriminant is sqroot
    # the root of this is a/b = (-B +- sqroot)/2A, 
    # so we will set a as the (reduced) numerator, and b as the (reduced) denominator
        # print("A = {}, B = {}, C = {}, sqroot = {}".format(A,B,C,sqroot))
        roots = []
        if (-B + sqroot)/A > 0:    
            d = gcd(-B + sqroot, 2*A)
            roots.append([int((-B + sqroot)/d),int((2*A)/d)])
        if (-B - sqroot)/A > 0:
            d = gcd(-B - sqroot, 2*A)
            roots.append([int((-B - sqroot)/d),int((2*A)/d)])
        if len(roots) == 0:
            return #This is when the root is zero (i.e. A = 0) or both roots are negative

        # print("Roots: {}".format(roots))

        root = [] # this will be the root we select
        
        # TODO: Both roots might work?
        a = sol[0]; b = sol[1]; c = sol[2]; d = sol[3]
        for potential_root in roots:
            aa = a[0] * potential_root[0] + a[1] * potential_root[1]
            bb = b[0] * potential_root[0] + b[1] * potential_root[1]
            cc = c[0] * potential_root[0] + c[1] * potential_root[1]
            dd = d[0] * potential_root[0] + d[1] * potential_root[1]
            if aa > 0 and bb > 0 and cc > 0 and dd > 0:
                root = sol
                break

        if len(root) == 0:
            return
        
        # if (-B + sqroot)/A > 0:    
        #     root = [-B + sqroot,2*A]
        #     c = c[0] * root[0] + c[1] * root[1]
        #     d = d[0] * root[0] + d[1] * root[1]
        #     if c < 0 or d < 0:
                
        # elif (-B - sqroot)/A > 0:
        #     root = [-B - sqroot, 2*A]
        # else:
        #     return #This is when the root is zero (i.e. A = 0) or both roots are negative
    
    # Using custom gcd function so that get positive numbers
        # root = [[int(elem[0]/gcd()), int(elem[1]/GCD)] for elem in root_temp]
        # GCD1 = gcd(root_temp1[0],root_temp1[1])
        # GCD2 = gcd(root_temp2[0],root_temp2[1])
        # # root = [int(root_temp1[0]/GCD), int(root_temp[1]/GCD)]
        # if root[0] < 0 and root[1] < 0:
        #     print("PLEASE CHECK SOME ISSUE HAS OCCURED")
    # Now a = root[0], b = root[1]
        # print("Equation {}a^2+({})ab+({})b^2 has a root a/b = {}".format(m_square_minus_vol[0],m_square_minus_vol[1],m_square_minus_vol[2],root))
        
        # a = sol[0]; b = sol[1]; c = sol[2]; d = sol[3]
        # a = a[0] * root[0]; b = b[1] * root[1]
        # c = c[0] * root[0] + c[1] * root[1]
        # d = d[0] * root[0] + d[1] * root[1]

        print("Yess!!")
        # print("[a,b,c,d] = [{},{},{},{}]".format(a,b,c,d))
        # print("Equation {}a^2+({})ab+({})b^2 has a root a/b = {}".format(m_square_minus_vol[0],m_square_minus_vol[1],m_square_minus_vol[2],root))
        # print("Vertices:")
        A = [0,0]
        B = add(A, scale(aa,[fan[1][0],fan[1][1]]))
        C = add(B, scale(bb,[fan[2][0],fan[2][1]]))
        D = add(C, scale(cc,[fan[3][0],fan[3][1]]))
        vertices = [A,B,C,D]
        # print(vertices)

        GCD_vertices = math.gcd(B[0],B[1],C[0],C[1],D[0],D[1])
        A_new = [0,0]
        B_new = [int(B[0]/GCD_vertices),int(B[1]/GCD_vertices)]
        C_new = [int(C[0]/GCD_vertices),int(C[1]/GCD_vertices)]
        D_new = [int(D[0]/GCD_vertices),int(D[1]/GCD_vertices)]
        print("(Reduced) Vertices: [{},{},{},{}]".format(A_new,B_new,C_new,D_new))
        return 1
    else:
        return
    
# NOTE: ASSUMES FIRST VERTEX IS BLACK !!
# i.e., list_of_black[0] = 0
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
    # if ans == 1:
    #     print("List of black:")
    #     print(list_of_black)
    #     print("Vectors:")
    #     print(vectors)
    #     print("Toric Fan:")
    #     print(fan)
    #     print("Solution")
    #     print(solution)
    #     print("Perimeter")
    #     print(m_vol[0])
    #     print("Volume")
    #     print(m_vol[1])


# vec = [1,2,2,2,3,2,2,2,1,5,2,2,1,2,4]
# # print(len(vec))
# list = [0,8,12,14]

# vec = [4,1,2,2,2,3,2,2,2,1,5,2,2,1,2]
# # print(len(vec))
# list = [0,1,9,13]

# vec = [1,3,4,1,2,2,2,3,2,1,3,2,2,2]
# list = [0,3,9,12]

# vec = [3,1,2,4,2,6,1,2,2,2,2,3,2,1,2,2,2]
# list = [0,1,6,13]

# print(vec)
# vec_check(vec, list)

# vec2 = [1,2,4,2,6,1,2,2,2,2,3,2,1,2,2,2,3]
# list2 = [0,5,12,16]

# print(vec2)
# vec_check(vec2, list2)

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

##### Check A, ending at P^2

def checkA1(i1,i2,i3,i4,i5,i6):
    arr = [2,2,2,2,2,2,2,2,2,2,2,2]
    arr[i1[0]] = i1[1]; arr[i2[0]] = i2[1]; arr[i3[0]] = i3[1]
    arr[i4[0]] = i4[1]; arr[i5[0]] = i5[1]; arr[i6[0]] = i6[1] 
    carr = contract(arr)
    # End at P^2
    if carr == [1,1,1]:
        # print(arr)
        # list_of_black = []
        # for count in range(3):
        #     temp = []
        #     which_three = count
        #     added_three = -1
        #     for i in range(len(arr)):
        #         if arr[i] == 1:
        #             temp.append(i)
        #         if arr[i] == 3:
        #             added_three += 1
        #             if added_three == which_three:
        #                 temp.append(i)
        #     list_of_black.append(temp)
        print("arr ",arr)
        # print("black edges: ",list_of_black)
        # for list in list_of_black:
        #     vec_check(arr,list)
        # return
    if len(carr) == 4:
        if (carr[0] == 0 and carr[2] == 0 and carr[1] + carr[3] == 0):
            print("arr ending at F{}: {}".format(abs(carr[1]),arr))
        elif (carr[1] == 0 and carr[3] == 0 and carr[0] + carr[2] == 0):
            print("arr ending at F{}: {}".format(abs(carr[0]),arr))
        return

def checkA2(i1,i2,i3,i4,i5,i6):
    arr = [2,2,2,2,2,2,2,2,2,2,2,2]
    arr[i1[0]] = i1[1]; arr[i2[0]] = i2[1]; arr[i3[0]] = i3[1]
    arr[i4[0]] = i4[1]; arr[i5[0]] = i5[1]; arr[i6[0]] = i6[1] 
    carr = contract(arr)
    # End at P^2
    if carr == [1,1,1]:
        # print(arr)
        
        # print("arr ",arr)
        # print("black edges: ",list_of_black)
        # for list in list_of_black:
        #     vec_check(arr,list)
        # return
        list_of_black = []
        for i in range(len(arr)):
            if arr[i] == 1 or arr[i] == 4:
                list_of_black.append(i)
        # print("black edges: ",list_of_black)
        vec_check(arr,list_of_black)


    elif len(carr) == 4:
        if (carr[0] == 0 and carr[2] == 0 and carr[1] + carr[3] == 0):
            # print("arr ending at F{}: {}".format(abs(carr[1]),arr))
            list_of_black = []
            for i in range(len(arr)):
                if arr[i] == 1 or arr[i] == 4:
                    list_of_black.append(i)
            # print("black edges: ",list_of_black)
            vec_check(arr,list_of_black)

        elif (carr[1] == 0 and carr[3] == 0 and carr[0] + carr[2] == 0):
            # print("arr ending at F{}: {}".format(abs(carr[0]),arr))
            list_of_black = []
            for i in range(len(arr)):
                if arr[i] == 1 or arr[i] == 4:
                    list_of_black.append(i)
            # print("black edges: ",list_of_black)
            vec_check(arr,list_of_black)

def checkA3(i1,i2,i3,i4,i5,i6):
    arr = [2,2,2,2,2,2,2,2,2,2,2,2]
    count = 0
    arr[i1[0]] = i1[1]; arr[i2[0]] = i2[1]; arr[i3[0]] = i3[1]
    arr[i4[0]] = i4[1]; arr[i5[0]] = i5[1]; arr[i6[0]] = i6[1] 
    carr = contract(arr)
    # End at P^2
    if carr == [1,1,1]:
        print("arr ",arr)
        count += 1
        # print("black edges: ",list_of_black)
        # for list in list_of_black:
        #     vec_check(arr,list)
        # return
        # list_of_black = []
        # for i in range(len(arr)):
        #     if arr[i] == 1 or arr[i] == 5:
        #         list_of_black.append(i)
        # # print("black edges: ",list_of_black)
        # vec_check(arr,list_of_black)


    elif len(carr) == 4:
        if (carr[0] == 0 and carr[2] == 0 and carr[1] + carr[3] == 0):
            print("arr ending at F{}: {}".format(abs(carr[1]),arr))
            # list_of_black = []
            # for i in range(len(arr)):
            #     if arr[i] == 1 or arr[i] == 5:
            #         list_of_black.append(i)
            # # print("black edges: ",list_of_black)
            # vec_check(arr,list_of_black)
            count += 1

        elif (carr[1] == 0 and carr[3] == 0 and carr[0] + carr[2] == 0):
            print("arr ending at F{}: {}".format(abs(carr[0]),arr))
            # list_of_black = []
            # for i in range(len(arr)):
            #     if arr[i] == 1 or arr[i] == 5:
            #         list_of_black.append(i)
            # # print("black edges: ",list_of_black)
            # vec_check(arr,list_of_black)
            count += 1

def checkers(i1,i2,i3,i4,i5,i6):
    arr = [2 for i in range(40)]

    arr[i1[0]] = i1[1]; arr[i2[0]] = i2[1]; arr[i3[0]] = i3[1]
    arr[i4[0]] = i4[1]; arr[i5[0]] = i5[1]; arr[i6[0]] = i6[1] 
    carr = contract(arr)
    # End at P^2
    if carr == [1,1,1]:
        print("arr ",arr)
        return 1
        # print("black edges: ",list_of_black)
        # for list in list_of_black:
        #     vec_check(arr,list)
        # return
        # list_of_black = []
        # for i in range(len(arr)):
        #     if arr[i] == 1 or arr[i] == 5:
        #         list_of_black.append(i)
        # # print("black edges: ",list_of_black)
        # vec_check(arr,list_of_black)


    elif len(carr) == 4:
        if (carr[0] == 0 and carr[2] == 0 and carr[1] + carr[3] == 0):
            print("arr ending at F{}: {}".format(abs(carr[1]),arr))
            return 1 
            # list_of_black = []
            # for i in range(len(arr)):
            #     if arr[i] == 1 or arr[i] == 5:
            #         list_of_black.append(i)
            # # print("black edges: ",list_of_black)
            # vec_check(arr,list_of_black)

        elif (carr[1] == 0 and carr[3] == 0 and carr[0] + carr[2] == 0):
            print("arr ending at F{}: {}".format(abs(carr[0]),arr))
            return 1
            # list_of_black = []
            # for i in range(len(arr)):
            #     if arr[i] == 1 or arr[i] == 5:
            #         list_of_black.append(i)
            # # print("black edges: ",list_of_black)
            # vec_check(arr,list_of_black)
    return 0
# # # print(list_of_black)

i1 = 0; i2=0; i3=0; i4=0; j=0; k=0 
count = 0
for i2, i3, i4, j, k in itertools.product(range(i1+1, 10), range(i2+1, 11), range(i3+1, 12), range(11), range(j+1, 12)):
   if j != i1 and j != i2 and j != i3 and j != i4 and k != i1 and k != i2 and k!= i3 and k!= i4:
    # checkA1([i1,1], [i2,1], [i3,1], [i4,3], [j,3], [k,3])
    # checkA2([i1,1], [i2,1], [i3,1], [i4,4], [j,2], [k,3]) 
    count += checkers([i1,1], [i2,1], [i3,1], [i4,32], [j,2], [k,3]) 
print(count)

# print(contract([1, 2, 2, 3, 2, 1, 3, 3, 2, 1, 2, 2]))

# i1 = 0; i2=0; i3=0; i4=0; j=0; k=0 
# for i2, i3, i4, j, k in itertools.product(range(i1+1, 10), range(i2+1, 11), range(i3+1, 12), range(11), range(j+1, 12)):
#    if j != i1 and j != i2 and j != i3 and j != i4 and k != i1 and k != i2 and k!= i3 and k!= i4:
#     checkA1([i1,1], [i2,1], [i3,2], [i4,2], [j,3], [k,3])
# i1 = 0      
# for i2 in range(i1+1, 10):
#     for i3 in range(i2+1, 11):
#         for i4 in range(i3+1, 12):
#             for j in range(11):
#                 if j != i1 and j != i2 and j != i3 and j != i4:
#                     for k in range(j+1, 12):
#                         if k != i1 and k != i2 and k!= i3 and k!= i4:
#                             # temp = a
#                             check214([2,2,2,2,2,2,2,2,2,2,2,2], i1, i2, i3, i4, j, k)
                
# # a = [0,1,1,3,3,1,2,5,1,3,1,3]
# # print(contract(a))2


################### EXPAND METHOD

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

#     def __repr__(self):
#         return self.data

# class LinkedList:
#     def __init__(self):
#         self.head = None

#     def __repr__(self):
#         node = self.head
#         nodes = []
#         while node is not None:
#             nodes.append(node.data)
#             node = node.next
#         nodes.append("None")
#         return " -> ".join(nodes)

# def print_linked(lst):
#     curr = lst.head
#     arr = []
#     while curr != None:
#         arr.append(curr.data)
#         curr = curr.next
#     print(arr)

# # def linked_len(lst):
# #     curr = lst.head
# #     i = 0
# #     while curr != None:
# #         i += 1
# #         curr = curr.next
# #     return i

# # llst = LinkedList()
# # llst.head = Node("1")
# # llst.head.next = Node("2")
# # llst.head.next.next = Node("5")
# # print_linked(llst)

# def copy(llst):
#     copy = LinkedList()
#     copy.head.data = Node("")
#     curr_llst = llst.head
#     curr_copy = copy.head
#     while curr_llst != None:
#         curr_copy.data = Node(curr_llst.data)
#         curr_llst = curr_llst.next
#         curr_copy = curr_copy.next
#     return copy

# def expand(llst, len):
#     if len == 5:
#         print_linked(llst)
#         return
#     else:
#         for k in range(len):
#             if k == 0:
#                 llst.head.data -= 1

#                 curr = llst.head
#                 while curr.next != None:
#                     curr = curr.next
#                 curr.data -= 1

#                 newNode = Node(-1)
#                 newNode.next = llst.head
#                 llst.head = newNode
#             else:
#                 count = 1
#                 curr = llst.head
#                 while count != k:
#                     curr = curr.next
#                     count += 1
#                 if curr.next == None:
#                     llst.head.data -= 1
#                 else:
#                     curr.next.data -= 1
#                 curr.data -= 1

#                 newNode = Node(-1)
#                 newNode.next = curr.next 
#                 curr.next = newNode
#             expand(llst, len+1)
            
# P2 = LinkedList()
# P2.head = Node(1)
# P2.head.next = Node(1)
# P2.head.next.next = Node(1)
# print_linked(copy(P2))
# expand(P2,3)
