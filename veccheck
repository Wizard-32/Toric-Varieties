import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter
import turtle
from sympy import Matrix
import array as arr
import itertools # For cartesian product of for loops
import sys

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

# Invert 2x2 matrices
def invertMat(M):
    x = M[0]; y = M[1]; z = M[2]; w = M[3]
    det = x*w-y*z
    if det == 0:
        print("Not invertible")
        return 
    return [det, Matrix([[w,-y],[-z,x]])]

# M = Matrix([1,2,3,4])
# N = invertMat(M)[1]
# print(N[2])

#### NOTE: arguments are with the correct signs, i.e. we contract (-1)s
def contract(arr):
    ans = arr.copy()
    while -1 in ans:
        i = ans.index(-1)
        next = (i+1) % len(ans); prev = (i-1) % len(ans)
        ans[next] += 1
        if prev != next:
            ans[(i-1) % len(ans)] += 1
        del ans[i]

    return ans

def coefficients(vec):
    n = len(vec)
    v0 = [1,0]; v1 = [0,1]
    ans = [v0,v1]
    for i in range(n):
        next = (i+1) % n
    # return ans
    M = Matrix([add(ans[n],[-1,0]),add(ans[n+1],[0,-1])])
    return M

def coefficientsCheck(vec, v0, v1):
    result = coefficients(vec)
    for v in result:
        print(add(scale(v[0], v0),scale(v[1],v1)))
        print()

# Prints the toric fan given self intersection numbers
def fan_from_surface(vec, list_of_black):
    n = len(vec)
    v0 = [1,0]; v1 = [0,1]
    recursive_list = [v0,v1]
    ans = []
    for i in range(n):
        next = (i+1) % n
        next_vec = add(scale(-vec[next],recursive_list[i+1]),scale(-1,recursive_list[i]))
        recursive_list.append(next_vec)

        if i in list_of_black:
            ans.append(recursive_list[i])
    
    if recursive_list[n] != [1,0] or recursive_list[n+1] != [0,1]:
        print("Something is wrong in the fan")
        # print("Recursive list")
        # print(recursive_list)
        return -1
    print(ans)
    return ans


# Prints the vector of the M = Hom(N, ZZ) lattice
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

# print(poly_fan([[-31, 37], [-1, 1], [-1, -1], [2, -1]]))

# Chooses a basis u = [1,0], v = [0,1] and puts a,b,c,d in terms of these basis (as a Z module)
def solve_fan(fan):
    M = Matrix([[fan[0][0],fan[1][0]],[fan[0][1],fan[1][1]]])
    N = Matrix([[-fan[2][0],-fan[3][0]],[-fan[2][1],-fan[3][1]]]) 
    d = invertMat(N)[0]; Ni = invertMat(N)[1]
    a = [d,0]; b = [0,d] 
    c = [(Ni * M)[0], (Ni * M)[1]]
    d = [(Ni * M)[2], (Ni * M)[3]]
    # print([a,b,c,d])
    return [a,b,c,d] # Call this sol

# poly_fan = poly_fan([[-31, 37], [-1, 1], [-1, -1], [2, -1]]) 


def latticepts(sol):
    return add_array([sol[0],sol[1],sol[2],sol[3]],2)

# At this point we need a way to multiply elements of the basis, for instead (2u+v)(3u+v)=6u^2 + 5uv + v^2
# This should be given by auxmult([2,1],[3,1])=[6,5,1]
# auxmult stands for auxilliary multiplication
# NOTE: output is an array not a matirx, so can use the scale function on it
def auxmult(u,v):
    return [u[0]*v[0],u[0]*v[1]+u[1]*v[0],u[1]*v[1]]

def volume(fan, sol):
    a = sol[0]; b = sol[1]; c = sol[2]; d = sol[3]
    M = Matrix([[fan[0][0],fan[1][0]],[fan[0][1],fan[1][1]]])
    N = Matrix([[-fan[2][0],-fan[3][0]],[-fan[2][1],-fan[3][1]]])
    dm = M.det(); dn = N.det()

    return add(scale(dm,auxmult(a,b)),scale(dn,auxmult(c,d)))

# print(latticepts(sol))
# print(vol(poly_fan,sol))

def check_good(fan, sol):
    m = latticepts(sol) 
    vol = volume(fan,sol)

    eq = add(vol,scale(-1,auxmult(m,m)))
    A = eq[0]; B = eq[1]; C = eq[2]
    print("Equation is {}u^2 + {}uv + {}v^2 = 0".format(A,B,C))

    disc = B^2 - 4 * A * C
    coeff_gcd = math.gcd(A,B,C)
    Ar = int(A/coeff_gcd); Br = int(B/coeff_gcd); Cr = int(C/coeff_gcd); discr = (Br)**2-4*(Ar)*(Cr)
    print("Reduced (Equation) {}u^2 + {}uv + {}v^2, disc = {}".format(Ar,Br,Cr,discr))
    
    if(disc < 0):
        return 0

    if is_square(disc):
        sqroot = int(math.sqrt(discr))
    # quadratic is Aa^2+Bab+Cb^2=0, discriminant is sqroot
    # the root of this is a/b = (-B +- sqroot)/2A, 
    # so we will set a as the (reduced) numerator, and b as the (reduced) denominator
        print("Ar = {}, Br = {}, Cr = {}, sqroot = {}".format(Ar,Br,Cr,sqroot))
        roots = []
        if (-B + sqroot)/A > 0:    
            d = gcd(-B + sqroot, 2*A)
            roots.append([int((-B + sqroot)/d),int((2*A)/d)])
        if (-B - sqroot)/A > 0:
            d = gcd(-B - sqroot, 2*A)
            roots.append([int((-B - sqroot)/d),int((2*A)/d)])
        # print("Roots: {}".format(roots))
        if len(roots) == 0:
            return 0 #This is when the root is zero (i.e. A = 0) or both roots are negative

        print("Roots: {}".format(roots))

        root = [] # this will be the root we select
        
        # TODO: Both roots might work?
        a = sol[0]; b = sol[1]; c = sol[2]; d = sol[3]
        for potential_root in roots:
            aa = a[0] * potential_root[0] + a[1] * potential_root[1]
            bb = b[0] * potential_root[0] + b[1] * potential_root[1]
            cc = c[0] * potential_root[0] + c[1] * potential_root[1]
            dd = d[0] * potential_root[0] + d[1] * potential_root[1]
            print(aa,bb,cc,dd)
            if aa > 0 and bb > 0 and cc > 0 and dd > 0:
                root = sol
                break

        if len(root) == 0:
            return 0

        print("Yess!!")
        print("[a,b,c,d] = [{},{},{},{}]".format(a,b,c,d))
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
        return 0

def vec_check(vec,list_of_black):
    vectors = fan_from_surface(vec, list_of_black)
    if vectors == -1:
        return 0
    fan = poly_fan(vectors) 
    # print("Dual Fan: ",end = ' ')
    # print(fan)
    sol = solve_fan(fan)
    ans = check_good(fan, sol) #returns 1 if found, 0 otherwise
    return ans


poly_fan = [[1,1],[-1,1],[-1,-2],[37,31]]
sol = solve_fan(poly_fan)
check_good(poly_fan,sol)

