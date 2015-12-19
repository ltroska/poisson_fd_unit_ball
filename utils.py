import numpy as np
from math import ceil, sqrt, exp, sin

# --------------------------------------- HELPERS --------------------------------------- #

#checks if node (i,j) is inside the unit ball
def inside_unit_ball(i, j, n, m, eps = 1e-15):
    h = 1./n
    return (-1+i*h)**2+(-1+j*h)**2 - 1 < -eps

#number of even numbers x st a<=x<=b
def num_even_between(a, b):
    if b < a:
        return 0

    return int(ceil((b-a)/2.))+ (1 if a%2==0 and b%2 == 0 else 0)

#number of odd numbers x st a<=x<=b
def num_odd_between(a, b):
    if b < a:
        return 0

    return num_even_between(a+1, b+1)

#gets the factor to warp node outside boundary of unit ball to the boundary
def get_factor(i, j, k, l, h):
    x = -1+i*h
    y = -1+j*h

    if k == 0 and l == 0:
        return 0
    if k != 0 and l == 0:
       return (np.sign(k)*sqrt(1-y**2)-x)/float(k*h)
    if l != 0 and k == 0:
        return (np.sign(l)*sqrt(1-x**2)-y)/float(l*h)

# --------------------------------------- ORDERINGS --------------------------------------- #

#index for node (i,j) in the lexicographical ordering for rectangle
def lexi_ordering(i, j, n, m):
    return j*(n+1)+i

#number of boundary nodes before node (i,j) in lexicographical ordering for rectangle
def lexi_num_boundary_nodes_before(i, j, n, m):
    return 2*j + 1 + (n-1)

#number of boundary nodes before node (i,j) in lexicographical ordering for unit ball
def lexi_unit_ball_num_boundary_nodes_before(i, j, n, m):
    two_n = 2*n

    cnt = 0
    for l in range(0, j+1):
        for k in range(0, two_n+1):
            if not inside_unit_ball(k,l,n,m):
                cnt += 1

            if k == i and l == j:
                return cnt

#index for node (i,j) in the lexicographical checker ordering for rectangle
def lexi_checker_ordering(i, j, n, m):
    if (i+j)%2 == 0:
        return num_even_between(0, j-1)*num_even_between(0, n) + num_odd_between(0, j-1)*num_odd_between(0, n) +( num_even_between(0, i-1) if j%2 == 0 else num_odd_between(0, i-1))
    else:
        return num_even_between(0, m)*num_even_between(0, n) + num_odd_between(0, m)*num_odd_between(0, n) \
                + num_even_between(0, j-1)*num_odd_between(0, n) + num_odd_between(0, j-1)*num_even_between(0, n) + (num_even_between(0, i-1) if j%2 == 1 else num_odd_between(0, i-1))

#number of boundary nodes there are before node (i,j) in the lexicographical checker ordering for rectangle
def lexi_check_num_boundary_nodes_before(i, j, n, m):
    if (n+1) % 2 == 0:
        if (i+j) % 2 == 1:
            return (n+1)+(m-1) + num_odd_between(0, n) + j-1 + (1 if j%2 == 1 else 0)
        else:
            return num_even_between(0, n) + j-1 + (1 if j%2 == 0 else 0)
    else:
        if (i+j) % 2 == 1:
            return (n+1)+(m-1) + num_odd_between(0, n) + j
        else:
            return num_even_between(0, n) + (j-1)


# --------------------------------------- ASSEMBLING L_H --------------------------------------- #

def assemble_l_h_rectangular_grid(n, m, interior_ordering):
    h = 1./n

    l_h = np.zeros(((n-1)*(m-1),(n-1)*(m-1)))

    #setup stencil
    for i in range(1, n):
        for j in range(1, m):
            l_h[interior_ordering(i, j, n, m)][interior_ordering(i, j, n, m)] = 4./(h*h)
            if i-1 > 0:
             l_h[interior_ordering(i, j, n, m)][interior_ordering(i-1, j, n, m)] = -1/(h*h)
            if i+1 < n:
                l_h[interior_ordering(i, j, n, m)][interior_ordering(i+1, j, n, m)] = -1/(h*h)
            if j-1 > 0:
                l_h[interior_ordering(i, j, n, m)][interior_ordering(i, j-1, n, m)] = -1/(h*h)
            if j+1 < m:
                l_h[interior_ordering(i, j, n, m)][interior_ordering(i, j+1, n, m)] = -1/(h*h)

    return l_h

#split nodes into interior and boundary
def split_nodes_unit_ball(n, m):
    h = 1./n
    two_n = 2*n

    interior_nodes = np.array([[-1, -1]])
    for j in range(two_n+1):
        for i in range(two_n+1):

            if inside_unit_ball(i, j, n, m):
                interior_nodes = np.append(interior_nodes, [[i, j]], axis=0)

    interior_nodes = np.delete(interior_nodes, 0, axis=0)

    boundary_nodes = np.array([[-4, -4]])

    for idx, node in enumerate(interior_nodes):
        i, j = node
        for k in [-1, 0, 1]:
            for l in [-1,0, 1]:
                if (k == 0 and l == 0) or (k != 0 and l != 0):
                    continue

                if not inside_unit_ball(i+k, j+l, n, m):
                    x = -1+i*h+k*get_factor(i, j, k, l, h)*h
                    y = -1+j*h+l*get_factor(i, j, k, l, h)*h
                    boundary_nodes = np.append(boundary_nodes, [[x,y]], axis=0)

    boundary_nodes = np.delete(boundary_nodes, 0, axis=0)

    return interior_nodes, boundary_nodes

#assemble l_h, k_h, f_h, g_h for unit ball problems
def assemble_on_unit_ball(n, m, f=None, g=None):
    h = 1./n
    two_n = 2*n

    interior_nodes, boundary_nodes = split_nodes_unit_ball(n, m)

    num_interior_nodes = len(interior_nodes)
    num_boundary_nodes = len(boundary_nodes)

    #mapping to find respective index in the arrays
    def get_boundary_index(i, j, k, l, eps = 1e-13):
        if inside_unit_ball(i+k, j+l, n, m) or abs(get_factor(i, j, k, l, h)) - 1 > eps :
            return -1

        x = -1+i*h+k*get_factor(i, j, k, l, h)*h
        y = -1+j*h+l*get_factor(i, j, k, l, h)*h

        return np.where(np.all(boundary_nodes==[x,y], axis=1))[0][0]

    def get_interior_index(i, j):
        tmp = np.where(np.all(interior_nodes==[i, j], axis=1))[0]
        if len(tmp) == 0:
            return -1

        return tmp[0]

    # ---------------------------- L_H/K_H ---------------------------- #

    #setup stencils
    l_h = np.zeros((num_interior_nodes, num_interior_nodes))
    k_h = np.zeros((num_interior_nodes, num_boundary_nodes))

    for idx, node in enumerate(interior_nodes):
        i, j = node

        #adaptive stepsize (Shortley-Weller)
        if inside_unit_ball(i+1, j, n, n):
             h_e = h
        else:
             h_e = get_factor(i, j, 1, 0, h)*h

        if inside_unit_ball(i-1, j, n, n):
             h_w = h
        else:
             h_w = get_factor(i, j, -1, 0, h)*h

        if inside_unit_ball(i, j+1, n, n):
             h_n = h
        else:
             h_n = get_factor(i, j, 0, 1, h)*h

        if inside_unit_ball(i, j-1, n, n):
             h_s = h
        else:
             h_s = get_factor(i, j, 0, -1, h)*h

        l_h[idx][idx] = 2./(h_e*h_w)+2./(h_s*h_n)

        #factor goes into l_h if neighbor is interior node, into k_h otherwise
        if inside_unit_ball(i-1, j, n, m):
            l_h[idx][get_interior_index(i-1, j)] = -2./(h_w*(h_e+h_w))
        else:
            k_h[idx][get_boundary_index(i, j, -1, 0)] = -2./(h_w*(h_e+h_w))

        if inside_unit_ball(i+1, j, n, m):
            l_h[idx][get_interior_index(i+1, j)] = -2./(h_e*(h_e+h_w))
        else:
            k_h[idx][get_boundary_index(i, j, 1, 0)] = -2./(h_e*(h_e+h_w))

        if inside_unit_ball(i, j-1, n, m):
            l_h[idx][get_interior_index(i, j-1)] = -2./(h_s*(h_s+h_n))
        else:
            k_h[idx][get_boundary_index(i, j, 0, -1)] = -2./(h_s*(h_s+h_n))

        if inside_unit_ball(i, j+1, n, m):
            l_h[idx][get_interior_index(i, j+1)] = -2./(h_n*(h_s+h_n))
        else:
            k_h[idx][get_boundary_index(i, j, 0, 1)] = -2./(h_n*(h_s+h_n))

    # ---------------------------- G_H / F_H ---------------------------- #

    #sample f on all interior and g on all boundary nodes
    if f is not None and g is not None:
        f_h = np.zeros(num_interior_nodes)
        g_h = np.zeros(num_boundary_nodes)

        for idx, node in enumerate(interior_nodes):
            i, j = node
            f_h[idx] = f(-1+i*h, -1+j*h)

        for idx, node in enumerate(boundary_nodes):
            g_h[idx] = g(*node)

        return l_h, k_h, f_h, g_h

    return l_h, k_h

def solve_on_unit_ball(n, m, f, g, ret_lh = False):
    #get matrices/vectors
    l_h, k_h, f_h, g_h = assemble_on_unit_ball(n, m, f, g)

    #solve
    u_h = np.linalg.solve(l_h, f_h-k_h.dot(g_h))

    if ret_lh:
        return u_h, l_h

    return u_h
