from utils import *
from math import log
import matplotlib.pyplot as plt

def make_convergence_plot(exact_sol, u_hs, ns, norm_type = None):
    norms = []
    actual_ns = (2*np.array(ns)+1)**2
    for indx, n in enumerate(ns):
        h = 1./n
        int_nodes, _ = split_nodes_unit_ball(n, n)

        u_h = u_hs[indx]
        u_e = np.zeros_like(u_h)

        for idx, node in enumerate(int_nodes):
            i, j = node
            u_e[idx] = exact_sol(-1+i*h, -1+j*h)

        norms.append(np.linalg.norm(u_e-u_h, ord=norm_type))

    lbda = (log(norms[-1]) - log(norms[0])) / (log(actual_ns[-1]) - log(actual_ns[0]))

    print "Estimated convergence rate:", lbda
    plt.plot(actual_ns, norms)
    plt.loglog()
    plt.grid()
    plt.xlabel("#nodes")
    h = plt.ylabel("$||u_e-u_h||$")
    #h.set_rotation(0)
    plt.title("convergence "+( "2" if norm_type is None else str(norm_type))+ " norm")
    plt.show()

def make_stability_plot(l_hs, ns, norm_type = None):
    norms = []

    for l_h in l_hs:
        norms.append(np.linalg.norm(np.linalg.inv(l_h), ord=norm_type))

    actual_ns = (2*np.array(ns)+1)**2

    plt.plot(actual_ns, norms)
    plt.grid()
    plt.xlabel("#nodes")
    h = plt.ylabel("$||L_h^{-1}||$")
    #h.set_rotation(0)
    plt.title("stability "+( "2" if norm_type is None else str(norm_type))+ " norm")
    plt.show()

if __name__ == "__main__":
    f = lambda x, y: 0
    g = lambda x, y: exp(x)*sin(y)

    n = 2
    it = 10
    inc_func = lambda n: n+1


    u_hs = []
    l_hs = []
    ns = []
    for i in range(it):
        print "iteration", i
        ns.append(n)

        u_h, l_h = solve_on_unit_ball(n, n, f, g, ret_lh=True)

        u_hs.append(u_h)
        l_hs.append(l_h)

        n = inc_func(n)

    make_convergence_plot(g, u_hs, ns)
    make_stability_plot(l_hs, ns)