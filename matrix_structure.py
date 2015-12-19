from utils import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 5
    m = 10

    interior_ordering = lambda i, j, n, m : lexi_ordering(i, j, n, m)-lexi_num_boundary_nodes_before(i, j, n, m)
    l_h = assemble_l_h_rectangular_grid(n, n, interior_ordering)
    plt.spy(l_h)
    plt.title("lexi order square domain n==m")
    plt.show()

    interior_ordering = lambda i, j, n, m : lexi_ordering(i, j, n, m)-lexi_num_boundary_nodes_before(i, j, n, m)
    l_h = assemble_l_h_rectangular_grid(n, m, interior_ordering)
    plt.spy(l_h)
    plt.title("lexi order rectangular domain n!=m")
    plt.show()

    interior_ordering = lambda i, j, n, m : lexi_checker_ordering(i, j, n, m)-lexi_check_num_boundary_nodes_before(i, j, n, m)
    l_h = assemble_l_h_rectangular_grid(n, m, interior_ordering)
    plt.spy(l_h)
    plt.title("lexi check order rectangular domain n!=m")
    plt.show()


    l_h, _  =  assemble_on_unit_ball(n, m)
    plt.spy(l_h)
    plt.title("lexi order unit ball")
    plt.show()

