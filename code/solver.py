"""Functions for the ILP formulation and a driver function."""

# from collections import deque
# import random
# import sys

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

from cuts import integer_separation
import math
import networkx as nx


def fix_variables():
    """
    Uses Lagrangian-based variable fixing to fix many of the variables to zero
    

    Returns
    -------
    None.

    """

   
def contiguity_callback(model: gp.Model, where: int):
    """Add cutting planes to the model if contiguity is violated.

    Args:
        model: The Gurobi model, passed through callback.
        where: An enum-like return code to indicate solver status.
    """

    global g0_contiguity_graph  # np.mat (n_spas x n_spas) storing contiguity graph
    global g0_upper_bounds      # np.array (n_schools) of upper bounds on school attendance
    global buffer, solution, g0_G
    global total_num_cuts 
    if where != GRB.Callback.MIPSOL:
        return
 #   print("in contiguity_callback")
    n_spas=g0_contiguity_graph.number_of_nodes()
    n_schools=len(g0_upper_bounds)
    # Retrieve and reformat the current MIP node from Gurobi.
    # Round the solution from float output to a binary matrix.
    buffer = list(map(round, model.cbGetSolution(model.getVars())))
    buffer = np.array(buffer, np.int64)
 #   print(buffer)
    solution = np.reshape(buffer, (n_schools, n_spas))
    solution = np.transpose(solution)
 #   print(solution)

    cutsets = integer_separation(g0_contiguity_graph, g0_upper_bounds, 
                                 g0_anchor_nodes, solution, "pop",
                                 "CUT")
#    print(cutsets)
    
    # add constraints for cuts
    
    num_cuts=0
    for saz in range(n_schools):
#        print("saz = ", saz, "len(cutsets) =", len(cutsets))        
#        print("cutsets[saz]=", cutsets[saz])
 
        for spa,cut in enumerate(cutsets[saz]):
            # print("spa = ", spa, "saz = ", saz)
            # print("cut = ", cut)
            
            lhs_vars = gp.quicksum(model._vars[saz,j] for j in {cut[0]})
            lhs_vars -= gp.quicksum(model._vars[saz,j] for j in cut[1])
            model.cbLazy(lhs_vars <= 0)
            num_cuts+=1
    total_num_cuts += num_cuts        
    print("new_cuts=", num_cuts)
    check3  = True
    if num_cuts==0:
        # confirm connectedness
        for i in range(n_schools):
            saz_index_set = np.where(solution[i, :] == 1)[0]
            subgraph = g0_G[saz_index_set, :][:, saz_index_set]
            subgraph = nx.from_numpy_matrix(subgraph)
            if not nx.is_connected(subgraph):
                check3 = False
                break
    
#        print("SAZs are connected:   ", "Pass" if check3 else "Fail")
    


def formulate_baseline_model(adj_matrix: np.ndarray, SAZ_centers: np.ndarray, 
                 pop: np.ndarray, low_bounds: np.ndarray, 
                 up_bounds: np.ndarray, dist_to_school: np.ndarray,
                 method: int):
    """ Formulates the baseline optimization model (without contiguity constraints)
        for the school redistricting problem.   The model has the form 

        min sum_{i in SPAs} sum_{j in SAZs}   W[i,j] x[i,j]
        s/t     
            sum_{j in SAZs} x[i,j] = 1,          i in SPAs
            L[j] <= sum_{i} pop[i]*x[i,j] <= U[j],    j in SAZs
            x[i,j] binary,     i in SPAs, j in SAZs
               
        where 
            x[i,j] is a binary variable indicating whether SPA jis assigned to
                school i (1=yes, 0=no).        
            W[i,j] is the cost of assigning SPA j to school i, and is determined
                by the problem data.  We implement several options determined
                by input parameter 'method'
                  

            L[i], U[i] are lower and upper bounds on school attendance                 
            
        input parameters:  (n = |SPAs|, k = |SAZs|)
        -----------------
            adj_matrix:  n x n adjacency matrix for contiguity graph
                adj_matrix[i,j] = 1 if node i is adjacent to node j
            SAZ_centers: array of indices indicating the SAZ centers.
                SAZ_centers[j] = index of SPA that contains the school
                    for SAZ j.  (np.ndarray of length k)
            pop:  number of students in each of the SPAs
                (np.ndarray of length n)
            low_bounds, up_bounds:  lower and upper enrollment bounds
                (np.ndarray of length k)
            dist_to_school:    distances of SPAs to SAZs
                dist_to_school[i,j] = distance of SPA j to SAZ i
            method:  specifies which method is used to define the objective
                coefficients:
                method==1 (Hess model):  W[i,j] = pop[j]*d[i,j]^2
                    where d[i,j] denotes the distance from i to j.
                method==2: W[i,j] = pop[j]*d[i,j]
                method==3: (NOT YET IMPLEMENTED)
                    W[i,j] = 0 if school i is in walking distance from
                             SPA j
                           = pop[j]*d[i,j]^2 if not walkable
                
    """ 
    n_schools = len(up_bounds)
    n_spas = len(pop)
    model = gp.Model("redistricting")
    
    # Declare X to be a matrix of variables: n_schools x n.
    
    X_indices = gp.tuplelist([(i, j) for i in range(n_schools)
                              for j in range(n_spas)])
    X = model.addVars(X_indices, vtype=GRB.BINARY, name="X")

    # Use a temporary objective function to get other things working.
    # minimize weighted sum of distances from center of spa to assigned school
    # where weight is equal to the number of students travelling
    
    W = dict()
    for i, j in X_indices:
        W[i, j] = pop[j]*dist_to_school[i,j]**2
        
    model.setObjective(X.prod(W), GRB.MINIMIZE)
    
    # Insist that each SPA be assigned exactly once.
    for j in range(n_spas):
        model.addConstr(X.sum("*", j) == 1, "Col_%d_exactly1" % j)
    

    # Assign schools to the SPA they're located in.
    for i in range(n_schools):
        model.addConstr(X[i,SAZ_centers[i]] == 1, "SAZ_col_%d" % i)

    # Attendance constraints. P is a matrix due to Gurobi's API.
    
    P = dict()

    for i, j in X_indices:
        P[i, j] = pop[j]
        
    for i in range(n_schools):
        A = X.prod(P, i, "*")
        model.addConstr(A <= up_bounds[i], "Attendance_UB_%d" % i)
        model.addConstr(A >= low_bounds[i], "Attendance_LB_%d" % i)
        

    model.update()
    model._vars = X
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 0

    return model, X, X_indices
# end function formulate_baseline_model


def  solve_model(model, n_schools, n_spas, X, X_indices):
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 0

    model.optimize(contiguity_callback)
#    model.optimize()

    model.printStats()
    model.printQuality()
    
    solution = []
    if model.status == GRB.OPTIMAL:
        solution = np.zeros((n_schools, n_spas), dtype=np.int64)
        grb_output = model.getAttr("x", X)
        for i, j in X_indices:
            # Output is in floating point format; round solutions.
            X_ij = grb_output[i, j]
            assert abs(X_ij - round(X_ij)) <= 1e-8, X_ij
            solution[i, j] = round(X_ij)
    return solution
# end of function solve_model



def main():

    # Global variables needed for contiguity_callback
    
    global g0_contiguity_graph  # np.mat (n_spas x n_spas) storing contiguity graph
    global g0_upper_bounds      # np.array (n_schools) of upper bounds on school attendance
    global g0_anchor_nodes      # np.array (n_schools) of node indexes for the home spa
                                # for each school.
    global buffer, solution, g0_G    
    global total_num_cuts
    
    # Threshold on attendance balancing constraint.
    tau = {"hs": 0.2, "ms": 0.2, "es": 0.25}
    school_level="hs"   # just use high school model for now.  We will eventually
                        # loop over all types of schools
    
    #   Load spa data

    G = np.loadtxt("data/adj_matrix.csv", delimiter=",", dtype=np.int64)
    g0_G = G
    spas_df = pd.read_csv("data/spas.csv")
    n_spas = len(spas_df)
    x_P = {"es": "TOTAL_KG_5", "ms": "TOTAL_6_8", "hs": "TOTAL_9_12"}
    pop=np.empty((n_spas),int)
    x_loc = np.empty((n_spas),np.float64)
    y_loc = np.empty((n_spas),np.float64)
    for i in range(n_spas):
        pop[i]=spas_df[x_P[school_level]].iloc[i]
        x_loc[i]=spas_df["x"].iloc[i]
        y_loc[i]=spas_df["y"].iloc[i]
        
    # load school data


    schools_df = pd.read_csv("data/%s.csv" % school_level)
    n_schools = len(schools_df)
    school_spa_location = schools_df["spa"] # index of spa each school 
                                            # is located in
            
    # calculate bounds on school attendance

    up_bounds = np.empty((n_schools))
    low_bounds = np.empty((n_schools))
    anchor_nodes = np.empty((n_schools))
    
    for i in range(n_schools):
        capacity = schools_df["capacity"].iloc[i]
        up_bounds[i] = math.floor((1+tau[school_level])*capacity)
        low_bounds[i] = math.floor((1-tau[school_level])*capacity)
        anchor_nodes[i]= school_spa_location[i]
        # Attendance constraints. P is a matrix due to Gurobi's API.
  
         
    # calculate distance matrix 
    # dist_to_school[i][j] = distance from school i to spa j
    dist_to_school = np.empty((n_schools,n_spas),np.float64)
    for i in range(n_schools):
        for j in range(n_spas):
            x_part = (x_loc[j] - schools_df["x"].iloc[i])**2
            y_part = (y_loc[j] - schools_df["y"].iloc[i])**2
            dist_to_school[i,j] = math.sqrt(x_part+y_part) 
            

                
    # set up global variables for contiguity callback

    g0_upper_bounds = up_bounds
    g0_contiguity_graph = nx.from_numpy_matrix(np.transpose(G),
                                               create_using=nx.Graph)   
    nx.set_node_attributes(g0_contiguity_graph,[],"pop")
    for i in g0_contiguity_graph.nodes:
        g0_contiguity_graph.nodes[i]["pop"] = pop[i] 
    g0_anchor_nodes = anchor_nodes


    
    # set up and solve the optimization model
    total_num_cuts = 0
    method = 1   # use Hess model
    model, X, X_indices = formulate_baseline_model(
        G, anchor_nodes, pop, low_bounds, up_bounds, dist_to_school, method)
    solution = solve_model(model, n_schools, n_spas, X, X_indices)

    solution.dump("results/%s_solution.pkl" % school_level)    
    print("total number of cuts=", total_num_cuts)

    # if len(sys.argv) > 1:
    #     assert sys.argv[1] in ["es", "ms", "hs"]
    #     solve_model(sys.argv[1], tau)
    # else:
    #     # Solve smallest number of schools first.
    #     for school_level in ["hs", "ms", "es"]:
    #         print("Starting solve on %s level.\n\n" % school_level.upper())
    #         solve_model(school_level, tau)


if __name__ == "__main__":
    main()
