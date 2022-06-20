#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:39:56 2022

@author: stephenbillups
  
Integer separation for CUT and LCUT models from Validi, Buchanan, et al, 2021.
"""

import numpy as np
import networkx as nx
import timeit



# def bfs(G, node):
#     """Performs breadth first search to find the set of nodes that
#        can be reached from a given node in a graph G

#     Parameters
#     ----------
#     G :  networkx graph
#     node : integer (0<=node<=m-1) index of node in 

#     Returns
#     -------
#     a set of nodes that is the connected component of G containing node
#     """

#     m = G.shape[0]
#     visited = {node}
#     queue = {node}
#     while queue:
#         s = queue.pop()
#         for neighbor in range(m):
#             if G[s, neighbor] == 1:
#                 if neighbor not in visited:
#                     visited = visited.union({neighbor})
#                     queue = queue.union({neighbor})
#     return visited


def neighborhood_closed(G: nx.Graph, subset: set):
    """Finds the closed neighborhood of a subset of nodes within a graph

    Parameters
    ----------
    G: networkx graph
    subset: a subset of G.nodes

    Returns
    -------
    neighborhood: the closed neighborhood of the subset--i.e., the set of 
        all neighbors of nodes in the subset together with the subset itself
    """

    neighborhood = subset.copy()
    for s in subset:
        neighborhood = neighborhood.union(G.neighbors(s))
    return neighborhood

def test_neighborhood(numit:int):
    import timeit
    G = nx.Graph()
    elist = [(0, 3), (0, 2), 
             (1, 3), (1, 6), 
             (2, 0), (2, 4), (2, 5), 
             (3, 0), (3, 1), (3, 4),
             (4, 3), (4, 2), (4, 6), (4, 7), 
             (5, 2), (5, 7), 
             (6, 1), (6, 4), (6, 8), 
             (7, 4), (7, 5), (7, 8), 
             (8, 6), (8, 7)]
    G.add_edges_from(elist)
    nodeset = {0, 2, 3}
    starttime = timeit.default_timer()
    for i in range(numit):
       neighborhood = neighborhood_closed(G,nodeset) 
    stoptime = timeit.default_timer()
    print('Time: ', stoptime-starttime)
    print('neighborhood = ', neighborhood)
    return neighborhood

def minimal_separator(G: nx.Graph, node: int, subset: set):

    """Returns a minimal node separator that separates a node from a subset
       of nodes in a graph.

    This implementation is based on Algorithm 1 from Fischetti et al [1].


    Parameters                                                                  
    ----------                                                                  
    G :  networkx graph
    node:  a node in the graph G
    subset:  subset of nodes which is *not* connected a

    
    Returns                                                                     
    -------                                                                     
    cutset:  a set of nodes that is a minimal node separator.  
                The following properties are satisfied:

        1) (node separator) every path between the input node and the subset 
           must contain a node in cutset (node separator)
        2) (minimal) cutset does not have a proper subset that is also a 
           node separator
           
    Notes
    -----
    This function assumes (without checking) that the input node is *not* a 
    neighbor of the subset
    
    """

    # References
    # ----------
    # .. [1]  Fischetti, Leitner, et al, Thinning out Steiner trees: a
    #         node-based model for uniform edge costs.  Mathematical Programming
    #         Computation, 9(2):231-296, 2017.

    closed_neighborhood = neighborhood_closed(G, subset)
    open_neighborhood = closed_neighborhood.difference(subset)

    # Get the set of nodes reachable from the input node going without 
    # going through nodes in the closed neighborhood.   
    # To do this, convert G to a directed graph and remove all edges 
    # emanating from nodes in the closed neighborhood.
    
    subgraph = G.to_directed()
    edgeset = [(i,j) for i in closed_neighborhood for j in G.nodes]
    subgraph.remove_edges_from(edgeset)
    reach_from_node = nx.descendants(subgraph,node)
    cutset = reach_from_node & open_neighborhood
    return cutset
    


def test_minimal_separator():
    G = nx.Graph()
    elist = [(0, 3), (0, 2), 
             (1, 3), (1, 6), 
             (2, 0), (2, 4), (2, 5), 
             (3, 0), (3, 1), (3, 4),
             (4, 3), (4, 2), (4, 6), (4, 7), 
             (5, 2), (5, 7), 
             (6, 1), (6, 4), (6, 8), 
             (7, 4), (7, 5), (7, 8), 
             (8, 6), (8, 7)]
    G.add_edges_from(elist)

    nodeset = {0, 2, 3, 6, 7, 8}
    node = 4
    subset={1}
    return minimal_separator(G, node, subset)


def find_farthest_node_wrto_edge_weights(D: nx.DiGraph, start_node: int, 
                                         subset: set, weight = "weight"):
    """ Finds the farthest node in a subset from a given node in a 
        directed graph with nonnegative edge weights.
        
           
        Parameters
        ----------
        D: A networkx digraph with weighted edges.  It is assumed that
           that the weight of edge (i,j) is given by D.edges[i,j][weight], 
           where weight is a text string given as input.
        start_node: start node from which distances are calculated
        subset: subset of nodes from which to choose the farthest node
        weight:  string, optional (default="weight")  name of the edge 
           attribute for the edge weights.
           
        Returns
        -------
        far_node: node that is farthest from the start_node
        max_length: length of the shortest path from start_node
                    to far_node
    """

    path_lengths = nx.shortest_path_length(D, source=start_node, weight=weight)
    # get max path length and node number of max path length
    
    
    max_length=0
    far_node=start_node
    for i in subset:
        v=path_lengths[i]
        if v > max_length:
            max_length = v
            far_node = i
    return far_node, max_length

def find_farthest_node_wrto_node_weights(G:nx.Graph, start_node, subset,
                                         weight = "weight"):
    """ Finds the farthest node in a subset from a given node in an undirected
        graph using node weights to define distance.  That is, the length of
        a path is defined as the sum of the weights of the nodes on the
        path.   The distance to a node is the length of the shortest path to 
        the node from the start_node.  
        
        Parameters
        ----------
        G: A networkx graph with weighted nodes.   It is assumed that
           that the weight of node i is given by G.nodes[i][weight], 
           where weight is a text string given as input.
        start_node: start node from which distances are calculated
        subset: subset of nodes from which to choose the farthest node
        weight:  string, optional (default="weight")  name of the edge 
           attribute for the edge weights.
        
        
        Returns
        -------
        far_node: node that is farthest from the start_node
        max_length: length of the shortest path from start_node
                    to far_node
        
        Notes:
        ------
        This routine is about 50% slower than converting to an edge weighted
        digraph and using the networkx shortest path algorithm.  However, this
        routine can be customized to get greater efficiency within the LCUT
        algorithm
        """

    n=G.number_of_nodes()
    distances = [1000000]*n  # change this big number to infinity
    distances[start_node]=G.nodes[start_node][weight]
    current = start_node
    fixed_nodes = {current}  # nodes for which shortest path has been found
    temp_nodes = set(G.neighbors(current))  # unfixed nodes for which a path  
                                            # has been found
    nodes_to_update = set(G.neighbors(current)) # nodes to update each iteration
    while temp_nodes:

        # update distances for neighbors of current node
 
        for i in nodes_to_update:
            tentative_value = distances[current]+G.nodes[i][weight]
            if tentative_value < distances[i]:
                distances[i] = tentative_value
        
        # find closest temp node and make it the next current node
        
        smallest_dist = 1000000
        for i in temp_nodes:
            if distances[i]<smallest_dist:
                smallest_dist = distances[i]
                current = i
                
        # mark new current node as fixed 
        fixed_nodes = fixed_nodes | {current}
        temp_nodes = temp_nodes - {current}
        
        nodes_to_update = set(G.neighbors(current))-fixed_nodes      
        temp_nodes = temp_nodes | nodes_to_update

    far_node = start_node
    for i in subset:
        if distances[i] > distances[far_node]:
            far_node = i
#    print(distances)
    return far_node, distances[far_node]
            
def test_farthest_node(numit: int):
    
    G = nx.Graph()
    elist = [(0, 3), (0, 2), 
             (1, 3), (1, 6), 
             (2, 0), (2, 4), (2, 5), 
             (3, 0), (3, 1), (3, 4),
             (4, 3), (4, 2), (4, 6), (4, 7), 
             (5, 2), (5, 7), 
             (6, 1), (6, 4), (6, 8), 
             (7, 4), (7, 5), (7, 8), 
             (8, 6), (8, 7)]
    weights = [1,2,3,4,5,6,7,8,9]
    G.add_edges_from(elist)
    nx.set_node_attributes(G,[],"weight")
    for i in G.nodes():
        G.nodes[i]["weight"] = weights[i]
    subset = {6,7,8}
    start_node = 0
    D=G.to_directed()
    nx.set_edge_attributes(D,[],"weight")
    for i,j in D.edges:
        D.edges[i,j]["weight"]=D.nodes[j]["weight"]
        
    starttime = timeit.default_timer()
    for n in range(numit):
        far_node, max_length = find_farthest_node_wrto_edge_weights(
            D, start_node, subset, weight="weight")
    stoptime = timeit.default_timer()
    print("run time for find_farthest_node = ", stoptime-starttime)
    print(far_node, max_length)

    starttime = timeit.default_timer()
    for n in range(numit):
        far_node, max_length = find_farthest_node_wrto_node_weights(
            G, start_node, subset, weight="weight")
    stoptime = timeit.default_timer()
    print("run time for max_length_to_component = ", stoptime-starttime)
    print(far_node, max_length)
 
    return far_node, max_length
            
# def reduce_separating_sets(G: nx.Graph, upper_bound: int, start_node: int, 
#                            component: set, separating_set: set, weight="weight"):
#     """
#     Finds a subset S of a separating set (separating start_node from a 
#     connected component of the graph G) that forms a length-U separating set. 
    
#     S is called a length-U separating set if there is at least
#     one node in component of distance greater that U from the start_node in the 
#     subgraph H formed by removing S from G.  Here, the distance is defined by
#     the sum of the node weights on the shortest path
    

#     Parameters
#     ----------
#     G : nx.Graph
#         An undirected graph with node weights given by node attribute weight
#     upper_bound : int
#         The upper bound U
#     start_node : int
#     component : set
#         A connected component of the graph G.
#     separating_set : set
#         A set of nodes that separates start_node from component
#     weight : test, optional, default="weight"
#         Name of the attribute specifying the weight of each node

#     Returns
#     -------
#     S : set
#         The length-U separating set
#     far_node : a node in component which has distance greater than U from
#         start_node in the subgraph H=G\S

#     """

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects


def integer_separation(G: nx.Graph, ub: np.ndarray, start_nodes: np.ndarray,
                       x: np.ndarray, weight_attr="weight", method = "CUT"):

    """ Finds a length-U (a,b)-separator for each disconnected component of
        each subset of a partition of the nodes in a graph G.

        Implements Algorithm 1 of Validi, Buchanan, et al [1]


    Parameters
    ----------
    G : nx.Graph
        Undirected graph with weighted nodes
    ub : np.ndarray,  length = # of partitions
        Upper bounds for each of the partitions.  
        ub[j] is upper bound for partition j
    start_nodes : np.ndarray, length = # of partitions
        start_nodes[j] = starting node for partition j.  
        (note: It must be assigned to partition j)
    x : np.ndarray, binary
        x[v,j] = 1 iff node v is assigned to partition j
    weight_attr : text, default = "weight"
        Name of the weight attribute for the nodes in the graph G.
        i.e., G.nodes[i][weight_attr] = weight of node i
    method:  text, default="CUT"
        specifies cut model to be used (either "CUT" or "LCUT")

    Returns
    -------
    cutsets : list of list of data structures describing the 
        (a,b)-separators for each partition.  
        
        cutsets[b] is a list of data structures describing the
            separating sets for each connected component of the partition
            corresponding to start_nodes[b].
            
            If this partition is connected, then cutsets[b] is an empty list.
            Otherwise cutsets[b][i] describes the separating set for the
            ith component of partition b (not including the component that
            that contains start_nodes[b].  It has the form
            
            cutsets[b][i] = (node_to_cut, separating_set),
            
            where 
                separating_set is a set of nodes not in partition b
                    that forms a length-ub[b] separator between start_node[b]
                    and the nodes in the ith connected component of partition b.
                node_to_cut is a node in the ith connected component of 
                    partition b that has distance greater than ub[b] from
                    start_node[b] in the subgraph formed by deleting the 
                    nodes in separating_set from G

    """
    

 
    n_parts = len(x[0, :])  # number of subsets in the partition

    # node_set[i] = list of nodes assigned to school i by x

    node_set = [list() for i in range(n_parts)]
    for b in range(n_parts):
        node_set[b] = list(np.nonzero(x[:, b])[0])

    cutsets = init_list_of_objects(n_parts) 

    # Create digraph version of G, moving node weights to edges

    D=G.to_directed()
    nx.set_edge_attributes(D,[],weight_attr)
    for i,j in D.edges:
        D.edges[i,j][weight_attr]=G.nodes[j][weight_attr]

    
    for b in range(n_parts):
        # V_b = set of nodes assigned to precinct b
        V_b = node_set[b]
        node_b=start_nodes[b]  # start node for partition b

        # create subgraph containing only edges within V_b

        G_b = G.subgraph(V_b)

        c_b = []  # initialize list of separating sets

        # find connected components of subgraph
        
        components = list(nx.connected_components(G_b))

        if len(components) <= 1:
            continue  # partition is connected--nothing to do

        # find component containing the start node for the partition
        # and remove it from the list of components

        for i in range(len(components)):
            if node_b in components[i]:
                del components[i]
                break

        # find separating sets for each of the other components

        for component_a in components:
 
            separating_set = minimal_separator(G, node_b, component_a)
 

              
            if method == "LCUT":
                
                # remove any nodes from the separating set that fail the length-U
                # separation test, and choose node_a in component_a 
                # such that dist(node_a,node_b) > U in the subgraph formed
                # by removing the reduced separating set
                
                node_a = None
                # node_a = list(component_a)[0]
                for node_c in separating_set:
                    # create subgraph obtained by removing separating_set-{c}
                    # from the original graph
    
                    subset = separating_set.difference({node_c})
                    subgraph = D.copy()
                    subgraph.remove_nodes_from(list(subset))

                    if node_a == None:
                        # find longest length path from node_b to any node in
                        # component_a within subgraph
                        node_a, path_length = find_farthest_node_wrto_edge_weights(
                            subgraph, node_b, component_a, weight_attr)
                    else:
                        # find path length to node_a
                        path_length = nx.shortest_path_length(
                            subgraph, source=node_b, target=node_a, weight=weight_attr)
                        path_length=path_length+D.nodes[node_b][weight_attr]

                    if path_length > ub[b]:
                        # remove node c from separating set
                        separating_set = separating_set.difference({node_c})
                        # print("path_length=", path_length, "ub[b]=", ub[b])
                        # print("separating set = ", separating_set)
            else:  # method == "CUT"
                # Use the whole separating set and choose node_a to be an
                # arbitrary element of component_a
              
                node_a = list(component_a)[0]
                
                
            # if any nodes remain in the separating set, add node_a together
            # with the separating set to the list of separating sets for 
            # node_b

            if len(separating_set) > 0:
                # test if separating_set has already been found
                in_set = False
                for ss in c_b:
                    if separating_set == ss:
                        in_set = True
                        break
                if in_set == False:
                    # add the separating set to the list
                    c_b.append((node_a, separating_set))
        cutsets[b]=c_b
    return cutsets


def test_integer_separation():

    G = nx.Graph()
    elist = [(0, 3), (0, 2), 
             (1, 3), (1, 6), 
             (2, 0), (2, 4), (2, 5), 
             (3, 0), (3, 1), (3, 4),
             (4, 3), (4, 2), (4, 6), (4, 7), 
             (5, 2), (5, 7), 
             (6, 1), (6, 4), (6, 8), 
             (7, 4), (7, 5), (7, 8), 
             (8, 6), (8, 7)]
    weights = [1,2,3,4,5,6,7,8,9]
    G.add_edges_from(elist)
    nx.set_node_attributes(G,[],"weight")
    for i in G.nodes():
        G.nodes[i]["weight"] = weights[i]
    start_nodes = [0,1]    

    ub = [100, 100]
    x = np.zeros((9,2))
    x[0, 0] = 1
    x[1, 1] = 1
    x[2, 0] = 1
    x[3, 0] = 1
    x[4, 1] = 1
    x[5, 1] = 1
    x[6, 0] = 1
    x[7, 0] = 1
    x[8, 0] = 1
    weight_attr = "weight"
    solution = integer_separation(G, ub, start_nodes, x, weight_attr)

    return solution
