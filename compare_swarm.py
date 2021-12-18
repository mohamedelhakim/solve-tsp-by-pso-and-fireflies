# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:18:33 2021

@author: mohamed elhakim
"""

from swarm import *
import math
def graph_to_metrix(graph,amount_vertices):
    arr=np.ones((amount_vertices,amount_vertices),dtype=np.int16)
    for edge in graph.edges:
        arr[edge[0],edge[1]]= graph.edges[edge]
    for i in range(amount_vertices):
        arr[i,i]=0
    return( arr)

for amount_vertices in range(100,101):
    random_graph = CompleteGraph(amount_vertices)
    random_graph.generates()
    print("  ") 
    print("pso : ") 
    population=initial_path(random_graph,10)
    pso = PSO(random_graph, iterations=200, population=population, beta=1, alfa=1)
    pso.run()
    
    print('gbest: %s | cost: %d\n' % (pso.getGBest().getPBest(), pso.getGBest().getCostPBest()))
    print("FA : ")
    solver = fireflies(random_graph,population)
    new_order,cost = solver.run(iterations=200,beta=0.7)
    print('best fly: %s | cost: %d\n' % (new_order,cost))
    
