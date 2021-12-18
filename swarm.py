# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:48:42 2021

@author: mohamed elhakim
"""
import random
import math
import operator
from operator import attrgetter
import collections
import copy
import sys, time
def hamming_distance_with_info(a, b):
	"return number of places and places where two sequences differ"
	assert len(a) == len(b)
	ne = operator.ne
	differ = list(map(ne, a, b))
	return sum(differ), differ

def hamming_distance(a, b):
	dist, info = hamming_distance_with_info(a, b)
	return dist


    
class Graph:

	def __init__(self, amount_vertices):
		self.edges = {} # dictionary of edges
		self.vertices = set() # set of vertices
		self.amount_vertices = amount_vertices # amount of vertices


	# adds a edge linking "src" in "dest" with a "cost"
	def addEdge(self, src, dest, cost = 0):
		# checks if the edge already exists
		if not self.existsEdge(src, dest):
			self.edges[(src, dest)] = cost
			self.vertices.add(src)
			self.vertices.add(dest)


	# checks if exists a edge linking "src" in "dest"
	def existsEdge(self, src, dest):
		return (True if (src, dest) in self.edges else False)


	# shows all the links of the graph
	def showGraph(self):
		print('Showing the graph:\n')
		for edge in self.edges:
			print('%d linked in %d with cost %d' % (edge[0], edge[1], self.edges[edge]))

	# returns total cost of the path
	def getCostPath(self, path):
		
		total_cost = 0
		for i in range(self.amount_vertices - 1):
			total_cost += self.edges[(path[i], path[i+1])]

		# add cost of the last edge
		total_cost += self.edges[(path[self.amount_vertices - 1], path[0])]
		return total_cost


	# gets random unique paths - returns a list of lists of paths
	def getRandomPaths(self, max_size):

		random_paths, list_vertices = [], list(self.vertices)

		initial_vertice = random.choice(list_vertices)
		if initial_vertice not in list_vertices:
			print('Error: initial vertice %d not exists!' % initial_vertice)
			sys.exit(1)

		list_vertices.remove(initial_vertice)
		list_vertices.insert(0, initial_vertice)

		for i in range(max_size):
			list_temp = list_vertices[1:]
			random.shuffle(list_temp)
			list_temp.insert(0, initial_vertice)

			if list_temp not in random_paths:
				random_paths.append(list_temp)

		return random_paths
class CompleteGraph(Graph):

	# generates a complete graph
	def generates(self):
		for i in range(self.amount_vertices):
			for j in range(self.amount_vertices):
				if (i != j)&(j>i):
					weight = random.randint(1, 10)
					self.addEdge(j, i, int(weight))
					self.addEdge(i, j, int(weight))


# class that represents a particle
class Particle:

	def __init__(self, solution, cost):

		# current solution
		self.solution = solution

		# best solution (fitness) it has achieved so far
		self.pbest = solution

		# set costs
		self.cost_current_solution = cost
		self.cost_pbest_solution = cost

		# velocity of a particle is a sequence of 4-tuple
		# (1, 2, 1, 'beta') means SO(1,2), prabability 1 and compares with "beta"
		self.velocity = []

	# set pbest
	def setPBest(self, new_pbest):
		self.pbest = new_pbest

	# returns the pbest
	def getPBest(self):
		 return self.pbest
        

	# set the new velocity (sequence of swap operators)
	def setVelocity(self, new_velocity):
		self.velocity = new_velocity

	# returns the velocity (sequence of swap operators)
	def getVelocity(self):
		return self.velocity

	# set solution
	def setCurrentSolution(self, solution):
		self.solution = solution

	# gets solution
	def getCurrentSolution(self):
		return self.solution

	# set cost pbest solution
	def setCostPBest(self, cost):
		self.cost_pbest_solution = cost

	# gets cost pbest solution
	def getCostPBest(self):
		return self.cost_pbest_solution

	# set cost current solution
	def setCostCurrentSolution(self, cost):
		self.cost_current_solution = cost

	# gets cost current solution
	def getCostCurrentSolution(self):
		return self.cost_current_solution

	# removes all elements of the list velocity
	def clearVelocity(self):
		del self.velocity[:]

class PSO:

	def __init__(self, graph, iterations, population, beta=1, alfa=1):
		self.graph=graph
		self.iterations = iterations # max of iterations
		self.size_population = len(population) # size population
		self.particles = [] # list of particles
		self.beta = beta # the probability that all swap operators in swap sequence (gbest - x(t-1))
		self.alfa = alfa # the probability that all swap operators in swap sequence (pbest - x(t-1))

		# initialized with a group of random particles (solutions)
		solutions = population    
		# checks if exists any solution
		if not solutions:
			print('Initial population empty! Try run the algorithm again...')
			sys.exit(1)

		# creates the particles and initialization of swap sequences in all the particles
		for solution in solutions:
			# creates a new particle
			particle = Particle(solution=solution, cost=graph.getCostPath(solution))
			# add the particle
			self.particles.append(particle)

		# updates "size_population"
		self.size_population = len(self.particles)


	# set gbest (best particle of the population)
	def setGBest(self, new_gbest):
		self.gbest = new_gbest

	# returns gbest (best particle of the population)
	def getGBest(self):
		return self.gbest


	# shows the info of the particles
	def showsParticles(self):

		print('Showing particles...\n')
		for particle in self.particles:
			print('pbest: %s\t|\tcost pbest: %d\t|\tcurrent solution: %s\t|\tcost current solution: %d' \
				% (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
							particle.getCostCurrentSolution()))
		print('')


	def run(self):
		self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))
		# for each time step (iteration)
		for t in range(self.iterations):

			# updates gbest (best particle of the population)
			self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))

			# for each particle in the swarm
			for particle in self.particles:

				particle.clearVelocity() # cleans the speed of the particle
				temp_velocity = []
				solution_gbest = copy.copy(self.gbest.getPBest()) # gets solution of the gbest
				solution_pbest = particle.getPBest()[:] # copy of the pbest solution
				solution_particle = particle.getCurrentSolution()[:] # gets copy of the current solution of the particle

				# generates all swap operators to calculate (pbest - x(t-1))
				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_pbest[i]:
						# generates swap operator
						swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)
						
						# append swap operator in the list of velocity
						temp_velocity.append(swap_operator)

						# makes the swap
						aux = solution_pbest[swap_operator[0]]
						solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
						solution_pbest[swap_operator[1]] = aux

				# generates all swap operators to calculate (gbest - x(t-1))
				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_gbest[i]:
						# generates swap operator
						swap_operator = (i, solution_gbest.index(solution_particle[i]), self.beta)

						# append swap operator in the list of velocity
						temp_velocity.append(swap_operator)

						# makes the swap
						aux = solution_gbest[swap_operator[0]]
						solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
						solution_gbest[swap_operator[1]] = aux

				
				# updates velocity
				particle.setVelocity(temp_velocity)

				# generates new solution for particle
				for swap_operator in temp_velocity:
					if random.random() <= swap_operator[2]:
						# makes the swap
						aux = solution_particle[swap_operator[0]]
						solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
						solution_particle[swap_operator[1]] = aux
				
				# updates the current solution
				particle.setCurrentSolution(solution_particle)
				# gets cost of the current solution
				cost_current_solution = self.graph.getCostPath(solution_particle)
				# updates the cost of the current solution
				particle.setCostCurrentSolution(cost_current_solution)

				# checks if current solution is pbest solution
				if cost_current_solution < particle.getCostPBest():
					particle.setPBest(solution_particle)
					particle.setCostPBest(cost_current_solution)
                    
                    
class fireflies():
	def __init__(self, graph,population):
		"points is list of objects of type City"
		self.graph = graph
		self.absorptions = []
		self.population = population
		self.light_intensities = []
		self.best_solution_cost = None
		self.n = None

	def f(self, individual): # our objective function? lightness?
		"objective function - describes lightness of firefly"
		return self.graph.getCostPath(individual)

	def determine_initial_light_intensities(self):
		"initializes light intensities"
		self.light_intensities = [self.f(x) for x in self.population]

	def generate_initial_absorptions(self):
		for i in range(len(self.population)):
			self.absorptions.append(random.random()*0.9+0.1 )

	def check_if_best_solution(self, index):
		new_cost = self.light_intensities[index]
		if new_cost < self.best_solution_cost: 
			self.best_solution = copy.deepcopy(self.population[index])
			self.best_solution_cost = new_cost

	def find_global_optimum(self):
		"finds the brightest firefly"
		index = self.light_intensities.index(min(self.light_intensities))
		return index

	def move_firefly(self, a, b, r):
		"moving firefly a to b in less than r swaps"
		number_of_swaps = random.randint(0, r-2)    
		
		distance, diff_info = hamming_distance_with_info(self.population[a], self.population[b])
		while number_of_swaps > 0:
			distance, diff_info = hamming_distance_with_info(self.population[a], self.population[b])
			random_index = random.choice([i for i in range(len(diff_info)) if diff_info[i]])
			value_to_copy = self.population[b][random_index]
			index_to_move = self.population[a].index(value_to_copy)

			if number_of_swaps == 1 and self.population[a][index_to_move] == self.population[b][random_index] and self.population[a][random_index] == self.population[b][index_to_move]:
				break

			self.population[a][random_index], self.population[a][index_to_move] = self.population[a][index_to_move], self.population[a][random_index]
			if self.population[a][index_to_move] == self.population[b][index_to_move]:
				number_of_swaps -= 1
			number_of_swaps -= 1

		self.light_intensities[a] = self.f(self.population[a])

	def rotate_single_solution(self, i, value_of_reference):
		point_of_reference = self.population[i].index(value_of_reference)
		self.population[i] = collections.deque(self.population[i])
		l = len(self.population[i])
		number_of_rotations = (l - point_of_reference) % l
		self.population[i].rotate(number_of_rotations)
		self.population[i] = list(self.population[i])

	def rotate_solutions(self, value_of_reference):
		for i in range(len(self.population)):
			self.rotate_single_solution(i, value_of_reference)

	def I(self, index, r):
		return self.light_intensities[index] * math.exp(-1.0 * r)

	def run(self, iterations=200, beta=0.7):
		"gamma is parameter for light intensities, beta is size of neighbourhood according to hamming distance"
		# hotfix, will rewrite later
		value_of_reference = self.population[0][0]
		self.rotate_solutions(value_of_reference)   
		self.determine_initial_light_intensities()
		number_of_individuals=len(self.population)
		self.best_solution=self.population[self.find_global_optimum()] 
		self.best_solution_cost = self.graph.getCostPath(self.best_solution)
		self.generate_initial_absorptions()
		individuals_indexes = range(number_of_individuals)
		self.n = 0
		neighbourhood = beta * len(individuals_indexes)
		while self.n < iterations:
			for j in individuals_indexes:
				for i in individuals_indexes:
					r = hamming_distance(self.population[i], self.population[j])
					if self.I(i, r) > self.I(j, r) and r > neighbourhood:
						self.move_firefly(i, j, r)						
						self.check_if_best_solution(i)
			self.n += 1
		return self.best_solution,self.best_solution_cost
                   
def initial_path(graph,number_of_population):
    population = graph.getRandomPaths(number_of_population)
    return population