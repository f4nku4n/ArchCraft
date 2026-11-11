import random

from evo_utils import Log
from genetic.population import Population
from genetic.evaluate import FitnessEvaluate
from genetic.mutation import Mutation
import numpy as np
import copy

def run_evolve():
    params = {}
    params['pop_size'] = 10    # Population size
    params['max_gen'] = 20    # Maximum number of iteration generations
    evoCNN = EvolveCNN(params)
    evoCNN.do_work(params)

class EvolveCNN(object):
    def __init__(self, params):
        self.parent_pops = None
        self.params = params
        self.pops = None
        self.fitness_cache = {}

    def initialize_population(self):
        pops = Population(self.params, 0)
        pops.initialize()
        self.pops = pops

    def fitness_evaluate(self, gpu_ids):
        FitnessEvaluate(self.pops.individuals, Log, gpu_ids, self.fitness_cache).evaluate()

    def generate_offspring(self):
        cm = Mutation(Log,self.pops.individuals, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process(mut_offspring_num=self.params['pop_size'])
        
        _str = []
        for ind in offspring:
            _str.append(str(ind))
            _str.append('-' * 100)
        file_name = './populations/offspring_%02d.txt' % self.pops.gen_no
        with open(file_name, 'w') as f:
            f.write('\n'.join(_str))

        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        indi_list = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
        pop_size = self.params['pop_size']
        elitism = 0.2
        e_count = int(pop_size * elitism)
        indi_list.sort(key=lambda x: x.acc, reverse=True)
        # descending order
        next_individuals = indi_list[0:e_count]
        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)

        for i in range(pop_size - e_count):
            idx1 = random.randrange(0, len(left_list))
            indi1 = left_list.pop(idx1)
            idx2 = random.randrange(0, len(left_list))
            indi2 = left_list.pop(idx2)
            if indi1.acc > indi2.acc:
                next_individuals.append(indi1)
            else:
                next_individuals.append(indi2)

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops

        self.pops.gen_no += 1

    def do_work(self, params):
        gpu_ids = FitnessEvaluate.visible_gpu_ids()
        self.fitness_cache.clear()
        Log.info(f'Starting a new search on CUDA GPUs {gpu_ids}')
        self.initialize_population()

        Log.info('Evaluating the initial population')
        self.fitness_evaluate(gpu_ids)
        self.pops.gen_no += 1
        for generation in range(1, params['max_gen'] + 1):
            self.params['gen_no'] = generation
            Log.info(f'Generation {generation}: mutating the population and evaluating offspring')
            self.generate_offspring()
            self.fitness_evaluate(gpu_ids)
            self.environment_selection()

        best = max(self.pops.individuals, key=lambda individual: individual.acc)
        Log.info('Best architecture: %s' % best)


if __name__ == '__main__':
    run_evolve()
