import importlib
from multiprocessing import get_context
from multiprocessing.connection import wait

import torch

from evo_utils import Utils


def evaluate_candidate(individual_id, gpu_id, data_lock, result):
    with result:
        module = importlib.import_module(f'scripts.{individual_id}')
        fitness = module.RunModel().do_work(gpu_id, individual_id, data_lock)
        result.send(fitness)


class FitnessEvaluate:
    def __init__(self, individuals, log, gpu_ids, fitness_cache):
        self.individuals = individuals
        self.log = log
        self.gpu_ids = gpu_ids
        self.fitness_cache = fitness_cache

    @staticmethod
    def visible_gpu_ids():
        gpu_ids = tuple(range(torch.cuda.device_count()))
        if not gpu_ids:
            raise RuntimeError('ArchCraft search requires at least one visible CUDA GPU.')
        return gpu_ids

    def evaluate(self):
        candidates = {}
        for individual in self.individuals:
            key, _ = individual.uuid()
            if key in self.fitness_cache:
                individual.acc = self.fitness_cache[key]
            else:
                candidates.setdefault(key, []).append(individual)

        if not candidates:
            return

        context = get_context('spawn')
        data_lock = context.Lock()
        pending = iter(candidates.items())
        running = {}

        def start_candidate(gpu_id):
            candidate = next(pending, None)
            if candidate is None:
                return
            key, individuals = candidate
            individual = individuals[0]
            Utils.generate_pytorch_file(individual)
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=evaluate_candidate,
                args=(individual.id, gpu_id, data_lock, sender),
            )
            try:
                process.start()
                running[process.sentinel] = (process, receiver, gpu_id, key)
            except BaseException:
                receiver.close()
                if process.pid is not None:
                    process.terminate()
                    process.join()
                process.close()
                raise
            finally:
                sender.close()
            self.log.info(f'Evaluating {individual.id} on CUDA GPU {gpu_id}')

        try:
            for gpu_id in self.gpu_ids:
                start_candidate(gpu_id)
            while running:
                for sentinel in wait(running):
                    process, receiver, gpu_id, key = running.pop(sentinel)
                    individual_id = candidates[key][0].id
                    with receiver:
                        process.join()
                        exit_code = process.exitcode
                        process.close()
                        if exit_code != 0:
                            raise RuntimeError(
                                f'{individual_id} failed on CUDA GPU {gpu_id} (exit code {exit_code}).'
                            )
                        try:
                            fitness = receiver.recv()
                        except EOFError as error:
                            raise RuntimeError(f'{individual_id} returned no fitness.') from error

                    self.fitness_cache[key] = fitness
                    for individual in candidates[key]:
                        individual.acc = fitness
                    self.log.info(f'{individual_id} AIA: {fitness:.3f}%')
                    start_candidate(gpu_id)
        finally:
            for process, _, _, _ in running.values():
                if process.is_alive():
                    process.terminate()
            for process, receiver, _, _ in running.values():
                process.join()
                process.close()
                receiver.close()
