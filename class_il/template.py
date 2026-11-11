"""
import json
import argparse
from datetime import datetime

import torch
from trainer import train

#generated_code

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/replay_for_evolve.json',
                        help='Json file of settings.')

    return parser

class TrainModel(object):
    def __init__(self, gpu_id, file_id):
        self.code = code
        self.file_id = file_id
        self.gpu_id = gpu_id

    def process(self, data_lock):
        args = setup_parser().parse_args()
        param = load_json(args.config)
        args = vars(args)  # Converting argparse Namespace to a dict.
        args.update(param)  # Add parameters from json

        args['device'] = [str(self.gpu_id)]
        args['depth'] = self.code[0]
        args['width'] = self.code[1]
        args['pool'] = self.code[2]
        args['double'] = self.code[3]
        return train(args, self.file_id, data_lock=data_lock)


    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()


class RunModel:
    def do_work(self, gpu_id, file_id, data_lock):
        torch.cuda.set_device(gpu_id)
        model = TrainModel(gpu_id, file_id)
        model.log_record(f'Using CUDA GPU {gpu_id}', first_time=True)
        return model.process(data_lock)
"""
