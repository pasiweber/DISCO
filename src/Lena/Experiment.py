import json
import mpire

from src.BaseExperiment import BaseExperiment
from src.filesystem import save_data


class Experiment(object):

    def __init__(self, exp_name):
        self.name = exp_name
        with open('config/experiments/{}.json'.format(exp_name)) as json_file:
            self.__exp_config = json.load(json_file)

        self.baseExperiments = [BaseExperiment(exp_name, dataname, min_points, self.__exp_config['repeat']) for dataname
                                in self.__exp_config['data'] for min_points in self.__exp_config['min_points']]
        print('Experiment defined')

    def run(self):
        print('run experiment')
        with mpire.WorkerPool(n_jobs=mpire.cpu_count(), daemon =False) as pool:
        #with mpire.WorkerPool(n_jobs=1, daemon=False) as pool:
            baseExperiments = pool.map(BaseExperiment.run, self.baseExperiments)
        results = [dicts for dict_list in baseExperiments for dicts in dict_list]
        save_data(results, self.name)
        print('finish experiment')

