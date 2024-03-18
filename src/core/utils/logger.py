import os
import json
import sys
from . import constants as Constants


class MetricsLogger(object):
    def __init__(self, config : dict, dirname : str = None, pretrained : str = None):
        """
        TODO explain the parameter: `pretrained`
        """

        self.config = config

        if dirname is None:
            if pretrained is None:
                raise Exception('Either --dir or --pretrained needs to be specified.')
            self.dirname = pretrained
        else:
            self.dirname = dirname

            if os.path.exists(dirname):
                raise Exception('Directory already exists: {}'.format(dirname))
            
            os.makedirs(dirname)
            os.mkdir(os.path.join(dirname, 'metrics'))

        # 1. write `config` to the log file outside of the `metrics` folder
        self.write_dict_as_json(config, os.path.join(self.dirname, Constants._CONFIG_FILE))
        
        # 2. write all the metrics to the log file located in the `metrics` folder
        if config['logging']:
            self.metrics_log_file = open(os.path.join(self.dirname, 'metrics', 'metrics.log'), 'a')

    def write_dict_as_json(self, data : dict, filename : str, mode='w'):
        with open(filename, mode) as file:
            file.write(json.dumps(data, indent=4, ensure_ascii=False))

    def print_log_on_screen(self, data):
        print(data)

    def write_to_file(self, text):
        if self.config['logging']:
            self.metrics_log_file.writelines(text + '\n')
            self.metrics_log_file.flush()

    def close_met_log_file(self):
        if self.config['logging']:
            self.metrics_log_file.close()

class MessageLogger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        pass
