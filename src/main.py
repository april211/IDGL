import argparse
import yaml
import torch
import numpy as np
from collections import defaultdict, OrderedDict

from core.model_handler import ModelHandler

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(simple_config : dict):
    print_config(simple_config)
    set_random_seed(simple_config['seed'])
    model = ModelHandler(simple_config)
    model.train()
    model.test()


def multi_run_main(complex_config : dict):
    """
    `complex_config`: a config dict contains some lists.

    You can add `--multi_run` in the command \
    to run multiple times with different random seeds. 
    
    Please see `config/cora/idgl.yml` for example. 
    
    TODO figure out how this works... 
    """
    print_config(complex_config)

    hyperparam_names = []
    for hyperparam_name, v in complex_config.items():
        if isinstance(v, list):
            hyperparam_names.append(hyperparam_name)

    scores = []
    simple_configs = gen_simple_configs(complex_config)

    for simple_config in simple_configs:
        for hyperparam_name in hyperparam_names:    # different dir for different config

            # The dir name only include parameters that are waiting to tune.
            simple_config['out_dir'] += '_{}_{}'.format(hyperparam_name, 
                                                            simple_config[hyperparam_name])
        
        print("\nOutput dir: " + simple_config['out_dir'])

        set_random_seed(simple_config['seed'])

        model = ModelHandler(simple_config)
        dev_metrics = model.train()
        test_metrics = model.test()
        scores.append(test_metrics[model.model.metric_name])

    print('Average score: {}'.format(np.mean(scores)))
    print('Std score: {}'.format(np.std(scores)))



################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml") -> dict:
    with open(config_path, "r") as setting:
        config = yaml.safe_load(setting)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (36 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def gen_simple_configs(complex_config : dict):
    """
    `complex_config`: a config dict contains some lists.

    This function split the complex config into multiple simple configs.

    If the value is not a list, then it is considered fixed.add()
    """
    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a : tuple):
            self.a = a

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively (max_depth == 2). 
        
        Accepts also `None` and returns always a (possibly empty) dictionary.
        """
        from functools import reduce
        def merge_two_dicts(x : dict, y : dict):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


    hyperparams = OrderedDict({k: v for k, v in complex_config.items() 
                                        if isinstance(v, list)})

    # convert tuple param into type `MncDc`
    for param_name, params in hyperparams.items():
        params_copy = []
        for param in params:
            params_copy.append(MncDc(param)             # ? `MncDc` just stores a tuple... 
                                    if isinstance(param, tuple) else param)
        hyperparams[param_name] = params_copy

    # each row is a group of hyperparameters
    param_groups = np.array(np.meshgrid(*hyperparams.values()), 
                                            dtype=object).T.reshape(-1, len(hyperparams.values()))
    return [merge_dicts(
        {k: v for k, v in complex_config.items() if not isinstance(v, list)},
        {k: row[i]() if isinstance(row[i], MncDc) else row[i] for i, k in enumerate(hyperparams)}
    ) for row in param_groups]


################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
    args = get_args()
    config = get_config(args['config'])
    if args['multi_run']:
        multi_run_main(config)
    else:
        main(config)
