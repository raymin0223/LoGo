from .fedavg import FedAvg
from .fedprox import FedProx
from .scaffold import SCAFFOLD

_method_class_map = {
    'fedavg': FedAvg,
    'fedprox': FedProx,
    'scaffold': SCAFFOLD
}


def get_fl_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
