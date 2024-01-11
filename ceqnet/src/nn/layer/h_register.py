from typing import Dict
from .schnet_layer import SchNetLayer
from .att_schnet_layer import AttentionSchNetLayer
from .so3krates_layer import So3kratesLayer

def get_layer(name: str, h: Dict):
    if name == 'schnet_layer':
        return SchNetLayer(**h)
    if name == 'attention_schnet_layer':
        return AttentionSchNetLayer(**h)
    if name == 'so3krates_layer':
        return So3kratesLayer(**h)
    else:
        msg = "Layer with `module_name={}` is not implemented.".format(name)
        raise NotImplementedError(msg)
