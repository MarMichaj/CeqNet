from typing import Dict
from .embed import (AtomTypeEmbed,
                    HardnessEmbed,
                    GeometryEmbed,
                    kQeqHardnessEmbed,
                    ChargeEmbed,
                    QeqEmbed)



def get_embedding_module(name: str, h: Dict):
    if name == 'atom_type_embed':
        return AtomTypeEmbed(**h)
    elif name == 'hardness_embed':
        return HardnessEmbed(**h)
    elif name == 'kQeqHardness_embed':
        return kQeqHardnessEmbed(**h)
    elif name == 'tot_charge_embed':
        return ChargeEmbed(**h)
    elif name == 'geometry_embed':
        return GeometryEmbed(**h)
    elif name == 'qeq_embed':
        return QeqEmbed(**h)
    else:
        msg = "No embedding module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
