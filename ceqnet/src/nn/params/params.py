from flax.core.frozen_dict import unfreeze, freeze

def iterate_weights(params, n):
    """
    :param params: dictionary containing network parameters. This must only contain one layer
    :param n: number of times layer parameters ought to be copied
    :return:
    """
    p = unfreeze(params)
    if len(p['params']) - 1 != 1:
        msg = 'Params-file contains weights for more than one layer. Weight iteration not possible'
        raise ValueError(msg)

    else:
        # we copy the params for the first layer n-1 times
        for i in range(1, n):
            p['params']['layers_' + str(i)] = p['params']['layers_0']

    return freeze(p)
