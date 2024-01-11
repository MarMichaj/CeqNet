import numpy as np
from depr.grand.src.io import qm9_parse, qm9_fetch
#import dmol

qm9_records = qm9_fetch()
data = qm9_parse(qm9_records)

def convert_record(d):
    # break up record
    (e, x), y = d
    #
    e = e.numpy()
    x = x.numpy()
    r = x[:, :3]
    # make ohc size larger
    # so use same node feature
    # shape later
    ohc = np.zeros((len(e), 16))
    ohc[np.arange(len(e)), e - 1] = 1
    return (ohc, r), y.numpy()[13]


for d in data:
    (e, x), y = convert_record(d)
    print("Element one hots\n", e)
    print("Coordinates\n", x)
    print("Label:", y)
    break


print(data.shape)