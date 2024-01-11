from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList

qm9data = QM9(
    './qm9.db',
    batch_size=10,
    num_train=110000,
    num_val=10000,
    transforms=[ASENeighborList(cutoff=5.)]
)
qm9data.prepare_data()
qm9data.setup()

print('Number of reference calculations:', len(qm9data.dataset))
print('Number of train data:', len(qm9data.train_dataset))
print('Number of validation data:', len(qm9data.val_dataset))
print('Number of test data:', len(qm9data.test_dataset))
print('Available properties:')

for p in qm9data.dataset.available_properties:
    print('-', p)