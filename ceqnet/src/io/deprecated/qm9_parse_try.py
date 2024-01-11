from depr.grand.src.io import qm9_parse

data = qm9_parse("Users/martin/Master/git/master_thesis/venv/lib/python3.10/site-packages/dmol-book/applied/qm9.tfrecords")

for d in data:
    print(d)
    break