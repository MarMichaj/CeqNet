from depr.grand.src.io import qm9_parse, qm9_fetch
#import dmol

qm9_records = qm9_fetch()
data = qm9_parse(qm9_records)


