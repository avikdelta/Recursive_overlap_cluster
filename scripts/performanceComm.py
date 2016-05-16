import sys
from metric_lib import *
import scipy
import scipy.io

if len(sys.argv) < 4:
    print 'Insufficient arguments ...'
    exit
else:
    settingsFile = sys.argv[1]
    commFile = sys.argv[2]
    ftype = int(sys.argv[3]) # ftype = 1 means node start from 1

print 'Loading config ....'
cfg = config(settingsFile)
print 'Edgelist file =', cfg.edgelist_file
print 'Number of nodes =', cfg.num_nodes
print 'Number of edges =', cfg.num_edges

print 'Loading edge list ...'
smat = edgelist2smat(cfg, 1,False)
print 'File loaded. '

smat = smat.tocsr()

fr = open(commFile,'r')
K = 0
for line in fr:
    line = line.strip()
    if len(line)>0:
        K += 1
fr.close()
print 'Number of communities =', K
computeCommMetrics(commFile, ftype, smat)

