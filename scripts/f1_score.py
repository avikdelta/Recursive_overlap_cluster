import sys
import os
import time
from metric_lib import CommF1

if len(sys.argv)<3:
    print 'Insufficient arguments !'
    exit
else:
    commFile = sys.argv[1]
    groundTruthFile = sys.argv[2]
    #key = sys.argv[3]

    outputFile = 'f1_'+os.path.splitext(commFile)[0]

    start = time.time()
    f1 = CommF1(commFile,groundTruthFile)
    runtime = (time.time()-start)

    fw = open(outputFile,'w')
    fw.write('Comm file = '+commFile+'\n')
    fw.write('Ground truth file = '+groundTruthFile+'\n')
    fw.write('Average F1 score = '+str(f1)+'\n')
    fw.write('F1 computation time = '+str(runtime)+' s')
    fw.close()
    print "---------------------------"
    print "F1 score =", f1
    print "Total time = ", runtime
    print "---------------------------"
    print 'Result saved !'
