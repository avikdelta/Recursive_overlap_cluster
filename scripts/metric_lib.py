import numpy as np
import scipy
import scipy.io
import sys
import matplotlib.pyplot as mp
from collections import defaultdict

class config:

    def __init__(self,configFile):

        fr = open(configFile,'r')

        # Line 1
        line = fr.readline()
        line = line.strip().split()
        self.threshold_init = int(line[1])

        # Line 2
        line = fr.readline()
        line = line.strip().split()
        self.dense_threshold = int(line[1])

        # Line 3
        line = fr.readline()
        line = line.strip().split()
        self.max_cluster_iter = int(line[1])

        # Line 4
        line = fr.readline()
        line = line.strip().split()
        self.K_max = int(line[1])

        # Line 5
        line = fr.readline()
        line = line.strip().split()
        self.sparse_svd_rank = int(line[1])

        # Line 6
        line = fr.readline()
        line = line.strip().split()
        self.default_num_threads = int(line[1])

        # Line 7, 8
        line = fr.readline()
        line = fr.readline()

        # Line 9
        line = fr.readline()
        line = line.strip().split()
        self.key = line[1]

        # Line 10, 11, 12
        line = fr.readline()
        line = fr.readline()
        line = fr.readline()

        # Line 13
        line = fr.readline()
        line = line.strip().split()
        self.edgelist_file = line[1]

        # Line 14
        line = fr.readline()
        line = line.strip().split()
        self.num_nodes = int(line[1])

        # Line 15
        line = fr.readline()
        line = line.strip().split()
        self.num_edges = int(line[1])
        
        fr.close()

def genGraphDict(edgelistFile):

    G = defaultdict(list)
    fr = open(edgelistFile,'r')
    for line in fr:
        line = line.strip().split()
        x = line[0]
        y = line[1]
        G[x].append(y)
        G[y].append(x)
        
    fr.close()
    
    return G

def genSnapEdgelistFile(edgelistFile,num_nodes,num_edges):

    snapEdgelistFile = 'snap_'+edgelistFile
    G = genGraphDict(edgelistFile)
    print 'Genreating SNAP edgelist file ...'
    fw = open(snapEdgelistFile,'w')
    fw.write("# Directed Node Graph\n") 
    fw.write("# Autonomous systems (graph is undirected, each edge is saved twice\n")
    fw.write("# Nodes: "+str(num_nodes)+"    Edges: "+str(num_edges)+"\n")
    fw.write("# SrcNId	DstNId\n")
    for i in range(num_nodes):
        x = str(i+1)
        for y in G[x]:
            fw.write(x+'\t'+y+'\n')
            
    fw.close()
    print 'File saved.'

    return snapEdgelistFile
        
def metis2edgelist(metisGFile, ftype, fkey, fsave):

    print 'Converting ...'
    fr = open(metisGFile,'r')
    edgelistFile = 'A_'+metisGFile
    dimFile = 'dim_'+metisGFile
    fwe = open(edgelistFile,'w')
    fwd = open(dimFile,'w')

    line = fr.readline().strip()
    n = int(line.split()[0])
    nedges = int(line.split()[1])

    fwd.write(str(n)+'\n')
    fwd.write(str(nedges))
    fwd.close()

    sMat = scipy.sparse.lil_matrix((n,n))
    nidx = 0
    edgeDict = {}
    for line in fr:
        nidx += 1
        line = line.strip()
        arr = line.split()

        for snb in arr:
            nb = int(snb)
            if ftype == 1:
                sMat[nidx-1,nb-1] = 1
                #sMat[nb-1,nidx-1] = 1
                if nb < nidx:
                    key = str(nb)+','+str(nidx)
                else:
                    key = str(nidx)+','+str(nb)
                edgeDict[key] = 1
            else:
                sMat[nidx-1,nb] = 1
                #sMat[nb,nidx-1] = 1
                if nb < nidx-1:
                    key = str(nb+1)+','+str(nidx)
                else:
                    key = str(nidx)+','+str(nb+1)
                edgeDict[key] = 1

    
    fr.close()
    outputFile = fkey+'_A.mat'
    if fsave:
        scipy.io.savemat(outputFile, {'A' : sMat}, oned_as='column')
        print 'File saved !'

    # Write edgelist file

#    for key in edgeDict.keys():
#
#        snode1 = key.split(',')[0]
#        snode2 = key.split(',')[1]
#        fwe.write(snode1+' '+snode2+'\n')

    if n>10:
        step = n/10 + 1
    else:
        step = 2

    sMat = sMat.tocsr()
    for i in range(n):
        if i%step == 0:
            print 'writing nodes ', i+1, 'to', i+step

        begin = sMat.indptr[i]
        end = sMat.indptr[i+1]
        for j in sMat.indices[begin:end]:

            if j>i:
                key = str(i+1)+','+str(j+1)
                fwe.write(str(i+1)+' '+str(j+1)+'\n')
            
    fwe.close()
    print 'Edgelist file written!'
    print 'Number of nodes = ' + str(n)
    print 'Number of edges = ' + str(nedges)

    return      
        

def edgelist2smat(config, ftype, fsave):

    fr = open(config.edgelist_file,'r')
    n = config.num_nodes
    nedges = config.num_edges

    sparseMat = scipy.sparse.lil_matrix((n,n))

    for line in fr:

        line = line.strip()
        rs, cs = line.split()
        n1 = int(rs)
        n2 = int(cs)
        if ftype == 1:
            sparseMat[n1-1,n2-1] = 1
            sparseMat[n2-1,n1-1] = 1
        else:
            sparseMat[n1,n2] = 1
            sparseMat[n2,n1] = 1
            
    fr.close()
    outputFile = config.key+'_A.mat'
    if fsave:
        scipy.io.savemat(outputFile, {'A' : sparseMat}, oned_as='column')
        print 'File saved !'
        
    print 'Number of nodes = ' + str(n)
    print 'Number of edges = ' + str(nedges)

    return sparseMat

def pr(trueList,predList):

    totPos = len(trueList)
    totPredPos = len(predList)
    truePos = len(set(trueList)&set(predList))
    precision = truePos/float(totPos)
    recall = truePos/float(totPredPos)

    return (precision,recall)

def scoreF1(trueList,predList):

    (p,r) = pr(trueList,predList)

    if p+r==0:
        return 0
    else:
        return 2*p*r/float(p+r)

    
def cutRatio(nodeIndex, sMatCSR):

    n = sMatCSR.shape[0]
    d = len(nodeIndex)
    degOut = np.zeros(d,dtype=np.int64);
    inCluster = np.zeros(n)
    for i in nodeIndex:
        inCluster[i] = 1
        
    for i in range(d):

        node = nodeIndex[i]
        begin = sMatCSR.indptr[node]
        end = sMatCSR.indptr[node+1]
        for j in sMatCSR.indices[begin:end]:

            if inCluster[j] == 0:
                degOut[i] += 1

    cr = sum(degOut)/float(d*(n-d))
    return cr

def conductance(nodeIndex, sMatCSR):

    n = sMatCSR.shape[0]
    d = len(nodeIndex)
    degOut = np.zeros(d,dtype=np.int64);
    degIn = np.zeros(d,dtype=np.int64);
    inCluster = np.zeros(n)
    for i in nodeIndex:
        inCluster[i] = 1
        
    for i in range(d):

        node = nodeIndex[i]
        begin = sMatCSR.indptr[node]
        end = sMatCSR.indptr[node+1]
        for j in sMatCSR.indices[begin:end]:

            if inCluster[j] == 0:
                degOut[i] += 1
            else:
                degIn[i] += 1

    cs = float(sum(degOut))
    ms = sum(degIn)/float(2)
    if 2*ms+cs>0:
        cd = cs/float(2*ms+cs)
    else:
        cd = 0

    return cd
    
def FOMD(nodeIndex, deg, sMatCSR):

    n = sMatCSR.shape[0]
    d = len(nodeIndex)
    degCluster = np.zeros(d,dtype=np.int64)
    inCluster = np.zeros(n)
    for i in nodeIndex:
        inCluster[i] = 1

    for i in range(d):

        node = nodeIndex[i]
        begin = sMatCSR.indptr[node]
        end = sMatCSR.indptr[node+1]
        for j in sMatCSR.indices[begin:end]:

            if inCluster[j] == 1:
                degCluster[i] += 1

    medianDegree = np.median(deg)
    count = 0
    for degree in degCluster:

        if degree >= medianDegree:
            count += 1

    fomd = count/float(d)

    return fomd

def nchoose2(arr):

    n = len(arr)
    c = n*(n-1)/2
    C = np.zeros((c,2))
    idx = 0
    for i in range(n):
        for j in range(i+1,n):

            x = arr[i]
            y = arr[j]
            C[idx,0] = x
            C[idx,1] = y
            idx += 1

    return C

def MOD(nodeIndex, degDict, numEdges, sMatCSR):

    n = sMatCSR.shape[0]
    d = len(nodeIndex)
    m = numEdges
    C = nchoose2(nodeIndex)
    mod = 0
    c = d*(d-1)/2
    nbdict = defaultdict(list)
    for node in nodeIndex:

        begin = sMatCSR.indptr[node]
        end = sMatCSR.indptr[node+1]
        for j in sMatCSR.indices[begin:end]:
            nbdict[node].append(j)    

    mod = 0
    for i in range(c):
        x = C[i,0]
        y = C[i,1]

        if y in nbdict[x]:
            Axy = 1
        else:
            Axy = 0

        mod += Axy - degDict[x]*degDict[y]/float(2*m)

    mod = mod/float(2*m)

    return mod
    

def TPR(nodeIndex, sMatCSR):

    n = sMatCSR.shape[0]
    d = len(nodeIndex)
    inCluster = np.zeros(n)
    for i in nodeIndex:
        inCluster[i] = 1

    tpr = 0
    for node in nodeIndex:

        begin = sMatCSR.indptr[node]
        end = sMatCSR.indptr[node+1]
        nblist = []
        for j in sMatCSR.indices[begin:end]:

            if inCluster[j] == 1:
                nblist.append(j)

        indeg = len(nblist)
        if indeg<2:
            continue
            
        c = indeg*(indeg-1)/2
        C = nchoose2(nblist)
        istp = False
        for k in range(c):

            nb1 = C[k,0]
            nb2 = C[k,1]
            begin1 = sMatCSR.indptr[nb1]
            end1 = sMatCSR.indptr[nb1+1]

            if nb2 in sMatCSR.indices[begin1:end1]:
                istp = True

        if istp==True:
            tpr += 1

    tpr = tpr/float(d)
    return tpr

def findDegree(sMatCSR):

    n = sMatCSR.shape[0]
    deg = np.zeros(n,dtype=np.int64)

    for i in xrange(sMatCSR.indptr.size-1):

        begin = sMatCSR.indptr[i]
        end = sMatCSR.indptr[i+1]
        deg[i] = end-begin

    return deg

def degreeDict(sMatCSR):

    n = sMatCSR.shape[0]
    deg = {}
    count = 0
    for i in xrange(sMatCSR.indptr.size-1):

        begin = sMatCSR.indptr[i]
        end = sMatCSR.indptr[i+1]
        deg[i] = end-begin
        count += deg[i]

    return (deg,count/2)
        

def CommCutRatio(commFile, ftype, sMatCSR):

    fr = open(commFile,'r')
    crlist = []
    for line in fr:

        line = line.strip()
        arr = line.split()
        nodeIndex = []
        for i in arr:

            if ftype == 1:
                node = int(i)-1
            else:
                node = int(i)

            nodeIndex.append(node)

        cr = cutRatio(nodeIndex,sMatCSR)
        crlist.append(cr)

    fr.close()
    return np.mean(crlist)
            
def CommConductance(commFile, ftype, sMatCSR):

    fr = open(commFile,'r')
    clist = []
    for line in fr:

        line = line.strip()
        arr = line.split()
        nodeIndex = []
        for i in arr:

            if ftype == 1:
                node = int(i)-1
            else:
                node = int(i)

            nodeIndex.append(node)

        c = conductance(nodeIndex,sMatCSR)
        clist.append(c)

    fr.close()
    return np.mean(clist)

def CommFOMD(commFile, ftype, sMatCSR):

    degree = findDegree(sMatCSR)

    fr = open(commFile,'r')
    fomdlist = []
    for line in fr:

        line = line.strip()
        arr = line.split()
        nodeIndex = []
        for i in arr:

            if ftype == 1:
                node = int(i)-1
            else:
                node = int(i)

            nodeIndex.append(node)

        f = FOMD(nodeIndex,degree,sMatCSR)
        fomdlist.append(f)

    fr.close()
    return np.mean(fomdlist)

def CommTPR(commFile, ftype, sMatCSR):

    fr = open(commFile,'r')
    tprlist = []
    for line in fr:

        line = line.strip()
        arr = line.split()
        nodeIndex = []
        for i in arr:

            if ftype == 1:
                node = int(i)-1
            else:
                node = int(i)

            nodeIndex.append(node)

        tpr = TPR(nodeIndex,sMatCSR)
        tprlist.append(tpr)

    fr.close()
    return np.mean(tprlist)

def CommMOD(commFile, ftype, sMatCSR):

    (degDict,num_edges) = degreeDict(sMatCSR)
    fr = open(commFile,'r')
    modlist = []
    for line in fr:

        line = line.strip()
        arr = line.split()
        nodeIndex = []
        for i in arr:

            if ftype == 1:
                node = int(i)-1
            else:
                node = int(i)

            nodeIndex.append(node)

        mod = MOD(nodeIndex, degDict, num_edges, sMatCSR)
        modlist.append(mod)

    fr.close()
    #return np.mean(modlist)
    return np.sum(modlist)

def CommF1(commFile,groundTruthFile):

    frc = open(commFile,'r')
    Kest = 0
    cDict = defaultdict(list)
    print 'Reading communities ...'
    for line in frc:

        line = line.strip().split()
        if len(line)==0:
            continue

        Kest += 1
        for snode in line:
            cDict[Kest].append(int(snode))
            
    frc.close()

    frg = open(groundTruthFile,'r')
    K = 0
    gDict = defaultdict(list)
    print 'Reading ground truth ...'
    for line in frg:

        line = line.strip().split()
        if len(line)==0:
            continue

        K += 1
        for snode in line:
            gDict[K].append(int(snode))
            
    frg.close()

    cMaxList = np.zeros(Kest)
    gMaxList = np.zeros(K)
    print 'Computing F1 ...'
    for i in range(Kest):

        commList = cDict[i+1]
        cMax = 0
        for j in range(K):

            gtList = gDict[j+1]
            f1 = scoreF1(gtList,commList)
            if f1>gMaxList[j]:
                gMaxList[j] = f1

            if f1>cMax:
                cMax = f1

        cMaxList[i] = cMax
    #print cMaxList
    #print gMaxList
    cMean = np.mean(cMaxList)
    gMean = np.mean(gMaxList)

    return (cMean+gMean)/float(2)

        
def computeCommMetrics(commFile, ftype, sMatCSR):

    print '-----------------------------------'
    print 'Computing community metrics ...'
    print '-----------------------------------'
    
    cr = CommCutRatio(commFile, ftype, sMatCSR)
    print 'Cut Ratio = ', cr

    c = CommConductance(commFile, ftype, sMatCSR)
    print 'Conductance = ', c

    f = CommFOMD(commFile, ftype, sMatCSR)
    print 'FOMD = ', f

    t = CommTPR(commFile, ftype, sMatCSR)
    print 'TPR = ', t
    
    m = CommMOD(commFile, ftype, sMatCSR)
    print 'MOD = ', m
    print '-----------------------------------'

    return

def estimateEdgeDensity(trainCommIndex, groundTruthCommFile, ftype, sMatCSR):

    fr = open(groundTruthCommFile,'r')
    K = 0
    plist = []
    for line in fr:

        K += 1
        if K in trainCommIndex:

            arr = line.strip().split()
            nodeIndex = []
            for i in arr:

                if ftype == 1:
                    node = i-1
                else:
                    node = i

                nodeIndex.append(node)

            S = sMatCSR[nodeIndex,:].tocsc()[:,nodeIndex].tocsr()
            d = len(nodeIndex)
            deg = findDegree(S)
            phat = sum(deg)/float(d*(d-1))
            plist.append(phat)

            
    fr.close()
    p = np.mean(plist)

    return p

def plotDegreeDist(edgelistFile,key):

    smat = edgelist2smat(edgelistFile, 1, key, False)
    print 'File loaded. '
    smat = smat.tocsr()
    deg = findDegree(smat)
    n = len(deg)
    # Histogram
    mp.figure()
    n, bins, patches = mp.hist(deg, 40, facecolor='green', alpha=0.75)
    mp.xlabel('degree')
    mp.ylabel('frequency')
    figFilePath = key+'_degree_dist.eps'
    mp.savefig(figFilePath,bbox_inches='tight')
    figFilePath2 = key+'_degree_dist.png'
    mp.savefig(figFilePath2,bbox_inches='tight')

    #stats
    mean_deg = np.mean(deg)
    median_deg = np.median(deg)
    sd_deg = np.sqrt(np.var(deg))

    print 'mean = ', mean_deg
    print 'median = ',median_deg
    print 'sd = ', sd_deg

    return

def test(settingsFile):

    print 'Loading config ....'
    cfg = config(settingsFile)
    print 'Edgelist file =', cfg.edgelist_file
    print 'Number of nodes =', cfg.num_nodes
    print 'Number of edges =', cfg.num_edges

    print 'Loading edge list ...'
    smat = edgelist2smat(cfg, 1,False)
    print 'File loaded. '
    smat = smat.tocsr()
    (degDict,num_edges) = degreeDict(smat)
    print 'Num edges =',num_edges
    comm = [2,3,4]
    print 'Modularity =', MOD(comm, degDict, num_edges, smat)
    
if __name__ == "__main__":

    # Examples
    test('config_test.txt')

    
