# Recursive Overlap Graph Clustering

A C++ implementation of a recursive parallel overlap graph clustering algorithm

COPYRIGHT (C) 2016-present by Avik Ray:

`RedSVD.h` copyright belongs to Okanohara Daisuke and Nicolas Tessore. Remaining all files copyright belongs to Avik Ray. The program and codes can be used under the Apache License, Version 2.0

contact: avik@utexas.edu

Please cite the reference below if used.

- Avik Ray, Javad Ghaderi, Sujay Sanghavi and Sanjay Shakkottai, "Overlap Graph Clustering via Successive Removal", Proceedings of 52nd Annual Allerton Conference (Allerton 2014), Allerton, USA.

## Folder description 

1. **base** -- Contains the main parallel implementation of overlap cluster algorithm  which also logs the communities on the fly as it discovers them

2. **scripts** -- Contains some python scripts to preprocess data and evaluate community performance metrics e.g. cut-ratio (CR), conductance (C), triangle participation ratio (TPR), fraction over median degree (FOMD),vmodularity (MOD), and average F1 score (F1).

3. **data** -- Contains synthetic graph dataset and corresponding config file

## Requirements 

1. Latest C++ compiler with OpenMP

2. Latest Eigen (download from http://eigen.tuxfamily.org/index.php?title=Main_Page)

## Compilation 

1. Install eigen to /path/to/eigen/

2. Go to directory "base"

3. For intel icc compiler use command

```
icc -I /path/to/eigen -openmp -o recoverlapcluster recoverlapcluster_main.cpp
```

Similar commands should work for gcc, g++ compilers

## Running algorithm

Files required in the same directory:

1. **edgeListFile**  -- This is an edgelist file containing the edges in the graph, each edge in one line. The nodes should be numbered starting from 1. Also the edges are arranged in increasing order of their number. For example if the graph is a triangle with 2 nodes 1,2,3 the edgelist files has
```
1 2
1 3
2 3
```
For a square with 4 nodes 1,2,3,4
```
1 2
1 3
2 3
3 4
```
and so on.

2. **config_file** -- This is a configuration file with several program settings (description below). Note that some edgelist files start from node number 0. For example in the DPLB dataset in SNAP repository (http://snap.stanford.edu/data/). Such files require conversion to the required format where nodes start from 1.


### Running the algorithm:

1. The program can be run using the command

```
./recoverlapcluster config_file num_threads
```

- **config_file** -- Name of the test configuration file

- **num_threads** -- Positive integer, number of parallel threads used to 
               run the algorithm 

#### Example: 

To run recursive overlap clustering on a synthetic test graph with 1000 nodes nodes and 5 communities.

Copy the edgelist file `data/ovp_n1000K5p8q1.txt` and its corresponding config file `data/config_ovp_n1000K5p8q1.txt` to the base folder. Then to run with a single thread we do

```
./recoverlapcluster config_ovp_n1000K5p8q1.txt 1
```

**Outputs:**

1. **C_testID_edgeListFile** -- File containing the communities, each line contains nodes in one community

2. **log_testID_edgeListFile** -- log file, each line has num_remaining_nodes time_in_sec num_recovered_communities

In addition when `LOGS_ON` is set to 1 in config_file the program outputs the community file on the go as it discovers them. The file is `CLOG_testID_edgeListFile`. Note that testID is also an unique string defined in the config_file.


## Community quality evaluation

Copy the scripts performanceComm.py, f1_score.py, and metric_lib.py in the folder containing the config file, edgeListFile and community output file (C_testID_edgeListFile). Run
```
python performanceComm.py config_file C_testID_edgeListFile 1
```
To test average F1 score with respect to the ground truth, run
```
python f1_score.py C_testID_edgeListFile ground_truth_community_file
```

## Config File Parameters

- THRESHOLD_INIT -- Initial degree threshold in each recursive call
- DENSE_THRESHOLD -- Dimension till which the program performs full SVD
- MAX_CLUSTER_ITER -- Number of iterations performed by ClusterCP subroutine
- K_MAX -- Default maximum number of communities in config class. (Currently not used) 
- SPARSE_SVD_RANK -- The default rank of sparse SVD
- DEFAULT_NUM_THREADS -- Default number of threads/core used
- VERBOSE_ON -- Flag to turn on verbose mode
- LOGS_ON -- Flag to enable output of communities on the fly as they are discovered.
- TEST_NUMBER -- Unique string to identify an experiment
- p -- Within community edge density
- q -- Across community edge density
- GAMMA -- Minimum size of a community (typically use 3 or more on real datasets)
- EDGELIST_FILE -- Name of edgelist file
- NUM_NODES -- Number of nodes in the graph
- NUM_EDGES -- Number of edges in the graph

