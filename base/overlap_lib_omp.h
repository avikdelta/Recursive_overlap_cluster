/* **************************************************************************
 * This file has various essential routines used by the overlap cluster 
 * algorithm
 *
 * Code by Avik Ray (avik@utexas.edu)
 *
 **************************************************************************** */

#ifndef OVERLAP_LIB_OMP_H
#define OVERLAP_LIB_OMP_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))


using namespace std;


/* config class */
class config_t{
public:
	long threshold_init;   // Beginning degree threshold
	long dense_threshold;  // Dimension till full SVD is performed
	long max_cluster_iter; // Maximum iteration in convex clustering subroutine
	long K_max;            // Maximum clusters to be found each iteration
	long sparse_svd_rank;  // Dimension from which sparse SVD is performed
	int num_threads;       // Default number of threads to use
	bool f_verbose;        // Flag to run in verbose mode
	bool f_log;            // Flag to log new clusters each iteration
	char testID[50];   // Unique string describing the experiment
	double p;	       // Value of parameter p
	double q;              // Value of parameter q
	long gamma;            // Value of parameter gamma
	char edgeListFile[100];// Name of edgelist file
        long nodes;            // Number of nodes
	long edges;            // Number of edges	
	config_t(): threshold_init(1), dense_threshold(250), max_cluster_iter(6), K_max(500), sparse_svd_rank(250), num_threads(1), f_verbose(true), f_log(true), p(0.1), q(0.01), gamma(5), nodes(0), edges(0) {}

	void init(){
	
		return;
	}

	void set_default(){
		threshold_init = 1;
	       	dense_threshold = 250;
	       	max_cluster_iter = 6;
	       	K_max = 500;
	       	sparse_svd_rank = 250;
		num_threads = 1;
		f_verbose = true;
	       	f_log = true;
		strcpy(testID,"001");
		p = 0.1;
		q = 0.01;
		gamma = 5;
		strcpy(edgeListFile,"A_test.txt");
		nodes = 0;
		edges = 0;
		return;
	}

	~config_t(){
	}

	void save(const char *filename){
		FILE *fp = fopen(filename,"w");
		fprintf(fp,"THRESHOLD_INIT %ld\n",threshold_init);
		fprintf(fp,"DENSE_THRESHOLD %ld\n",dense_threshold);
		fprintf(fp,"MAX_CLUSTER_ITER %ld\n",max_cluster_iter);
		fprintf(fp,"K_MAX %ld\n",K_max);
		fprintf(fp,"SPARSE_SVD_RANK %ld\n",sparse_svd_rank);
		fprintf(fp,"DEFAULT_NUM_THREADS %d\n",num_threads);
		fprintf(fp,"VERBOSE_ON %d\n",f_verbose);
		fprintf(fp,"LOGS_ON %d\n",f_log);
		fprintf(fp,"TEST_NUMBER %s\n",testID);
		fprintf(fp,"p %lf\n",p);
		fprintf(fp,"q %lf\n",q);
		fprintf(fp,"GAMMA %ld\n",gamma);
		fprintf(fp,"EDGELIST_FILE %s\n",edgeListFile);
		fprintf(fp,"NUM_NODES %ld\n",nodes);
		fprintf(fp,"NUM_EDGES %ld\n",edges);
	       	fclose(fp);
		return;
	}

	void print(){
		printf("\n *** TEST CONFIG SETTINGS ***");
		printf("\nTHRESHOLD_INIT %ld",threshold_init);
		printf("\nDENSE_THRESHOLD %ld",dense_threshold);
		printf("\nMAX_CLUSTER_ITER %ld",max_cluster_iter);
		printf("\nK_MAX %ld",K_max);
		printf("\nSPARSE_SVD_RANK %ld",sparse_svd_rank);
		printf("\nDEFAULT_NUM_THREADS %d",num_threads);
		printf("\nVERBOSE_ON %d",f_verbose);
		printf("\nLOGS_ON %d",f_log);
		printf("\nTEST_NUMBER %s",testID);
		printf("\np %lf",p);
		printf("\nq %lf",q);
		printf("\nGAMMA %ld",gamma);
		printf("\nEDGELIST_FILE %s",edgeListFile);
		printf("\nNUM_NODES %ld",nodes);
		printf("\nNUM_EDGES %ld",edges);
		printf("\n******************************\n");
		return;
	}

	void load(const char *filename){
		char buff[100];
		int _f_verbose, _f_log;
		FILE *fp = fopen(filename,"r");
		fscanf(fp,"%s %ld",buff,&threshold_init);
		fscanf(fp,"%s %ld",buff,&dense_threshold);
		fscanf(fp,"%s %ld",buff,&max_cluster_iter);
		fscanf(fp,"%s %ld",buff,&K_max);
		fscanf(fp,"%s %ld",buff,&sparse_svd_rank);
		fscanf(fp,"%s %d",buff,&num_threads);
		fscanf(fp,"%s %d",buff,&_f_verbose);
		fscanf(fp,"%s %d",buff,&_f_log);
		fscanf(fp,"%s %s",buff,testID);
		fscanf(fp,"%s %lf",buff,&p);
		fscanf(fp,"%s %lf",buff,&q);
		fscanf(fp,"%s %ld",buff,&gamma);
		fscanf(fp,"%s %s",buff,edgeListFile);
		fscanf(fp,"%s %ld",buff,&nodes);
		fscanf(fp,"%s %ld",buff,&edges);
		fclose(fp);
		
		if(_f_verbose==1){
			f_verbose = true;
		}
		else{
			f_verbose = false;
		}

		if(_f_log==1){
			f_log = true;
		}
		else{
			f_log = false;
		}


		return;
	}


};

struct qNode{
    long data;
    qNode* next;
    //qNode(long d, qNode* n=NULL): data(d), next(n){}
    qNode(long d): data(d), next(NULL){}
};


class queue_t{
    private:
        qNode* first;
        qNode* last;
    public:
	long length;
        queue_t() : first(NULL), last(NULL), length(0) {}
	~queue_t(){
		qNode *temp = first;
		while(temp!=NULL){
			first = first->next;
			delete temp;
			temp = first;
		}
	}

	void init()
	{
		first = NULL;
		last = NULL;
		length = 0;
		return;
	}

        void push(long x){
            if(first==NULL) {
                first=new qNode(x);
                last=first;
            }
            else {
                last->next=new qNode(x);
                last=last->next;
                }
	    length++;
        }   
        long pop(){
          if(first!=NULL){
            qNode* temp=first;
	    long val = temp->data;
            first=first->next;
            delete temp;
	    length--;
	    return(val);
          }
	  else{
		  return(-1);
	  }
        }
        void front(){
           if(first!=NULL) std::cout << first->data;
        }
        bool isempty(){
		if(first!=NULL)
		{
			return(false);
		}
		else
		{
			return(true);
		}
        }
	void print()
	{
		qNode *temp;
		printf("\nLength of queue = %ld",length);
		if(first==NULL)
			printf("\nQueue empty!");
		else
		{
			temp = first;
			while(temp!=NULL)
			{
				printf("\n%ld",temp->data);
				temp = temp->next;
			}
		}
	}
	void clear()
	{
		while(length>0)
		{
			long val = pop();
		}
	}
};

// Sparse matrix format CSR
class smat_t{
public:
	long rows, cols;
	long nnz, nedges;
	double *csr_val;
	long *csr_row_ptr;
	long *csr_col_idx;
	long *inv_map_index;
	long *original_index;
	bool mem_alloc_by_me;
	//-----------------------------------------------------
	smat_t(): csr_val(NULL), csr_row_ptr(NULL), csr_col_idx(NULL), inv_map_index(NULL), original_index(NULL), mem_alloc_by_me(false), rows(0), cols(0), nnz(0), nedges(0) {}
	smat_t(smat_t& m){ *this = m; mem_alloc_by_me = false; rows = 0; cols = 0; nnz=0; nedges = 0;}
	//-----------------------------------------------------
	void load(const config_t &config) {
		char fileAname[100];
		strcpy(fileAname,config.edgeListFile);
		// Read dimensions
		rows = config.nodes;
		nedges = config.edges;
		cols = rows;
		nnz = 2*nedges;

		printf("\nNumber of nodes = %ld",rows);
		printf("\nNumber of edges = %ld",nedges);

		printf("\nLoading data ...");
		// Allocate memory
		mem_alloc_by_me = true;
		csr_val = new double[nnz];
		csr_col_idx = new long[nnz];
		csr_row_ptr = new long[rows+1];
		long *temp_csr_row_ptr = new long[rows+1];
		memset(csr_row_ptr,0,sizeof(long)*(rows+1));
		memset(temp_csr_row_ptr,0,sizeof(long)*(rows+1));
		
		// Read data
		FILE *fpx = fopen(fileAname, "r");
		// Set csr_row_ptr from non-symmetric edge list
		for(long i=0,r,c; i<nedges; i++){
			fscanf(fpx,"%ld %ld", &r, &c);
			csr_row_ptr[r]++;
			csr_row_ptr[c]++;
			temp_csr_row_ptr[r]++;
			temp_csr_row_ptr[c]++;
		}
		for(long r=1; r<=rows; ++r) csr_row_ptr[r] += csr_row_ptr[r-1];
		for(long r=1; r<=rows; ++r) temp_csr_row_ptr[r] += temp_csr_row_ptr[r-1];

		// Read again to set column indices and values
		rewind(fpx);
		long prev_r_pos, prev_c_pos;
		for(long i=0,r,c; i<nedges; i++){
			fscanf(fpx,"%ld %ld", &r, &c);
			prev_r_pos = temp_csr_row_ptr[r-1]++; 
			prev_c_pos = temp_csr_row_ptr[c-1]++;
			csr_col_idx[prev_r_pos] = c-1;
			csr_col_idx[prev_c_pos] = r-1;
			csr_val[prev_r_pos] = 1;
			csr_val[prev_c_pos] = 1;
		}
		fclose(fpx);

		// set original index
		original_index = new long[rows];
		for(long i=0; i<rows; i++) original_index[i] = i;
		
		printf("\nFile Loaded !");
		delete [] temp_csr_row_ptr;
	}
	//-----------------------------------------------------
	void submatrix(long *map_index, long num_nodes, smat_t *A_src)
	{
		rows = num_nodes;
		cols = rows;
		mem_alloc_by_me = true;
		csr_row_ptr = new long[rows+1];
		memset(csr_row_ptr,0,sizeof(long)*(rows+1));
		inv_map_index = new long[num_nodes];
		long *idx_flag = new long[A_src->rows];
		memset(idx_flag,0,sizeof(long)*A_src->rows);
		original_index = new long[num_nodes];
		// init lookup table
		for(long i=0; i<num_nodes; i++)
		{
			idx_flag[map_index[i]] = i+1;
			inv_map_index[i] = map_index[i];
			original_index[i] = A_src->original_index[map_index[i]];
		}
		long node, r_start, r_end, c_idx;
		nnz = 0;
		queue_t q_cidx, q_val;
		for(long i=0; i<num_nodes; i++)
		{
			node = map_index[i];
			r_start = A_src->csr_row_ptr[node];
			r_end = A_src->csr_row_ptr[node+1]-1;
			for(long j=r_start; j<=r_end; j++)
			{
				c_idx = A_src->csr_col_idx[j];
				if(idx_flag[c_idx]>0)
				{
					nnz++;
					csr_row_ptr[i+1]++;
					q_cidx.push(idx_flag[c_idx]-1);
					q_val.push(A_src->csr_val[j]);
				}

			}

		}
		for(long r=1; r<=rows; ++r) csr_row_ptr[r] += csr_row_ptr[r-1];

		nedges = nnz/2;
		csr_val = new double[nnz];
		csr_col_idx = new long[nnz];
		for(long i=0; i<nnz; i++)
		{
			csr_col_idx[i] = q_cidx.pop();
			csr_val[i] = q_val.pop();
		}
		q_cidx.clear();
		q_val.clear();
		delete [] idx_flag;
		return;
	}
	//-----------------------------------------------------
	void print_csr(void)
	{
		printf("\n\n>>Printing CSR matrix:\n");
		printf("\nNumber of nodes = %ld",rows);
		printf("\nNumber of edges = %ld",nedges);
		long r_start, r_end;
		for(long i=0; i<rows; i++)
		{
			//printf("\n%d",csr_row_ptr[i]);
			r_start = csr_row_ptr[i];
			r_end = csr_row_ptr[i+1]-1;
			for(long j=r_start; j<=r_end; j++)
				printf("\nrow=%ld, col=%ld",i+1,csr_col_idx[j]+1);
		}
		printf("\n\noriginal index");
		for(long i=0; i<rows; i++)
		{
			printf("%ld,",original_index[i]);
		}
		return;
	}
	//-----------------------------------------------------	
	long nnz_of_row(int i) const {return (csr_row_ptr[i+1]-csr_row_ptr[i]);}
	//-----------------------------------------------------
	//void free(void *ptr) {if(!ptr) ::free(ptr);}
	~smat_t(){
		if(mem_alloc_by_me) {
			if(csr_val) delete [] csr_val;
			if(csr_row_ptr) delete [] csr_row_ptr;
			if(csr_col_idx) delete [] csr_col_idx;
			if(inv_map_index) delete [] inv_map_index;
			if(original_index) delete [] original_index;
		}
	}
	//-----------------------------------------------------
	void clear_space() {
		if(csr_val) delete [] csr_val;
		if(csr_row_ptr) delete [] csr_row_ptr;
		if(csr_col_idx) delete [] csr_col_idx;
		if(inv_map_index) delete [] inv_map_index;
		if(original_index) delete [] original_index;
		mem_alloc_by_me = false;
		rows = 0; cols = 0; nnz=0; nedges = 0;

	}
	//-----------------------------------------------------
	/*
	double* multiply (double *x) {
		double *y = (double *) malloc(rows * sizeof(double));
		#pragma omp parallel for schedule(dynamic,32)
		for (long i=0; i < rows; i++) {
			double v=0.0;
			for (long j=row_ptr[i]; j < row_ptr[i+1]; j++) {
				v += val[j]*x[col_idx[j]];
			}
			y[i] = v;
		}
		return y;
	}*/
};

class comm_t{
public:
	long K;
	long K_max;
	queue_t *q;
	long max_size;

	comm_t(long kmax): K(0), K_max(kmax), q(new queue_t[kmax]), max_size(0) { for(long k=0; k<kmax; k++) q[k].init(); }
	comm_t(){}

	~comm_t() {
		delete [] q;
	}

	void clear(){
		delete [] q;
		K = 0;
		max_size = 0;
		K_max = 0;
		return;
	}

	void init(long kmax)
	{
		K_max = kmax;
		K = 0;
		max_size = 0;
		q = new queue_t[kmax];
		return;
	}

	void add(long *theta_arr, long *node_idx, long num_node, long num_comm){
		if(K+num_comm>K_max)
		{
			printf("\n***Error***>> Too many communities !");
			return;
		}
		long cidx;
		long *size = new long[num_comm];
		memset(size,0,sizeof(long)*num_comm);
		for(long i=0; i<num_node; i++)
		{
			cidx = theta_arr[i];
			if(cidx>0)
			{
				q[K+cidx-1].push(node_idx[i]);
				size[cidx-1]++;
			}
		}

		for(long k=0; k<num_comm; k++)
		{
			if(size[k]>max_size) max_size = size[k];
		}

		K += num_comm;
		delete [] size;
		return;
	}

};

void findConnectedComponents(comm_t &cc, smat_t &A)
{
	long n = A.rows;
	long *found = new long[n];
	memset(found,0,sizeof(long)*n);
	queue_t q;
	long r_start, r_end, cidx;
	long comp_id = 0;
	for(long i=0; i<n; i++)
	{
		if(found[i]>0)
		{
			continue;
		}
			
		comp_id++;
		found[i] = comp_id;
		q.push(i);
		long node;
		while(q.isempty()==false)
		{
			node = q.pop();
			r_start = A.csr_row_ptr[node];
			r_end = A.csr_row_ptr[node+1]-1;
			for(long j=r_start; j<=r_end; j++)
			{
				cidx = A.csr_col_idx[j];
				if(found[cidx]==0)
				{
					found[cidx] = comp_id;
					q.push(cidx);
				}
			}
		}

	}
	cc.add(found,A.original_index,n,comp_id);
	delete [] found;
	return;
}

bool exploreWhisker(queue_t &q_nodes_visited,long start_node, long start_neighbor, long max_size,smat_t *A){
	long n = A->rows;
	queue_t q;
	long *found = new long[n];
	memset(found,0,sizeof(long)*n);
	bool isSmall = true;
			
	found[start_node] = 1;
	found[start_neighbor] = 1;
	q.push(start_node);
	q_nodes_visited.clear();  // Clear visited queue
	q_nodes_visited.push(start_node);
	long node;
	while(q.isempty()==false){
		node = q.pop();
		long r_start = A->csr_row_ptr[node];
		long r_end = A->csr_row_ptr[node+1]-1;
		for(long j=r_start; j<=r_end; j++)
		{
			long cidx = A->csr_col_idx[j];
			if(found[cidx]==0)
			{
				found[cidx] = 1;
				q.push(cidx);
				q_nodes_visited.push(cidx);
			}
		}

		if(q_nodes_visited.length>=max_size){
			isSmall = false;
			break;
		}
		
	}
	delete [] found;
	return(isSmall);
}

void countEdgesToCluster(long *deg, long *cluster, long size, smat_t *A)
{
	long n = A->rows;
	memset(deg,0,sizeof(long)*n);
	long r_start, r_end, cidx;
	long *skip = new long[n];
	memset(skip,0,sizeof(long)*n);
	for(long i=0; i<size; i++)
	{
		skip[cluster[i]] = 1;
	}
	long edge_count;
	for(long i=0; i<n; i++)
	{
		if(skip[i]==0)
		{
			r_start = A->csr_row_ptr[i];
			r_end = A->csr_row_ptr[i+1]-1;
			edge_count = 0;
			for(long j=r_start; j<=r_end; j++)
			{
				cidx = A->csr_col_idx[j];
				for(long k=0; k<size; k++)
				{
					if(cluster[k]==cidx) edge_count++;
				}
			}
			deg[i] = edge_count;
		}
	}
	delete [] skip;
	return;
}

void countEdgesToClusterPl(long *deg, long *cluster, long size, smat_t *A, int numThreads)
{
	long n = A->rows;
	memset(deg,0,sizeof(long)*n);
	
	long *skip = new long[n];
	memset(skip,0,sizeof(long)*n);
	for(long i=0; i<size; i++)
	{
		skip[cluster[i]] = 1;
	}
	
	omp_set_num_threads(numThreads);
	#pragma omp parallel for schedule(dynamic,32)
	for(long i=0; i<n; i++)
	{
		if(skip[i]==0)
		{
			long r_start = A->csr_row_ptr[i];
			long r_end = A->csr_row_ptr[i+1]-1;
			long edge_count = 0;
			for(long j=r_start; j<=r_end; j++)
			{
				long cidx = A->csr_col_idx[j];
				for(long k=0; k<size; k++)
				{
					if(cluster[k]==cidx) edge_count++;
				}
			}
			deg[i] = edge_count;
		}
	}
	delete [] skip;
	return;
}

class degree_t{
public:
	long *deg;
	bool f_mem_alloc;
	long max_degree;
	long *interimMax;
	degree_t(){f_mem_alloc = false; max_degree=0;}

	void calcDegree(smat_t *A)
	{
		long n = A->rows;
		deg = new long[n];
		f_mem_alloc = true;
		printf("\nComputing degree ...");
		//long rowFirst, rowLast;
		for(long i=0; i<n; i++)
		{
			deg[i] = 0;
			long rowFirst = A->csr_row_ptr[i];
			long rowLast = A->csr_row_ptr[i+1]-1;
			deg[i] = rowLast - rowFirst + 1;
		}
		return;
	}

	void calcDegreePl(smat_t *A, int numThreads)
	{
		long n = A->rows;
		deg = new long[n];
		f_mem_alloc = true;
		printf("\nComputing degree parallel ...");
		
		omp_set_num_threads(numThreads);
		#pragma omp parallel for schedule(dynamic,32)
		for(long i=0; i<n; i++)
		{
			deg[i] = 0;
			long rowFirst = A->csr_row_ptr[i];
			long rowLast = A->csr_row_ptr[i+1]-1;
			deg[i] = rowLast - rowFirst + 1;
		}
		printf("\nDegree computed !");
		return;
	}


	long calcMaxDegree(smat_t *A)
	{
		long n = A->rows;
		max_degree = 0;
		for(long i=0; i<n; i++)
		{
			if(deg[i]>max_degree)
				max_degree = deg[i];
		}
		return(max_degree);
	}

	long calcMaxDegreePl(smat_t *A)
	{
		long n = A->rows;
		if(n<800)
			return(calcMaxDegree(A));

		printf("\nComputing max degree parallel ...");
		max_degree = 0;
		interimMax = new long[200];
		memset(interimMax,0,200*sizeof(long));
		long step = n/200;
		#pragma omp parallel for schedule(dynamic,32)
		for(long i=0;i<200;i++){
			long start = i*step;
			long end= start+step-1;
			if(end>n-1) end = n-1;
			for(long j=start; j<=end; j++){
				if(deg[j]>interimMax[i])
					interimMax[i] = deg[j];
			}
		}

		// Find global max
		for(long i=0; i<200; i++){
			if(interimMax[i]>max_degree)
				max_degree = interimMax[i];
		}

		delete [] interimMax;
		return(max_degree);
	}

	long threshold(bool *res, long th, long n)
	{
		long num_pure_nodes = 0;
		for(long i=0; i<n; i++)
		{
			if(deg[i]<=th)
			{
				res[i] = true;
				num_pure_nodes++;
			}
			else
			{
				res[i] = false;
			}
		}
		return(num_pure_nodes);
	}

	void clear()
	{
		if(deg) delete [] deg;
		max_degree = 0;
		f_mem_alloc = false;
	}

	~degree_t()
	{
		if(f_mem_alloc){
		       	if(deg) delete [] deg;

		}
	}

};

long size_vdash(long *vdash, long n)
{
	long size = 0;
	for(long i=0; i<n; i++) size += vdash[i];
	return(size);
}


#endif /*overlap_lib.h*/
