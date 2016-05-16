/*
 * Main file for Recursive Overlap Cluster algorithm
 *
 * Code by Avik Ray (avik@utexas.edu)
 *
 * Copyright (C) 2016 Avik Ray

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 *
 * */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <time.h>
#include <omp.h>
#include "overlap_lib_omp.h"
#include "eigen_lib_omp.h"
#include "RedSVD.h"
#include "sparse_eigen_lib_omp.h"
#include <vector>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#define SAMPLE_SIZE 1000
#define LARGE_CLUSTER_TH 1000

//#define DEBUG

typedef std::vector<comm_t*> comm_vec_t;


void findCommunities(config_t *config, smat_t *A)
{
	//Time keeping
	double start_time, total_time, time_elapsed;
	start_time = omp_get_wtime();

	// Result and log filename
	char resFilename[100] = "C_";
	strcat(resFilename,config->testID);
	strcat(resFilename,"_");
	strcat(resFilename,config->edgeListFile);
	
	char logFilename[100] = "log_";
	strcat(logFilename,config->testID);
	strcat(logFilename,"_");
	strcat(logFilename,config->edgeListFile);
	
	char clogFilename[100] = "CLOG_";
	strcat(clogFilename,config->testID);
	strcat(clogFilename,"_");
	strcat(clogFilename,config->edgeListFile);

	// Init log file
	FILE *fpLog = fopen(logFilename,"w");
	fclose(fpLog);

	// Init Clog file
	FILE *fcLog = fopen(clogFilename,"w");
	fclose(fcLog);

	// Initialize
	int numThreads = config->num_threads;
	double p, q;
	p = config->p;
	q = config->q;
	long gamma = config->gamma;
	long n = A->rows;
	config->K_max = n;         // Initialize maximum number of communities
	long *vdash = new long[n];
	for(long i=0; i<n; i++) vdash[i] = 1;

	smat_t A_pure;
	degree_t D_curr;
	D_curr.calcDegreePl(A,numThreads); //D_curr.calcDegree(&A_curr);
	//D_curr.calcMaxDegree(A);
	D_curr.calcMaxDegreePl(A);

	// Remove degree 0 nodes and whiskers and recompute degree

	// Remove whiskers
	printf("\nRemoving whiskers ...");
	bool fWhisker = false;
	
	////A->print_csr(); //debug
	for(long i=0; i<n; i++){
		if(vdash[i]==0) continue;

		long r_start = A->csr_row_ptr[i];
		long r_end = A->csr_row_ptr[i+1]-1;
		for(long j=r_start; j<=r_end; j++){
			long cidx = A->csr_col_idx[j];

			// Compute connected component starting from i
			queue_t q_nodes_visited;
			bool is_whisker;
			long visited_node;
			is_whisker = exploreWhisker(q_nodes_visited,i,cidx,gamma,A);
			if(is_whisker==true){
				fWhisker = true;
				while(q_nodes_visited.length>0){
					visited_node = q_nodes_visited.pop();
					////printf("\nWhisker node = %ld",visited_node+1); // debug
					vdash[visited_node] = 0;
				}

			}

			// Compute connected component starting from cidx
			is_whisker = exploreWhisker(q_nodes_visited,cidx,i,gamma,A);
			if(is_whisker==true){
				fWhisker = true;
				while(q_nodes_visited.length>0){
					visited_node = q_nodes_visited.pop();
					////printf("\nWhisker node = %ld",visited_node+1); // debug
					vdash[visited_node] = 0;
				}

			}

		}
	}
	if(fWhisker==true) printf("\nWhiskers removed !");

	// Remove zero degree nodes
	printf("\nRemoving zero degree nodes ...");
	long *nz_deg_node = new long[n];
	long nz_count = 0;
	long num_zero_degree = 0;
	for(long i=0; i<n; i++)
	{
		// Consider valid node if degree>0 and its not in a whisker
		if((D_curr.deg[i]>0)&&(vdash[i]==1))  
		{
			nz_deg_node[nz_count] = i;
			nz_count++;
		}
		else
		{
			if(D_curr.deg[i]==0) num_zero_degree++;
			vdash[i] = 0;
		}
	}
	printf("\nNumber of nodes removed = %ld",n-nz_count);
	printf("\nNumber of whisker nodes removed = %ld",n-nz_count-num_zero_degree);
	printf("\nNumber of zero degree nodes = %ld",num_zero_degree);
	if(config->f_log == true)
	{
		fpLog = fopen(logFilename,"a");
		fprintf(fpLog,"Number of whisker nodes removed = %ld\n",n-nz_count-num_zero_degree);
		fprintf(fpLog,"Number of zero degree nodes = %ld\n",num_zero_degree);
		fprintf(fpLog,"Remaining nodes = %ld\n",nz_count);
		fclose(fpLog);
	}
	
	smat_t A_curr;
	A_curr.submatrix(nz_deg_node,nz_count,A);
	D_curr.clear();
	D_curr.calcDegreePl(&A_curr,numThreads); //D_curr.calcDegree(&A_curr);
	//D_curr.calcMaxDegree(&A_curr);
	D_curr.calcMaxDegreePl(&A_curr);
	delete [] nz_deg_node;

	long dense_threshold = config->dense_threshold;
	bool fverbose = config->f_verbose;
	long max_iter = config->max_cluster_iter;
	comm_t C(config->K_max);
	comm_t Cpure(config->K_max);


	// Main loop
	long th = config->threshold_init;
	long rank = config->sparse_svd_rank;
	long np;
	long iter_count = 0;
	long prev_num_pure_nodes = 0;
	while((A_curr.rows>0)&&(th<A_curr.rows))
	{
		iter_count++;
		// Increment degree threshold
		th++;

		if(th>D_curr.max_degree)
		{
			if(fverbose){
				printf("\n****WARNING**** >> MAX DEGREE EXCEEDED !!");
				printf("\nNumber of leftover nodes = %ld",A_curr.rows);
			}
			break;
		}

		if(fverbose){
		       printf("\n== Edge Threshold = %ld ========, max degree = %ld",th,D_curr.max_degree);
		       printf("\nRemaining nodes=%ld, K max=%ld, max size=%ld",A_curr.rows,C.K,C.max_size);
		}

		// degree thresholding
		bool *res_threshold = new bool[A_curr.rows];
		long num_pure_nodes = D_curr.threshold(res_threshold, th, A_curr.rows);
		
		/*
		if(num_pure_nodes==0)
		{
			continue;
		}*/
		if(num_pure_nodes<=prev_num_pure_nodes){
			continue;
		}
		else{
			prev_num_pure_nodes = num_pure_nodes;
			printf("\nNumber of pure nodes=%ld",num_pure_nodes);
		}

		long *buff = new long[num_pure_nodes];
		long count = 0;
		for(long i=0; i<A_curr.rows; i++)
		{
			if(res_threshold[i]==true)
			{
				buff[count] = i;
				count++;
			}
			
		}
		delete [] res_threshold;

		// Form pure node submatrix
		A_pure.clear_space();
		A_pure.submatrix(buff,num_pure_nodes,&A_curr);
		delete [] buff;

		// Find connected components
		np = A_pure.rows;
		comm_t cc(np);
		printf("\nFinding connected components ... ");
		findConnectedComponents(cc, A_pure);
		printf(" %ld components found.",cc.K);

		// Pure node clustering in each connected component
		comm_vec_t Cpure_cc;
		for(long k=0; k<cc.K; k++)
		{
			long c_size = cc.q[k].length;
			Cpure_cc.push_back(new comm_t(c_size));
		}

		Cpure.clear();
		Cpure.init(config->K_max);

		omp_set_num_threads(numThreads);
		#pragma omp parallel for schedule(dynamic,32)
		for(long k=0; k<cc.K; k++)
		{
			queue_t *t = &cc.q[k];
			long c_size = t->length;
			long *theta = new long[c_size];
			long num_comm;
			if(c_size<=3)
			{
				long *nidx = new long[c_size];
				for(long j=0; j<c_size; j++)
				{
					theta[j] = 1;
					nidx[j] = t->pop();
				}
				//theta[0] = 1;
				num_comm = 1;
				//long nidx = t->pop();
				Cpure_cc[k]->add(theta,nidx,c_size,1);
				delete [] nidx;
			}
			else
			{
				smat_t A_pure_cc;
				long *c_node_idx = new long[c_size];
				//printf("\n");
				for(long j=0; j<c_size; j++)
				{
					c_node_idx[j] = t->pop(); // these have original indices
					//printf("%ld ",c_node_idx[j]);
				}
				A_pure_cc.clear_space();
				A_pure_cc.submatrix(c_node_idx,c_size,A);
				delete [] c_node_idx;
			
				// Cluster	
				if(c_size<=dense_threshold)
				{
					emat_t denseA(c_size,c_size);
					smat2dmat(denseA, A_pure_cc);			
					num_comm = clusterCPDense(theta, denseA, p, q, gamma, max_iter);	
				}
				else
				{
					printf("\npure nodes = %ld, invoke sparse cluster...",c_size);
					long rank_sm = rank;
					//if(c_size >= 15000)
					if(c_size >= LARGE_CLUSTER_TH)
					{
						rank_sm = 75;
						printf(" sampling... rank %ld",rank_sm);
						long sample_size = SAMPLE_SIZE;
						num_comm = clusterCPLarge(theta, A_pure_cc, p, q, gamma, max_iter, rank_sm, sample_size);				
					}
				/*	else if((c_size<15000)&&(c_size>=10000))
					{
						rank_sm = 5;
					}
					else if((c_size<10000)&&(c_size>=5000))
					{
						rank_sm = 10;
					}
					else if((c_size<5000)&&(c_size>=2000))
					{
						rank_sm = 30;
					} */
					else if((c_size<LARGE_CLUSTER_TH)&&(c_size>=750))
					{
						emat_t denseA(c_size,c_size);
						smat2dmat(denseA, A_pure_cc);
						rank_sm = 75;
						printf(" rank %ld",rank_sm);
						num_comm = clusterCPSparse(theta, denseA, p, q, gamma, max_iter, rank_sm);
					}
					else if((c_size<750)&&(c_size>=500))
					{
						emat_t denseA(c_size,c_size);
						smat2dmat(denseA, A_pure_cc);
						rank_sm = 125;
						printf(" rank %ld",rank_sm);
						num_comm = clusterCPSparse(theta, denseA, p, q, gamma, max_iter, rank_sm);
					}
					else
					{
						emat_t denseA(c_size,c_size);
						smat2dmat(denseA, A_pure_cc);
						printf(" rank %ld",rank_sm);
						num_comm = clusterCPSparse(theta, denseA, p, q, gamma, max_iter, rank_sm);
					}
						
				}
				Cpure_cc[k]->add(theta,A_pure_cc.original_index,c_size,num_comm);
			}
			
			delete [] theta;

		}

		// Combine communities found from separate connected components
		for(long k=0; k<cc.K; k++)
		{
			long num_com_cc = Cpure_cc[k]->K;
			for(long i=0; i<num_com_cc; i++)
			{
				long csize = Cpure_cc[k]->q[i].length;
				// Only add large communities
				if(csize>=gamma)
				{
					long *theta = new long[csize];
					memset(theta,0,sizeof(long)*csize);
					long *org_idx = new long[csize];
					for(long j=0; j<csize; j++)
					{
						theta[j] = 1;
						org_idx[j] = Cpure_cc[k]->q[i].pop();
					}
					Cpure.add(theta,org_idx,csize,1);
					delete [] theta;
					delete [] org_idx;
				}
				
			}
			delete Cpure_cc[k];
		}

		// End of pure node clustering
		

		// Process clusters
		if(Cpure.max_size>=gamma)
		{
			fcLog = fopen(clogFilename,"a"); // open comm log file
	
			printf("\n%ld pure node cluster found !",Cpure.K);
			queue_t *temp;
			long *comm_add = new long[n];
			long *deg_to_cluster = new long[n];
			long Knew = 0;
			for(long k=0; k<Cpure.K; k++)
			{
				temp = &Cpure.q[k];
				long size = temp->length;
				//printf("\nsize=%ld",size);
				if(size>=gamma)
				{
					//printf("\nLarge cluster! Size = %ld",size);
					Knew++;
					long node;
					long *comm_nodes = new long[size];
					memset(comm_add,0,sizeof(long)*n);
					for(long i=0; i<size; i++)
					{
						node = temp->pop();
						comm_nodes[i] = node;
						comm_add[node] = 1;
						fprintf(fcLog,"%ld ",node+1);
						vdash[node] = 0;
					}

					// Find whole community
					//countEdgesToCluster(deg_to_cluster, comm_nodes, size, A);
					countEdgesToClusterPl(deg_to_cluster, comm_nodes, size, A, numThreads);
					
					double threshold = size*(p+q)/2;

					omp_set_num_threads(numThreads);
					#pragma omp parallel for schedule(dynamic,32)
					for(long i=0; i<n; i++)
					{
						if(deg_to_cluster[i]>=threshold)
						{
							comm_add[i] = 1;
							vdash[i] = 0;
							fprintf(fcLog,"%ld ",i+1); // Input nodes numbered from 1
						}
					}
					// Add community
					C.add(comm_add, A->original_index, n, 1);
					delete [] comm_nodes;
					fprintf(fcLog,"\n");

				}

			}
			fclose(fcLog);
			delete [] comm_add;
			delete [] deg_to_cluster;

			// Re-form current submatrix
			long num_nodes = size_vdash(vdash, n);
			time_elapsed = omp_get_wtime() - start_time;
			printf("\nNew number of nodes = %ld, time = %lf s",num_nodes,time_elapsed);
			if(config->f_log == true)
			{
				fpLog = fopen(logFilename,"a");
				fprintf(fpLog,"%ld %lf %ld\n",num_nodes,time_elapsed,C.K);
				fclose(fpLog);
			}
			long *buff = new long[num_nodes];
			long count = 0;
			for(long i=0; i<n; i++)
			{
				if(vdash[i]==1)
				{
					buff[count] = i;
					count++;
				}
			}
			A_curr.clear_space();
			A_curr.submatrix(buff,num_nodes,A);
			delete [] buff;

			// Re-calculate degree
			D_curr.clear();
			if(num_nodes<1000)
			{	
				D_curr.calcDegree(&A_curr);
			}
			else
			{
				D_curr.calcDegreePl(&A_curr,numThreads);
			}
			//D_curr.calcMaxDegree(&A_curr);
			D_curr.calcMaxDegreePl(&A_curr);
			printf("\nNew max degree = %ld",D_curr.max_degree);

			// Reset degree threshold
			th =  config->threshold_init;
			// Reset num pure nodes
			prev_num_pure_nodes = 0;
			
		} // End of process clusters


	}
	delete [] vdash;

	total_time = omp_get_wtime() - start_time;

	if(A_curr.rows==0)
	{
		printf("\nAll communities detected ! No leftover nodes !\n");
	}

	
	printf("\n\nTotal number of communities recovered = %ld",C.K);
	printf("\nRuntime = %lf s\n",total_time);

	// Save results
	fpLog = fopen(logFilename,"a");
	fprintf(fpLog,"%ld %lf %ld\n",A_curr.rows,total_time,C.K);
	fclose(fpLog);
	printf("\nWriting community file ...");
	FILE *fpRes = fopen(resFilename,"w");
	queue_t *temp;
	for(long k=0; k<C.K; k++)
	{
		temp = &C.q[k];
		long size = temp->length;
		for(long j=0; j<size; j++)
		{
			long node = temp->pop();
			//fprintf(fpRes,"%ld ",node);
			fprintf(fpRes,"%ld ",node+1);
		}
		fprintf(fpRes,"\n");
	}
	fclose(fpRes);
	printf("\nResult file written.\n");
	
	return;
}


int main(int argc, char *argv[])
{
	char configFileName[100];
	clock_t begin, end;
	double time_spent;
	
	/*-----------------------*/
	printf("\nHi! Starting Community Detection ...");
	strcpy(configFileName,argv[1]);
	
	/*----- Read Config ------*/
	config_t config;
	printf("\nReading config file.");
	config.load(configFileName);
	config.num_threads = atoi(argv[2]);
	config.print();

	/*----- Load Graph -----*/ 
	printf("\nGraph filename is %s",config.edgeListFile);
	smat_t dataMatA;
	dataMatA.load(config);
	
	// Find communities
	findCommunities(&config, &dataMatA);

}







