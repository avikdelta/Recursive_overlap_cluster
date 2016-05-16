/* **************************************************************************
 * This file has eigen based sparse routines using sampling and 
 * randomized svd required by the overlap cluster algorithm
 *
 * Code by Avik Ray (avik@utexas.edu)
 *
 **************************************************************************** */

#ifndef SPARSE_EIGEN_LIB_OMP_H
#define SPARSE_EIGEN_LIB_OMP_H
//#define DEBUG

#include <cstdio>
#include <cstdlib>
#include "overlap_lib_omp.h"
#include "RedSVD.h"
#include "eigen_lib_omp.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include <Eigen/SVD>
#include <cmath>
#include <omp.h>
#include "sample_lib.h"

#define MAX_SPARSE_ALM_ITER 150
#define MAX_SPARSE_ALM_TIME_S 15 
#define MAX_CLUSTER_TIME 180

//typedef Eigen::SparseMatrix<double> esmat_t;
//typedef Eigen::Triplet<double> etriplet_t;
//typedef Eigen::MatrixXd emat_t;
//typedef Eigen::VectorXd evec_t;
typedef RedSVD::RedSVD<Eigen::SparseMatrix<double> > redsvds_t;
typedef RedSVD::RedSVD<Eigen::MatrixXd> redsvd_t;

void smat2esmat(esmat_t &D, smat_t &S)
{
	long r_start, r_end, c_idx;
	std::vector<etriplet_t> triplet_list;
	triplet_list.resize(S.nnz);
	for(long i=0; i<S.rows; i++)
	{
		r_start = S.csr_row_ptr[i];
		r_end = S.csr_row_ptr[i+1]-1;
		for(long j=r_start; j<=r_end;j++)
		{
			c_idx = S.csr_col_idx[j];
			triplet_list[j] = etriplet_t(i,c_idx,S.csr_val[c_idx]);
		}
	}
	D.resize(S.rows,S.cols);
	D.setFromTriplets(triplet_list.begin(),triplet_list.end());
	return;
}


double NuclearNormMatSparse(esmat_t &X, double rank)
{
	redsvds_t rsvd(X,rank);
	evec_t Sigma = rsvd.singularValues();
	double norm = 0;
	for(long i=0; i<rank; i++)
	{
		norm += Sigma(i);
	}
	return(norm);
}

void ALMEigSparseJacobi(emat_t &Y, emat_t &S, emat_t &A, emat_t &C, float rho, long rank)
{
	// Initializations
	long n = A.rows();
	emat_t M = Eigen::MatrixXd::Zero(n,n);
	emat_t argmat1, argmat2;
	Y = Eigen::MatrixXd::Zero(n,n);
	S = Eigen::MatrixXd::Zero(n,n);
	emat_t Ybar;
	evec_t Sigma;
	emat_t SigmaDash;
	
	float delta = 0.01;
	float change = 100;
	int iter_count = 0;
	int max_iter = MAX_SPARSE_ALM_ITER;
	float mu0 = 1;
	float mu = mu0;
	double iter_start_time, iter_time;
	
	//Eigen::JacobiSVD<Eigen::MatrixXd> svd(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
	
	redsvd_t rsvd(Y,rank);
	Sigma = rsvd.singularValues();
	argmat1 = A-Y;
	double objOld = Sigma.sum() +  L1NormMatDense(C, argmat1);
	double objNew;

	// Main Loop
	iter_time = 0;
	while(change>delta)
	{
		iter_start_time = omp_get_wtime();	
		#ifdef DEBUG
		if(iter_count%10==0) printf("\nIterations %d to %d",iter_count+1,iter_count+10);
		#endif
		if(iter_count>max_iter){
			printf("\n****WARNING****>> SPARSE ALM not converging ... Terminating");
			break;
		}
		if(iter_count*iter_time>MAX_SPARSE_ALM_TIME_S){
			printf("\n****WARNING****>> SPARSE ALM TIMEOUT ... Terminating");
			break;
		}
		
		//svd.compute(A-S+(1/mu)*M,Eigen::ComputeThinU | Eigen::ComputeThinV);
		rsvd.compute(A-S+(1/mu)*M,rank);
		Sigma = rsvd.singularValues();
		SigmaDash = Eigen::MatrixXd::Zero(rank,rank);
		for(long i=0; i<rank; i++)
		{
			if(Sigma(i)>1/mu)
			{
				SigmaDash(i,i) = Sigma(i) - 1/mu;
			}
			else if(Sigma(i)<(-1)/mu)
			{
				SigmaDash(i,i) = Sigma(i) + 1/mu;
			}
			else
			{
				SigmaDash(i,i) = 0;
			}
		}
		//Ybar = svd.matrixU()*SigmaDash*svd.matrixV().transpose();
		Ybar = rsvd.matrixU()*SigmaDash*rsvd.matrixV().transpose();

		for(long i=0; i<n; i++)
		{
			for(long j=0; j<=i; j++)
			{
				Y(i,j) = max(min(Ybar(i,j),1.0),0.0);
            			Y(j,i) = Y(i,j);
			}
		}

		argmat1 = A-Y+(1/mu)*M;
		argmat2 = pow(1/mu,2)*C;
		softThDense(S, argmat1, argmat2, true);
		M = M + mu*(A-Y-S);
		mu = rho*mu;

		iter_count++;

		// Compute change in objective
		//svd.compute(Y,Eigen::ComputeThinU | Eigen::ComputeThinV);
		rsvd.compute(Y,rank);
		Sigma = rsvd.singularValues();
		argmat1 = A-Y;
		objNew = Sigma.sum() +  L1NormMatDense(C, argmat1);
		change = abs(objOld-objNew);
		objOld = objNew;
		iter_time = omp_get_wtime() - iter_start_time;

	}
	return;
}

long clusterCPSparse(long *theta, emat_t &A, double p, double q, long gamma, int max_iter, long rank)
{
	// Initializations
	long n = A.rows();
	double t = p/4+3*q/4;
	double lu = gamma/2;
	double factor = 1.1;
	int iter_count = 1;
	double kappa;
	float rho;
	emat_t C(n,n);
        emat_t Y(n,n);
	emat_t Y1(n,n);
	emat_t S(n,n);
	evec_t SingVal;
	double sigma, c1, c2;

	Y1 = Eigen::MatrixXd::Zero(n,n);
	//Eigen::JacobiSVD<Eigen::MatrixXd> svd(Y1, Eigen::ComputeThinU | Eigen::ComputeThinV);
	redsvd_t rsvd(Y1,rank);

	// Main loop
	double cluster_start_time, cluster_total_time;
	cluster_start_time = omp_get_wtime();
	while(1)
	{
		#ifdef DEBUG
		printf("\npartial clustering iteration %d",iter_count);
		#endif

		kappa = lu*(p-q)/(sqrt(p*(1-q)*n)*pow(log(n),2));

		if(n>1)
		{
			#ifdef DEBUG
			printf("\nkappa=%lf, p=%lf, q=%lf",kappa,p,q);
			#endif
        		c1 = sqrt((1-t)/(t*n*log(n)))/kappa;
        		c2 = sqrt(t/((1-t)*n*log(n)))/kappa;
			
		}
		else
		{
			c1 = n;
			c2 = n;
		}


		genCMatDense(C, A, c1, c2, p, q);
		rho = 1.5;
		ALMEigSparseJacobi(Y, S, A, C, rho, rank);

		makeClusterDense(Y1, Y, .45);

		//svd.compute(Y1,Eigen::ComputeThinU | Eigen::ComputeThinV);
		rsvd.compute(Y1,rank);
		SingVal = rsvd.singularValues();
		sigma = calcSigma(SingVal,rank);

		cluster_total_time = omp_get_wtime() - cluster_start_time;
		if(sigma>lu)
		{
			break;
		}
		else if(iter_count>max_iter)
		{
			#ifdef DEBUG
			printf("\nmaximum iterations performed ... quiting");
			#endif
			break;
		}
		else if(cluster_total_time>MAX_CLUSTER_TIME)
		{
			printf("\nMAX CLUSTER TIME EXCEEDED ! QUITING.");
			break;
		}

		lu = lu/factor;
		iter_count++;

	}

	#ifdef DEBUG
	printf("\nAssigning communities ... ");
	#endif
	float threshold = .45;
	double *theta_sum = new double[n]; 
	for(long i=0; i<n; i++)
	{
		theta[i] = 0;
		theta_sum[i] = 0.0;
		for(long j=0; j<n; j++)
		{
			theta_sum[i] += Y(i,j);
		}
	}

	long cidx = 1;
	double inner;
	for(long i=0; i<n; i++)
	{
		if((theta[i]>0)||(theta_sum[i]==0))
			continue;

		theta[i] = cidx;
		for(long j=i+1; j<n; j++)
		{
			if((theta[j]>0)||(theta_sum[j]==0))
				continue;
			
			inner = 0;
			for(long k=0; k<n; k++) inner += Y(i,k)*Y(j,k);
			inner = inner/(rowNormDense(Y,i)*rowNormDense(Y,j));
			if(inner>=threshold){
				theta[j] = cidx;
			}
		}
		cidx++;		
	}

	#ifdef DEBUG
	printf("\nNumber of clusters = %ld",cidx-1);
	#endif
	delete [] theta_sum;
	return(cidx-1);
}

double clusterCPLarge(long *theta, smat_t &A, double p, double q, long gamma, int max_iter, long rank, long sample_size)
{
	long n = A.rows;
	std::vector< long > samples(sample_size);
	SampleWithoutReplacement(n,sample_size,samples);
	long *in_sample = new long[n];
	memset(in_sample,0,sizeof(long)*n);
	long *buff = new long[sample_size];
	for(long i=0; i<sample_size; i++)
	{
		in_sample[samples[i]] = 1;
		buff[i] = samples[i];
	}
	smat_t AS;
	AS.submatrix(buff,sample_size,&A);
	delete [] buff;	
	emat_t denseAS(sample_size,sample_size);
	smat2dmat(denseAS, AS);
	long *theta_samp = new long[sample_size];
	long K = clusterCPSparse(theta_samp, denseAS, p, q, gamma, max_iter, rank);

	// Set communities of samples
	memset(theta,0,sizeof(long)*n);
	for(long i=0; i<sample_size; i++)
	{
		theta[samples[i]] = theta_samp[i];
	}
	
	// Set communities of remaining nodes
	for(long i=0; i<n; i++)
	{
		if(in_sample[i]==0)
		{
			long r_start = A.csr_row_ptr[i];
			long r_end = A.csr_row_ptr[i+1]-1;
			long arg_max_k = 0;
			long max_deg = 0;
			for(long k=0; k<K; k++)
			{
				long deg_to_comm = 0;
				for(long cnidx=0; cnidx<sample_size; cnidx++)
				{
					// Find if node is in this community
					if(theta_samp[cnidx]==k+1)
					{
						// Find if edge exists
						for(long j=r_start; j<=r_end; j++)
						{
							if(samples[cnidx]==A.csr_col_idx[j]) deg_to_comm++;

						}
					}
				}
				if(deg_to_comm>max_deg)
				{
					max_deg = deg_to_comm;
					arg_max_k = k+1;
				}

			}
			theta[i] = arg_max_k;
		}
	}
	delete [] theta_samp;
	delete [] in_sample;	
	return(K);
}	
#endif /*sparse_eigen_lib.h*/
