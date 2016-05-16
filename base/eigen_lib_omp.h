/* **************************************************************************
 * This file has eigen based routines used by the overlap cluster algorithm
 *
 * Code by Avik Ray (avik@utexas.edu)
 *
 **************************************************************************** */


#ifndef EIGEN_LIB_OMP_H
#define EIGEN_LIB_OMP_H
//#define DEBUG

#include <cstdio>
#include <cstdlib>
#include "overlap_lib_omp.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include <Eigen/SVD>
#include <cmath>

#define MAX_ALM_ITER 150

typedef Eigen::SparseMatrix<double> esmat_t;
typedef Eigen::Triplet<double> etriplet_t;
typedef Eigen::MatrixXd emat_t;
typedef Eigen::VectorXd evec_t;


void smat2dmat(emat_t &D, smat_t &S)
{
	long n = S.rows;
	long r_start, r_end, cidx;
	D = Eigen::MatrixXd::Zero(n,n);
	for(long i=0; i<n; i++)
	{
		r_start = S.csr_row_ptr[i];
		r_end = S.csr_row_ptr[i+1]-1;
		for(long j=r_start; j<=r_end; j++)
		{
			cidx = S.csr_col_idx[j];
			D(i,cidx) = S.csr_val[j];
		}
	}
	return;
}


void smat2emat(smat_t *S, emat_t &D)
{
	long r_start, r_end, c_idx;
	for(long i=0; i<S->rows; i++)
	{
		r_start = S->csr_row_ptr[i];
		r_end = S->csr_row_ptr[i+1]-1;
		for(long j=r_start; j<=r_end;j++)
		{
			c_idx = S->csr_col_idx[j];
			D(i,c_idx) = 1;
		}
	}
	return;
}


void EigDenseJacobiSVD(emat_t &S, emat_t &U, emat_t &V, evec_t &Sigma)
{
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
	U = svd.matrixU();
	V = svd.matrixV();
	Sigma = svd.singularValues();
	return;
}


double NuclearNormMatJacobi(emat_t &X)
{
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
	evec_t Sigma = svd.singularValues();
	double norm = 0;
	for(long i=0; i<X.rows(); i++)
	{
		norm += Sigma(i);
	}
	return(norm);
}

double L1NormMatDense(emat_t &C, emat_t &X)
{
	double norm = 0;
	for(long i=0; i<X.rows(); i++)
	{
		for(long j=0; j<X.cols(); j++)
		{
			norm += abs(C(i,j)*X(i,j));
		}
	}
	return(norm);
}

void softThDense(emat_t &R, emat_t &X, emat_t &Y,bool sym)
{
	long n = X.rows();
	long m = X.cols();
	if(sym==false){

		for(long i=0; i<n; i++)
		{
			for(long j=0; j<m; j++)
			{
				if(X(i,j)>Y(i,j))
				{
					R(i,j) = X(i,j)-Y(i,j);
				}
				else if(X(i,j)<Y(i,j))
				{
					R(i,j) = X(i,j)+Y(i,j);
				}
				else
				{
					R(i,j) = 0;
				}
			}
		}
	}
	else
	{
		for(long i=0; i<n; i++)
		{
			for(long j=0; j<=i; j++)
			{
				if(X(i,j)>Y(i,j))
				{
					R(i,j) = X(i,j)-Y(i,j);
				}
				else if(X(i,j)<Y(i,j))
				{
					R(i,j) = X(i,j)+Y(i,j);
				}
				else
				{
					R(i,j) = 0;
				}

				R(j,i) = R(i,j);
			}
		}
	}
	return;
}

double rowNormDense(emat_t &X, long row_idx)
{
	double norm_sqr = 0;
	long n = X.rows();
	for(long i=0; i<n; i++)
	{
		norm_sqr += pow(X(row_idx,i),2);
	}
	return(sqrt(norm_sqr));
}

void makeClusterDense(emat_t &Y1, emat_t &Y, double threshold)
{
	long n = Y.rows();
	Y1 = Eigen::MatrixXd::Zero(n,n);
	double val;

	for(long i=0; i<n; i++)
	{
		for(long j=0; j<=i; j++)
		{
			for(long k=0; k<n; k++) val += Y(i,k)*Y(j,k);
			val = val/(rowNormDense(Y,i)*rowNormDense(Y,j));

			if(val>=threshold)
			{
				Y1(i,j) = 1;
			}
			else
			{
				Y1(i,j) = 0;
			}
			Y1(j,i) = Y1(i,j);

		}
	}
	return;
}

double calcSigma(evec_t &SingVal, long n)
{
	evec_t diff = Eigen::VectorXd::Zero(n);
	double max_diff = 0;
	for(long i=0; i<n-1; i++)
	{
		diff(i) = SingVal(i) - SingVal(i+1);
		if(diff(i)>max_diff) max_diff = diff(i);
	}
	return(max_diff);
}
	
void genCMatDense(emat_t &C, emat_t &A, double c1, double c2, double p, double q)
{
	long n = A.rows();
	if(c1+c2==0.0)
	{
		c1 = (1/(16*sqrt(n*log(n)))) * min(sqrt((1-q)/q), sqrt(n/pow(log(n),4)));
		c2 = (1/(16*sqrt(n*log(n)))) * min(sqrt(p/(1-p)),1.0);
	}

	for(long i=0; i<n; i++)
	{
		C(i,i) = 0;
		for(long j=0;j<i;j++)
		{
			if(A(i,j)==1.0)
			{
				C(i,j) = c1;
			}
			else
			{
				C(i,j) = c2;
			}
			C(j,i) = C(i,j);
		}

	}
	#ifdef DEBUG
	printf("\nc1=%lf, c2=%lf",c1,c2);
	#endif
	return;	
	
}

void ALMEigDenseJacobi(emat_t &Y, emat_t &S, emat_t &A, emat_t &C, float rho)
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
	int max_iter = MAX_ALM_ITER;
	float mu0 = 1;
	float mu = mu0;
	
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Sigma = svd.singularValues();
	argmat1 = A-Y;
	double objOld = Sigma.sum() +  L1NormMatDense(C, argmat1);
	double objNew;

	// Main Loop
	while(change>delta)
	{	
		#ifdef DEBUG
		if(iter_count%10==0) printf("\nIterations %d to %d",iter_count+1,iter_count+10);
		#endif
		if(iter_count>max_iter){
			printf("\n****WARNING****>> ALM not converging ... Terminating");
			break;
		}
		
		svd.compute(A-S+(1/mu)*M,Eigen::ComputeThinU | Eigen::ComputeThinV);
		Sigma = svd.singularValues();
		SigmaDash = Eigen::MatrixXd::Zero(n,n);
		for(long i=0; i<n; i++)
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
		Ybar = svd.matrixU()*SigmaDash*svd.matrixV().transpose();

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
		svd.compute(Y,Eigen::ComputeThinU | Eigen::ComputeThinV);
		Sigma = svd.singularValues();
		argmat1 = A-Y;
		objNew = Sigma.sum() +  L1NormMatDense(C, argmat1);
		change = abs(objOld-objNew);
		objOld = objNew;

	}

	return;
}

long clusterCPDense(long *theta, emat_t &A, double p, double q, long gamma, int max_iter)
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
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Y1, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// Main loop
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
		ALMEigDenseJacobi(Y, S, A, C, rho);

		makeClusterDense(Y1, Y, .45);

		svd.compute(Y1,Eigen::ComputeThinU | Eigen::ComputeThinV);
		SingVal = svd.singularValues();
		sigma = calcSigma(SingVal,n);

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

#endif /*eigen_lib.h*/
