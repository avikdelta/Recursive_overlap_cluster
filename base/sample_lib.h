/* **************************************************************************
 * This file has some random sampling generation routines
 **************************************************************************** */

// Source: John D. Cook, http://stackoverflow.com/a/311716/15485

//#include <random>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>
/*
double GetUniform()
{
    static std::default_random_engine re;
    static std::uniform_real_distribution<double> Dist(0,1);
    return Dist(re);
}
*/

#define RANGE 100
double GetUniform()
{
	int num = rand() % (RANGE+1);
	long u = num/RANGE;
	return(u);
}

void SampleWithoutReplacement
(
    long populationSize,    // size of set sampling from
    long sampleSize,        // size of each sample
    std::vector<long> & samples  // output, zero-offset indicies to selected items
)
{
    // Use Knuth's variable names
    long& n = sampleSize;
    long& N = populationSize;

    long t = 0; // total input records dealt with
    long m = 0; // number of items selected so far
    double u;

    while (m < n)
    {
        u = GetUniform(); // call a uniform(0,1) random number generator

        if ( (N - t)*u >= n - m )
        {
            t++;
        }
        else
        {
            samples[m] = t;
            t++; m++;
        }
    }
    return;
}

