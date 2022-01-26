#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <limits.h>
#include <omp.h>
#include "common.h"


int total;
int seed=50;

int rando()
{
    seed = (214013*seed+2531011);
    return seed>>16;
}

float frando()
{
    return (rando()/65536.0f);
}

void freeTSet( int np, char **tset ){
	for( int i = 0; i < np; i++ ) free( tset[i] );
	free(tset);
}

void trainN()
{

    char** tSet;

    float DeltaWeightIH[NUMHID][NUMIN], DeltaWeightHO[NUMOUT][NUMHID];
    float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
    int ranpat[NUMPAT];
    float Hidden[NUMHID], Output[NUMOUT], DeltaO[NUMOUT], DeltaH[NUMHID];
    float SumO = 0.0;
    float SumH = 0.0;	
    float SumDOW = 0.0;
    int trobat = 0;

    if ((tSet = loadPatternSet(NUMPAT, "optdigits.tra", 1)) == NULL) {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }
    #pragma omp parallel shared (trobat, Hidden, Output, DeltaO, BError, Error, DeltaH,DeltaWeightIH,ranpat,DeltaWeightHO) firstprivate(SumH, SumO, SumDOW)
    {
        for (int i = 0; i < NUMHID; i++)
            for (int j = 0; j < NUMIN; j++) {
                WeightIH[i][j] = 2.0 * (frando() + 0.01) * smallwt;
                DeltaWeightIH[i][j] = 0.0;
            }

        for (int i = 0; i < NUMOUT; i++)
            for (int j = 0; j < NUMHID; j++) {
                WeightHO[i][j] = 2.0 * (frando() + 0.01) * smallwt;
                DeltaWeightHO[i][j] = 0.0;
            }


        for (int epoch = 0; epoch < 1000000 && trobat == 0; epoch++)
        {    // iterate weight updates

            #pragma omp for
            for (int p = 0; p < NUMPAT; p++)   // randomize order of individuals
                ranpat[p] = p;

            #pragma omp single   //s
            for (int p = 0; p < NUMPAT; p++) {
                int x = rando();
                int np = (x * x) % NUMPAT;
                int op = ranpat[p]; ranpat[p] = ranpat[np]; ranpat[np] = op;

            }

            Error = BError = 0.0;
            printf("."); fflush(stdout);

            for (int nb = 0; nb < NUMPAT / BSIZE; nb++) 
            { // repeat for all batches      
                BError = 0.0;
                for (int np = nb * BSIZE; np < (nb + 1) * BSIZE; np++)
                {    // repeat for all the training patterns within the batch

                    int p = ranpat[np];

                    #pragma omp for 
                    for (int j = 0; j < NUMHID; j++)
                    {    // compute hidden unit activations

                        for (int i = 0; i < NUMIN; i++) SumH += tSet[p][i] * WeightIH[j][i];
                        Hidden[j] = 1.0 / (1.0 + exp(-SumH));
                        SumH = 0.0;
                    }

                    #pragma omp for reduction(+:BError)
                    for (int k = 0; k < NUMOUT; k++)
                    {    // compute output unit activations and errors

                        for (int j = 0; j < NUMHID; j++) SumO += Hidden[j] * WeightHO[k][j];
                        Output[k] = 1.0 / (1.0 + exp(-SumO));   // Sigmoidal Outputs
                        BError += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]);   // SSE
                        DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]); // Sigmoidal Outputs, SSE
                        SumO = 0.0;
                    }

                    #pragma omp for
                    for (int j = 0; j < NUMHID; j++)
                    {     // update delta weights DeltaWeightIH
                          //SumDOW = 0.0 ;             
                        for (int k = 0; k < NUMOUT; k++) {
                            SumDOW += WeightHO[k][j] * DeltaO[k];
                        }
                        DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]);

                        for (int i = 0; i < NUMIN; i++)
                            DeltaWeightIH[j][i] = eta * tSet[p][i] * DeltaH[j] + alpha * DeltaWeightIH[j][i];
                        SumDOW = 0.0;
                    }

                    #pragma omp for collapse(2)
                    for (int k = 0; k < NUMOUT; k++)    // update delta weights DeltaWeightHO
                        for (int j = 0; j < NUMHID; j++)
                            DeltaWeightHO[k][j] = eta * Hidden[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
                }


                #pragma omp single
                {
                    Error += BError;
                }
                
                #pragma omp for collapse(2)
                for (int j = 0; j < NUMHID; j++)     // update weights WeightIH
                    for (int i = 0; i < NUMIN; i++)
                        WeightIH[j][i] += DeltaWeightIH[j][i];
                //#pragma omp barrier
                #pragma omp for collapse(2)
                for (int k = 0; k < NUMOUT; k++)    // update weights WeightHO
                    for (int j = 0; j < NUMHID; j++)
                        WeightHO[k][j] += DeltaWeightHO[k][j];
            }

            #pragma omp single
            {
                Error = Error / ((NUMPAT / BSIZE) * BSIZE);	//mean error for the last epoch 		
                if (!(epoch % 100)) printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);

                if (Error < 0.0004) 
                {
                    trobat = 1;
                    printf("\nEpoch %-5d :   Error = %f \n", epoch, Error);   // stop learning when 'near enough'
                }
            }        



        }
       
    }

    freeTSet(NUMPAT, tSet);

    printf("END TRAINING\n");

}
void printRecognized( int p, float Output[] )
{
	int imax = 0;

	for( int i = 1; i < NUMOUT; i++)
		if ( Output[i] > Output[imax] ) imax = i;

	printf( "El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p] );
	
	if( imax == Validation[p] ) total++;
    for( int k = 0 ; k < NUMOUT ; k++ )
        	printf( "\t%f\t", Output[k] ) ;
    
	printf( "\n" );
}

void runN()
{
	char **rSet;
	char *fname[NUMRPAT];

	if( (rSet = loadPatternSet( NUMRPAT, "optdigits.cv", 0 )) == NULL)
	{
		printf( "Error!!\n" );
		exit( -1 );
	}

	float Hidden[NUMHID], Output[NUMOUT];

    	for( int p = 0 ; p < NUMRPAT ; p++ ) 
		{    // repeat for all the recognition patterns
			
        	for( int j = 0 ; j < NUMHID ; j++ ) 
			{    // compute hidden unit activations
            		float SumH = 0.0;
            		for( int i = 0 ; i < NUMIN ; i++ ) SumH += rSet[p][i] * WeightIH[j][i];
            		Hidden[j] = 1.0/(1.0 + exp( -SumH )) ;
        	}
			
        	for( int k = 0 ; k < NUMOUT ; k++ ) 
			{    // compute output unit activations
            		float SumO = 0.0;
            		for( int j = 0 ; j < NUMHID ; j++ ) SumO += Hidden[j] * WeightHO[k][j] ;
            		Output[k] = 1.0/(1.0 + exp( -SumO )) ;   // Sigmoidal Outputs
        	}
        	printRecognized( p, Output );
    	}

	printf( "\nTotal encerts = %d\n", total );

	freeTSet( NUMRPAT, rSet );
}

int main() 
{
	clock_t start = clock();

	trainN();
	runN();

	clock_t end = clock();
	printf( "\n\nGoodbye! (%f sec)\n\n", (end-start)/(1.0*CLOCKS_PER_SEC) ) ;

    return 1 ;
}
