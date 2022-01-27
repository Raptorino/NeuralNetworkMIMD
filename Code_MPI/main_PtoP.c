#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <limits.h>
#include <mpi.h>
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

void trainN(int my_rank, int nprocs){

	char **tSet;

    float DeltaWeightIH[NUMHID][NUMIN], DeltaWeightHO[NUMOUT][NUMHID];
    float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
 	int ranpat[NUMPAT];
 	float Hidden[NUMHID], Output[NUMOUT], DeltaO[NUMOUT], DeltaH[NUMHID];
 	float SumO, SumH, SumDOW;
    //int trobat = 0;
    float aWeightIH[NUMHID][NUMIN];
    float aWeightHO[NUMOUT][NUMHID];
    float aError = 0.0;
    MPI_Status state;


	if( (tSet = loadPatternSet( NUMPAT, "optdigits.tra", 1 ) ) == NULL){
        printf( "Loading Patterns: Error!!\n" );
		exit( -1 );
	}

	for( int i = 0; i < NUMHID; i++ )
		for( int j = 0; j < NUMIN; j++ ){
			WeightIH[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
			DeltaWeightIH[i][j] = 0.0;
		}

	for( int i = 0; i < NUMOUT; i++)
		for( int j = 0; j < NUMHID; j++ ){
			WeightHO[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
			DeltaWeightHO[i][j] = 0.0;
		}


    	for( int epoch = 0 ; epoch < 1000000; epoch++ ) 
        {

        	for( int p = 0 ; p < NUMPAT ; p++ )   // randomize order of individuals
            	ranpat[p] = p;
        	
            for( int p = 0 ; p < NUMPAT ; p++) 
            {
                int x = rando();
                int np = (x*x)%NUMPAT;
                int op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        	}

        	Error = BError = 0.0;

        	printf("."); fflush(stdout);

            for (int nb = my_rank*(NUMPAT/BSIZE)/nprocs; nb < (my_rank + 1)*(NUMPAT/BSIZE)/nprocs; nb++) // repeat for all batches
            { 
            	BError = 0.0;
                for( int np = nb*BSIZE ; np < (nb + 1)*BSIZE ; np++ ) // repeat for all the training patterns within the batch
                {    

                    int p = ranpat[np];
                    for( int j = 0 ; j < NUMHID ; j++ )     // compute hidden unit activations
                    {
                        SumH = 0.0;
                        for( int i = 0 ; i < NUMIN ; i++ ) SumH += tSet[p][i] * WeightIH[j][i];
                        Hidden[j] = 1.0/(1.0 + exp( -SumH )) ;
                    }

                    for( int k = 0 ; k < NUMOUT ; k++ )     // compute output unit activations and errors
                    {
                        SumO = 0.0;
                        for( int j = 0 ; j < NUMHID ; j++ ) SumO += Hidden[j] * WeightHO[k][j] ;
                        Output[k] = 1.0/(1.0 + exp(-SumO)) ;   // Sigmoidal Outputs
                        BError += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]) ;   // SSE
                        DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]) ;   // Sigmoidal Outputs, SSE
                    }

                    for( int j = 0 ; j < NUMHID ; j++ )     // update delta weights DeltaWeightIH
                    { 
                        SumDOW = 0.0 ;
                        for( int k = 0 ; k < NUMOUT ; k++ ) SumDOW += WeightHO[k][j] * DeltaO[k] ;
                        DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]) ;
                        for( int i = 0 ; i < NUMIN ; i++ )
                            DeltaWeightIH[j][i] = eta * tSet[p][i] * DeltaH[j] + alpha * DeltaWeightIH[j][i];
                    }

                    for( int k = 0 ; k < NUMOUT ; k ++ )    // update delta weights DeltaWeightHO
                        for( int j = 0 ; j < NUMHID ; j++ )
                            DeltaWeightHO[k][j] = eta * Hidden[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];

                }

                Error += BError;
                for( int j = 0 ; j < NUMHID ; j++ )     // update weights WeightIH
                        for( int i = 0 ; i < NUMIN ; i++ )
                            WeightIH[j][i] += DeltaWeightIH[j][i] ;

                for( int k = 0 ; k < NUMOUT ; k ++ )    // update weights WeightHO
                        for( int j = 0 ; j < NUMHID ; j++ )
                            WeightHO[k][j] += DeltaWeightHO[k][j] ;
            
            }


            if(my_rank == 0)
            {

                for (int i = 1; i < nprocs; i++)
                {

                    MPI_Recv(&aError, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &state);
                    Error += aError;
                   
                }

                Error = Error / nprocs;
               

                for (int x = 1; x < nprocs; x++)
                {

                    MPI_Send(&Error, 1, MPI_FLOAT, x, 0, MPI_COMM_WORLD);
                   
                }


                for (int i = 1; i < nprocs; i++)
                {
                    MPI_Recv(aWeightIH, NUMHID*NUMIN, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &state);
                   
                    for( int j = 0 ; j < NUMHID ; j++ )     // update weights WeightIH
                        for( int i = 0 ; i < NUMIN ; i++ )
                            WeightIH[j][i] +=  aWeightIH[j][i] ;

                    MPI_Recv(aWeightHO, NUMOUT*NUMHID, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &state);
                    
                    for( int k = 0 ; k < NUMOUT ; k ++ )    // update weights WeightHO
                            for( int j = 0 ; j < NUMHID ; j++ )
                                WeightHO[k][j] += aWeightHO[k][j] ;


                }

            
                for( int j = 0 ; j < NUMHID ; j++ )     // Promig / nprocs
                    for( int i = 0 ; i < NUMIN ; i++ )
                        WeightIH[j][i] = WeightIH[j][i] / nprocs ;

                for( int k = 0 ; k < NUMOUT ; k ++ )    // Promig / nprocs
                    for( int j = 0 ; j < NUMHID ; j++ )
                        WeightHO[k][j] = WeightHO[k][j] / nprocs ;        

     

            }
            else
            {
                //MPI_Bcast(&Error, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Send(&Error, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

                MPI_Recv(&Error, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &state);

                MPI_Send(WeightIH, NUMHID*NUMIN, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(WeightHO, NUMOUT*NUMHID, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

               // MPI_Recv(WeightIH, NUMHID*NUMIN, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &state);
               // MPI_Recv(WeightHO, NUMOUT*NUMHID, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &state);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            Error = Error/((NUMPAT/BSIZE)*BSIZE);	//mean error for the last epoch 		
            //printf( "\n Errorh %f :Im the process %i \n", Error, my_rank);               

            if( !(epoch%100) ) printf( "\nEpoch %-5d :   Error = %f \n", epoch, Error ) ;
            
            if( Error < 0.0004 ) 
            {
        		printf( "\nEpoch %-5d :   Error = %f \n", epoch, Error ) ; 
                printf("Im the process %i \n", my_rank);               
                break ;  // stop learning when 'near enough'
            }
            

    	}

    	freeTSet( NUMPAT, tSet );

     	printf( "END TRAINING\n" );
}

void printRecognized( int p, float Output[] ){
	int imax = 0;

	for( int i = 1; i < NUMOUT; i++)
		if ( Output[i] > Output[imax] ) imax = i;

	printf( "El patrÃ³ %d sembla un %c\t i Ã©s un %d", p, '0' + imax, Validation[p] );
	
    if( imax == Validation[p] ) total++;
    
    for( int k = 0 ; k < NUMOUT ; k++ )
        	printf( "\t%f\t", Output[k] ) ;
    
    printf( "\n" );
}


void runN(){

    printf( "\n Inici Run \n") ;
    
    
        char **rSet;
        char *fname[NUMRPAT];

        printf( "\n Inici Run Per Root \n") ;

        if( (rSet = loadPatternSet( NUMRPAT, "optdigits.cv", 0 )) == NULL)
        {
            printf( "Error!!\n" );
            exit( -1 );
        }

        printf( "\n Pattern carregat \n") ;

        float Hidden[NUMHID], Output[NUMOUT];

            for( int p = 0 ; p < NUMRPAT ; p++ ) // repeat for all the recognition patterns
            {    
                for( int j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
                        float SumH = 0.0;
                        for( int i = 0 ; i < NUMIN ; i++ ) SumH += rSet[p][i] * WeightIH[j][i];
                        Hidden[j] = 1.0/(1.0 + exp( -SumH )) ;
                }

                for( int k = 0 ; k < NUMOUT ; k++ ) {    // compute output unit activations
                        float SumO = 0.0;
                        for( int j = 0 ; j < NUMHID ; j++ ) SumO += Hidden[j] * WeightHO[k][j] ;
                        Output[k] = 1.0/(1.0 + exp( -SumO )) ;   // Sigmoidal Outputs
                }
                
                printRecognized( p, Output );

            }

        printf( "\nTotal encerts = %d\n", total );

        freeTSet( NUMRPAT, rSet );
    

}

int main(int argc, char *argv[]) 
{
    int my_rank, nprocs;

	clock_t start = clock();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    printf( "\n MPI iniciat \n") ;
	
    trainN(my_rank, nprocs);

    printf("Im the process %i \n", my_rank);               
    
    if (my_rank == 2)
    {    
        runN();
    }
    
    MPI_Finalize();


	clock_t end = clock();

	printf( "\n\nGoodbye! (%f sec)\n\n", (end-start)/(1.0*CLOCKS_PER_SEC) ) ;

    return 0 ;
}
