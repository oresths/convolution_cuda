#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

void random_initialization(float **pinakas, int m, int n){

	int i, j;

	srand(m*n*time(NULL)); // generate differenT random numbers
	// srand(0); // generate the same random numbers on every run
	// random floating points [0 1]
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			pinakas[i][j] = (float)(rand()/(float)RAND_MAX) + (float)(rand()%5 - rand()%5);
		}

	}
}

void save(float **pinakas, int m, int n, char *filename){

	FILE *outfile;

	remove(filename);	//Afairei prwta tyxon arxeia me to idio onoma.
	printf("Saving data to files: "); printf(filename); printf("\n");

	/*===========Save to file===========*/
	if((outfile=fopen(filename, "a+b")) == NULL){
		printf("Can't open output file\n");
	}

	int i;
	
	for (i=0;i<m;i++)
		fwrite(pinakas[i], sizeof(float), n, outfile);

	fclose(outfile);

}

void error_message(){

	char *help = "Error using syneliksh: Four argumenTs required\n"
			"First: mF\n"
			"Second: nF\n"
			"Third: mT\n"
			"Fourth: nT\n";

	printf(help);

}

void seiriakh_syneliksh(float **F, float **T, float **pF, float **Y, int mF, int nF, int mT, int nT)
{
	int mpF=mT+mT+mF-2, npF=nT+nT+nF-2, mY=mF+mT-1, nY=nF+nT-1;
	int i, j, ii, jj;
	for (i=0;i<mpF;i++){
		for (j=0;j<npF;j++){
			pF[i][j]=0;
		}
	}
	for (i=0;i<mF;i++){
		for (j=0;j<nF;j++){
			pF[i+mT-1][j+nT-1]=F[i][j];
		}
	}

	for (i = 0; i<mY ;i++){
		for (j = 0; j<nY;j++){
			Y[i][j]=0;
			for (ii=0;ii<mT;ii++){
				for (jj=0; jj<nT; jj++){

					Y[i][j] = Y[i][j] + pF[i + ii][j + jj] * T[ii][jj];
				}
			}
		}
	}
}


int main(int argc, char **argv){

	struct timeval first, second, lapsed;
	struct timezone tzp;

	if(argc<5){
		error_message();
		return 0;
		//printf("Error using kmeans: Three argumenTs required\n");
	}

	int mF = atoi(argv[1]);
	int nF = atoi(argv[2]);
	int mT = atoi(argv[3]);
	int nT = atoi(argv[4]);
	
	int mpF=mT+mT+mF-2, npF=nT+nT+nF-2, mY=mF+mT-1, nY=nF+nT-1;

	int i;
	
	float **F=(float **)malloc(mF*sizeof(float *));
	for (i=0;i<mF;i++)
		F[i]=(float *)malloc(nF*sizeof(float));

	float **T=(float **)malloc(mT*sizeof(float *));
	for (i=0;i<mT;i++)
		T[i]=(float *)malloc(nT*sizeof(float));

	float **pF=(float **)malloc((mpF)*sizeof(float *));
	for (i=0;i<mpF;i++)
		pF[i]=(float *)malloc((npF)*sizeof(float));

	float **Y=(float **)malloc((mY)*sizeof(float *));
	for (i=0;i<mY;i++)
		Y[i]=(float *)malloc((nY)*sizeof(float));

	
	random_initialization(F, mF, nF);
	random_initialization(T, mT, nT);
	

	gettimeofday(&first, &tzp);
	seiriakh_syneliksh(F, T, pF, Y, mF, nF, mT, nT);
  	gettimeofday(&second, &tzp);
  	
  	int j;
  	printf("\n");
  	for (i=0;i<mF;i++){
  		printf("\n");
  		for (j=0;j<nF;j++)
  			printf("%f ", F[i][j]);
  	}
	
	  	printf("\n\n");
  	for (i=0;i<mT;i++){
  		printf("\n");
  		for (j=0;j<nT;j++)
  			printf("%f ", T[i][j]);
  	}
	
	  	printf("\n");
  	for (i=0;i<mpF;i++){
  		printf("\n");
  		for (j=0;j<npF;j++)
  			printf("%f ", pF[i][j]);
  	}
  	
  	  	printf("\n");
  	for (i=0;i<mY;i++){
  		printf("\n");
  		for (j=0;j<nY;j++)
  			printf("%f ", Y[i][j]);
  	}
	printf("\n");
	
	if(first.tv_usec>second.tv_usec){
		second.tv_usec += 1000000;
		second.tv_sec--;
	}

	lapsed.tv_usec = second.tv_usec - first.tv_usec;
	lapsed.tv_sec = second.tv_sec - first.tv_sec;

	printf("\nTime elapsed: %d.%06dsec\n", (int)lapsed.tv_sec, (int)lapsed.tv_usec);

	save(F, mF, nF, "F.bin");
	save(T, mT, nT, "T.bin");
	save(pF, mpF, npF, "pF.bin");
	save(Y, mY, nY, "Y.bin");

	return 0;

}
