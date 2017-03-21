#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cutil_inline.h>

//#define BLOCK_SIZE 4
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16


__constant__ float d_T[1000];
__constant__ int d_mpF, d_npF, d_mT, d_nT, d_mY, d_nY;
__constant__ int s_mpF, s_npF, plh8os_s_pF, phliko, modulo;


void random_initialization(float *pinakas, int m, int n){

	int i, j;

	srand(m*n*time(NULL)); // generate differenT random numbers
	// srand(0); // generate the same random numbers on every run
	// random floating points [0 1]
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			pinakas[i*n+j] = (float)(rand()/(float)RAND_MAX) + (float)(rand()%5 - rand()%5);
		}

	}
}

void save(float *pinakas, int m, int n, char *filename){

	FILE *outfile;

	remove(filename);	//Afairei prwta tyxon arxeia me to idio onoma.
	printf("Saving data to files: "); printf(filename); printf("\n");

	/*===========Save to file===========*/
	if((outfile=fopen(filename, "a+b")) == NULL){
		printf("Can't open output file\n");
	}
	
	fwrite(pinakas, sizeof(float), m*n, outfile);

	fclose(outfile);

}

void error_message(){

	char *help = "Error using syneliksh: Four arguments required\n"
			"First: mF\n"
			"Second: nF\n"
			"Third: mT\n"
			"Fourth: nT\n";

	printf(help);

}

void seiriakh_syneliksh(float *F, float *T, float *pF, float *Y, int mF, int nF, int mT, int nT)
{
	int mpF=mT+mT+mF-2, npF=nT+nT+nF-2, mY=mF+mT-1, nY=nF+nT-1;
	int i, j, ii, jj;

	/**Perikleei ton F me ta mhdenika poy xreiazetai gia na ginei h syneliksh.**/
	for (i=0;i<mpF;i++){
		for (j=0;j<npF;j++){
			pF[i*npF+j]=0;
		}
	}
	for (i=0;i<mF;i++){
		for (j=0;j<nF;j++){
			pF[ (i+(mT-1))*npF + (j+(nT-1)) ]=F[i*nF+j];
		}
	}

	/**YPologizei thn syneliksh.**/
	for (i = 0; i<mY ;i++){
		for (j = 0; j<nY;j++){
			Y[i*nY+j]=0;
			for (ii=0;ii<mT;ii++){
				for (jj=0; jj<nT; jj++){

					Y[i*nY+j] = Y[i*nY+j] + pF[(i + ii)*npF + (j + jj)] * T[ii*nT+jj];
				}
			}
		}
	}
}

__global__ void syneliksh(float *d_pF, float *d_Y)
{                          
	__shared__ float s_pF[4000];	//Xwros gia to kommati toy pF pou 8a paei sthn shared.
		
	/*if (blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==3&&threadIdx.y==3)
		*apo=1;*/

	/*int s_mpF=(blockDim.x-1)+d_mT, s_npF=(blockDim.y-1)+d_nT;
	int plh8os_s_pF=s_mpF*s_npF;
	int phliko=plh8os_s_pF/(blockDim.x*blockDim.y);
	int modulo=plh8os_s_pF%(blockDim.x*blockDim.y);*/
	
	/*//slower
	for(k=tx*(BLCK_SIZE_X+con_mT)/BLCK_SIZE_X;k<(tx+1)*(BLCK_SIZE_X+con_mT)/BLCK_SIZE_X;k++){
		for(l=ty*(BLCK_SIZE_Y+con_nT)/BLCK_SIZE_Y;l<(ty+1)*(BLCK_SIZE_Y+con_nT)/BLCK_SIZE_Y;l++){
			shared[k][l]=pF[(k+bx*BLCK_SIZE_X)*(con_nF+2*(con_nT-1))+l+by*BLCK_SIZE_Y];
		}
	}*/

	int deikths=threadIdx.x*BLOCK_SIZE_Y+threadIdx.y;	//Xarakthristikos deikths gia ka8e nhma mesa sto mplok.
	int k=0;
	if (deikths<modulo) k=1;	//8elw ola ta nhmata na metaferoyn peripou iso ari8mo stoixeiwn toy pF sthn shared, alla epeidh h diairesh den einai-> 
	int ar_metaf=phliko + k;	// ->teleia, ta prwta plh8os_s_pF%mege8os_mplok nhmata 8a metaferoun ena parapanw stoixeio.
	int arxikh_8esh=ar_metaf*deikths*k + ((phliko+1)*modulo+ar_metaf*(deikths-modulo)) * (1-k);    //H 8esh apo opou ksekinaei h perioxh thn opoia 8a metaferei ka8e nhma.
	int iL=blockIdx.x*BLOCK_SIZE_X;		//Oi syntetagmenes toy arxikoy shmeioy ston pinaka pF, apo to opoio kai meta ka8e mplok 8a analabei na->
	int jL=blockIdx.y*BLOCK_SIZE_Y;		// ->metaferei ta stoixeia apo ton pF ston s_pF.
	int ii, jj;
	for (ii=arxikh_8esh;ii<arxikh_8esh+ar_metaf;ii++)
		s_pF[ii]=d_pF[iL*d_npF + jL + (ii/s_npF)*d_npF + (ii%s_npF)];

	/*int iL=blockIdx.x*BLOCK_SIZE;
	int jL=blockIdx.y*BLOCK_SIZE;
	int ii, jj;
	for (ii=0;ii<s_mpF;ii++)
		for (jj=0;jj<s_npF;jj++)
			s_pF[ii*s_npF + jj]=d_pF[(iL+ii)*d_npF + jL+jj];*/

	__syncthreads();	//Perimenw na teleiwsoyn ola ta nhmata thn metafora sth shared prin synexisw.

	
	float sum=0;
	int i=blockIdx.x*BLOCK_SIZE_X + threadIdx.x;	//Oi syntetagmenes toy ka8e nhmatos mesa ston Y, o opoios exei xristei se mplok.
	int j=blockIdx.y*BLOCK_SIZE_Y + threadIdx.y;	//									>>

	if ((i<d_mY)&&(j<d_nY)){	//Epeidh oi diastaseis toy Y den einai panta pollaplasia toy mplok kanoyme elegxo ka8e fora wste na mhn ksefygoyme.
		for (ii=0;ii<d_mT;ii++){
			for (jj=0; jj<d_nT; jj++){
				sum+= s_pF[(threadIdx.x+ii)*s_npF + threadIdx.y+jj] * d_T[ii*d_nT+jj] ;	   //Ka8e nhma ypologizei ena stoixeio toy Y.
			}
		}
		d_Y[i*d_nY + j]=sum;}
}


int main(int argc, char **argv){
	

	// get number of SMs on this GPU
	int devID;
    cudaDeviceProp props;
    cutilSafeCall(cudaGetDevice(&devID));
    cutilSafeCall(cudaGetDeviceProperties(&props, devID));

    printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);
	printf("Ari8mos MP: %d\n",props.multiProcessorCount);

	struct timeval first, second, lapsed;
	struct timezone tzp;

	if(argc<5){
		error_message();
		return 0;
	}

	int mF = atoi(argv[1]);
	int nF = atoi(argv[2]);
	int mT = atoi(argv[3]);
	int nT = atoi(argv[4]);
	
	int mpF=mT+mT+mF-2, npF=nT+nT+nF-2, mY=mF+mT-1, nY=nF+nT-1;
	
	float *F=(float *)malloc(mF*nF*sizeof(float));
	float *T=(float *)malloc(mT*nT*sizeof(float));
	float *pF=(float *)malloc((mpF*npF)*sizeof(float));
	float *Y=(float *)malloc((mY*nY)*sizeof(float));
		
	random_initialization(F, mF, nF);
	random_initialization(T, mT, nT);

	gettimeofday(&first, &tzp);
	seiriakh_syneliksh(F, T, pF, Y, mF, nF, mT, nT);   //Ftiaxnei kai ton pF poy xreiazetai kai apo thn cuda.
	gettimeofday(&second, &tzp);

	if(first.tv_usec>second.tv_usec){
		second.tv_usec += 1000000;
		second.tv_sec--;
	}

	lapsed.tv_usec = second.tv_usec - first.tv_usec;
	lapsed.tv_sec = second.tv_sec - first.tv_sec;

	printf("\nXronos seiriakhs synelikshs: %d.%06dsec\n", (int)lapsed.tv_sec, (int)lapsed.tv_usec);

	// create CUDA event handles
	cudaEvent_t start_event, stop_event;
	cutilSafeCall( cudaEventCreate(&start_event) );
	cutilSafeCall( cudaEventCreate(&stop_event) );

	// allocate device memory
	float *d_pF, *d_Y;
	cutilSafeCall(cudaMalloc((void**) &d_pF, mpF*npF*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**) &d_Y, mY*nY*sizeof(float)));

	/**Ypologismos kapoiwn sta8erwn poy 8a steilw sthn constant ths kartas**/
	int h_s_mpF=(BLOCK_SIZE_X-1)+mT, h_s_npF=(BLOCK_SIZE_Y-1)+nT;	//Oi diastaseis toy pinaka pF sthn shared.
	int h_plh8os_s_pF=h_s_mpF*h_s_npF;	  //To plh8os twn stoixeiwn toy pinaka pF sthn shared.
	int h_phliko=h_plh8os_s_pF/(BLOCK_SIZE_X*BLOCK_SIZE_Y);	   //Xreiazetai gia na ypologisw posa stoixeia toy pF 8a metaferei sthn shared ka8e nhma.
	int h_modulo=h_plh8os_s_pF%(BLOCK_SIZE_X*BLOCK_SIZE_Y);	   //										>>

	float xronos_metaforas_eisodou;
	cudaEventRecord(start_event, 0);
	// copy host memory to device
	cutilSafeCall(cudaMemcpy(d_pF, pF, mpF*npF*sizeof(float), cudaMemcpyHostToDevice) );

	cudaMemcpyToSymbol("d_T", T, mT*nT*sizeof(float));

	cudaMemcpyToSymbol("d_mpF", &mpF, sizeof(int));
	cudaMemcpyToSymbol("d_npF", &npF, sizeof(int));
	cudaMemcpyToSymbol("d_mT", &mT, sizeof(int));
	cudaMemcpyToSymbol("d_nT", &nT, sizeof(int));
	cudaMemcpyToSymbol("d_mY", &mY, sizeof(int));
	cudaMemcpyToSymbol("d_nY", &nY, sizeof(int));

	cudaMemcpyToSymbol("s_mpF", &h_s_mpF, sizeof(int));
	cudaMemcpyToSymbol("s_npF", &h_s_npF, sizeof(int));
	cudaMemcpyToSymbol("plh8os_s_pF", &h_plh8os_s_pF, sizeof(int));
	cudaMemcpyToSymbol("phliko", &h_phliko, sizeof(int));
	cudaMemcpyToSymbol("modulo", &h_modulo, sizeof(int));
	
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);   // block until the event is actually recorded
	cutilSafeCall( cudaEventElapsedTime(&xronos_metaforas_eisodou, start_event, stop_event) );
	printf("\n\nXronos metaforas twn dedomenwn eisodoy sthn karta: %fsec\n", xronos_metaforas_eisodou/1000);


	// setup execution parameters
	dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid(mY/threads.x +1, nY/threads.y +1);	//Pros8etw 1 se ka8e diastash toy grid epeidh oi diastaseis toy Y den einai panta pollaplasia toy mplok.

	printf("\nRun Kernel...\n\n");
	
	float xronos_synelikshs;
	cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

	// execute the kernel
	syneliksh<<< grid, threads >>>(d_pF, d_Y);


	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);   // block until the event is actually recorded
	cutilSafeCall( cudaEventElapsedTime(&xronos_synelikshs, start_event, stop_event) );
	printf("Xronos ypologismoy synelikshs: %fsec\n", xronos_synelikshs/1000);

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed");

	//cudaThreadSynchronize();	//Wait for compute-device to finish.

	// allocate host memory for the result
	float *h_Y = (float*) malloc(mY*nY*sizeof(float));
	float xronos_lhpshs_eksodou;
	cudaEventRecord(start_event, 0);
	cutilSafeCall(cudaMemcpy(h_Y, d_Y, mY*nY*sizeof(float), cudaMemcpyDeviceToHost));	//Apo8hkeyw ston h_Y to apotelesma ths kartas.
		
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);   // block until the event is actually recorded
	cutilSafeCall( cudaEventElapsedTime(&xronos_lhpshs_eksodou, start_event, stop_event) );
	printf("\nXronos poy apaiteitai gia thn lhpsh twn apotelesmatwn apo thn karta: %fsec\n", xronos_lhpshs_eksodou/1000);

	/**Ypologizei thn megisth diafora anamesa stoys pinakes poy ypologisthkan seiriaka kai parallhla.**/
	int i, j;
	float max=-1;
	for (i=0;i<mY;i++)
		for (j=0;j<nY;j++)
			if (fabs(h_Y[i*nY+j]-Y[i*nY+j]) > max) max=fabs(h_Y[i*nY+j]-Y[i*nY+j]);

	printf("\nMegisth diafora: %f\n\n", max);

	save(F, mF, nF, "F.bin");
	save(T, mT, nT, "T.bin");
	save(pF, mpF, npF, "pF.bin");
	save(Y, mY, nY, "Y.bin");

	free(F);
	free(T);
	free(pF);
	free(Y);
	free(h_Y);
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	cutilSafeCall(cudaFree(d_pF));
	cutilSafeCall(cudaFree(d_Y));

	cudaThreadExit();	//Exit and clean-up from CUDA launches.
	
	return 0;
}
