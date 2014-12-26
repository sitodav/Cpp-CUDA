//programma calcolo prod scalare tra 2 vettori

#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;
//input: n size vettori, n blocchi, nthread per blocco (NB: in questo esempio non si mappa 1 cella array input con un thread)
//questo esercizio usa il paradigma gather (ogni thread di un blocco calcola una sua porzione di array, vedi slide)
__host__
void inizializza(int *a,int n){
	for(int i=0;i<n;i++)
		a[i]=1+rand()%10;
}

__host__
void stampa(int *a,int n){
	
	cout<<endl<<"-------------------"<<endl;
	for(int i=0;i<n;i++)
		cout<<a[i]<<" ";
	cout<<endl;
}

__host__
void prodScalareCPU(int *a,int *b,unsigned long long int *c,int n){
	for(int i=0;i<n;i++){
		c[i]=a[i]*b[i];
		c[i]+= ((i>0) ? c[i-1] : 0);
	}
}

__global__
void prodScalareGPUNOSH(int *a,int *b,unsigned long long int *c,int n){
	
	int nThreadsGriglia = gridDim.x * blockDim.x; //numero di blocchi per numero di thread nel blocco, è proprio il numero di thread totali in griglia
	int stridePerThread = n / nThreadsGriglia; //il numero di elementi di competenza di ogni singolo thread
	int idOnGriglia = blockIdx.x * blockDim.x + threadIdx.x;
	c[idOnGriglia] = 0;
	for(int i=0;i<stridePerThread;i++){
		if(idOnGriglia * stridePerThread + i < n)
			c[idOnGriglia]+= a[idOnGriglia * stridePerThread + i] * b[idOnGriglia * stridePerThread + i];
		else break;
	}

}

__global__
void prodScalareGPUSH (int *a,int *b,unsigned long long int *c,int n){
	extern __shared__ int buffer[]; //questo viene allocato (nella chiamata kernel) per un totale del numero di threads NEL blocco
	int totThreadsInGriglia = gridDim.x * blockDim.x;
	int strideThread = n / totThreadsInGriglia; 
	int idOnGrid = blockDim.x * blockIdx.x + threadIdx.x;
	
	buffer[threadIdx.x]=0;
	
	for(int i=0;i<strideThread;i++){
		if(idOnGrid*strideThread + i < n)
			buffer[threadIdx.x]+= a[idOnGrid * strideThread +i] * b[idOnGrid * strideThread + i];
		else break;
	}
	__syncthreads();
	

	if(threadIdx.x == 0){
		c[blockIdx.x]=0;
		for(int i=0;i<blockDim.x;i++)
			c[blockIdx.x]+=buffer[i];
	}
}


int main(int argc,char **argv){
	srand((unsigned int)time(NULL));
	dim3 sizeGriglia,sizeBlocco;
	int n;
	float time1,time2;
	cudaEvent_t start1,stop1,start2,stop2;
	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);
	
	
	if(argc!=4){
		n=10;
		sizeGriglia.x=1;
		sizeBlocco.x=2; //2 threads per blocco, 1 blocco in totale, 10 elementi, 2 threads totali, 5 elementi calcolati per ogni thread
	}
	else{
		sscanf(argv[1],"%d",&n);
		sscanf(argv[2],"%d",&sizeGriglia.x);
		sscanf(argv[3],"%d",&sizeBlocco.x);
		//avrà sizeGriglia.x * sizeBlocco.x threads totali = threads tot.
		//avrà n/threads tot elementi per thread 
	}
	
	int *ha,*hb;
	unsigned long long int *hc;
	ha=new int[n];
	hb=new int[n];
	hc=new unsigned long long int[n]();
	inizializza(ha,n);
	inizializza(hb,n);
	//stampa(ha,n);
	//stampa(hb,n);
	cudaEventRecord(start1,0);
	prodScalareCPU(ha,hb,hc,n);
	cudaEventRecord(stop1,0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time1,start1,stop1);
	
	cout<<"il prod scalare (calcolato on cpu): "<<hc[n-1]<<" , tempo: "<<time1<<endl;
	
	//per calcolo su gpu no shared memory (vedi slide)
	int *da,*db;
	unsigned long long int *dc;
	cudaMalloc(&da,n*sizeof(int));
	cudaMalloc(&db,n*sizeof(int));
	int totThreads = sizeGriglia.x * sizeBlocco.x;
	cudaMalloc(&dc,sizeof(unsigned long long int) * totThreads);
	cudaMemcpy(da,ha,n*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(db,hb,n*sizeof(int),cudaMemcpyHostToDevice);
	prodScalareGPUNOSH<<<sizeGriglia,sizeBlocco>>>(da,db,dc,n);
	cudaThreadSynchronize();
	unsigned long long int *copy = new unsigned long long int[totThreads];
	cudaMemcpy(copy,dc,totThreads * sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
	unsigned long long int finalResult=0;
	
	
	for(int i=0;i<totThreads;i++)
		finalResult+=copy[i];
	cout<<"prod scalare calcolato on gpu senza shared memory: "<<finalResult<<endl;
	
	unsigned long long int *dc2;
	cudaMalloc(&dc2,sizeGriglia.x*sizeof(unsigned long long int)); //1 elemento per ogni blocco
	cudaEventRecord(start2,0);
	prodScalareGPUSH<<<sizeGriglia,sizeBlocco,sizeof(unsigned long long int)*sizeBlocco.x>>>(da,db,dc2,n);
	cudaThreadSynchronize();
	
	unsigned long long int *copy2 = new unsigned long long int[sizeGriglia.x];
	cudaMemcpy(copy2,dc2,sizeGriglia.x * sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
	finalResult=0;
	//stampa(copy2,sizeGriglia.x);
	for(int i=0;i<sizeGriglia.x;i++)
		finalResult+=copy2[i];
	cudaEventRecord(stop2,0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&time2,start2,stop2);
	
	cout<<"prod scalare calcolato su gpu con shared memory "<<finalResult<<" , tempo: "<<time2<<endl;
	
	
	
	delete[] ha;
	delete[] hc;
	delete[] hb;
	delete[] copy;
	cudaFree(db); 
	cudaFree(da);
	cudaFree(dc);
	cudaEventDestroy(start1);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop1);
	cudaEventDestroy(stop2);

}