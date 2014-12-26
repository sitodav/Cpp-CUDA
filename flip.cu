#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>


using namespace std;

//programma per il flip di un vettore
//input: n size vettore, n n blocchi 

__host__ void inizializza(int *a,int n){
for(int i=0;i<n;i++)
	a[i]=1+rand()%n;

}

__host__ void stampa(int *a,int n){
cout<<endl<<"--------------------------"<<endl;
for(int i=0;i<n;i++)
	cout<<a[i]<<" ";
	cout<<endl;
}

__global__ void flipGPU_nosh(int *a,int *b,int n){
	int idGriglia = threadIdx.x + blockDim.x * blockIdx.x;
	b[n-1-idGriglia] = a[idGriglia];
}

__global__ void flipGPU_sh(int *a,int *b,int n){
	extern __shared__ int buffer[];
	int idOnGrid = threadIdx.x + blockDim.x * blockIdx.x;
	buffer[blockDim.x - 1 - threadIdx.x] = a[idOnGrid];
	__synchthreads();
	
	b [ (gridDim.x -1 - blockIdx.x )*blockDim.x + threadIdx.x ] = buffer [threadIdx.x];
	
}

int main(int argc,char *argv[]){
	
srand((unsigned int)time(NULL));

int n;
dim3 sizeGriglia,sizeBlocchi;
if(argc!=3){
	n=20;
	sizeGriglia.x = 5;
}
else{
	sscanf(argv[1],"%d",&n);
	sscanf(argv[2],"%d",&sizeGriglia.x);
}
sizeBlocchi.x = n / sizeGriglia.x;

int *ha=new int[n];
int *da,*db;
cudaMalloc(&da,n*sizeof(int));
cudaMalloc(&db,n*sizeof(int));
cudaMemset(db,0,n*sizeof(int));
inizializza(ha,n);
stampa(ha,n);
cudaMemcpy(da,ha,n*sizeof(int),cudaMemcpyHostToDevice);
//flipGPU_nosh<<<sizeGriglia,sizeBlocchi>>>(da,db,n);
flipGPU_sh<<< sizeGriglia,sizeBlocchi,sizeBlocchi.x * sizeof(int)>>>(da,db,n);
int *copy=new int[n];
cudaMemcpy(copy,db,n*sizeof(int),cudaMemcpyDeviceToHost);
stampa(copy,n);

	
	
	
}