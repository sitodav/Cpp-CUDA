#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#define T 4

using namespace std;	

//prod matrice matrice terza versione(vedi slide)
//input: l,m,n, size blocco (blocchi bidimensionali, quadrati, l m ed n devono essere multipli interi di sizeblocco)

__host__ void allocaEInizializzaMatrice(int **res,int m,int n){
	*res=new int[m*n];
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			*((*res)+i*n+j)=1+rand()%10;
}

__host__ void stampaMatrice(int *a,int m,int n){
	cout<<"--------------------------------------"<<endl;
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++)
			cout<<a[i*n+j]<<" ";
		cout<<endl;
	}
}

__host__ void matMatCPU(int *a,int *b,int *res,int l,int m,int n){
	for(int i=0;i<l;i++)
		for(int j=0;j<n;j++){
			int v=0;
			for(int k=0;k<m;k++)
				v+=a[i*m+k] * b[k*n+j];
			res[i*n+j]=v;
		}
}


__global__ 
void matMatGPUv3(int *a,int *b,int *c,int l,int m,int n){
	//blockDim.x e blockDim.y sono uguali tra loro per modello !
	__shared__ int buffa[T][T], buffb[T][T];
	
	int globx=blockIdx.x * blockDim.x + threadIdx.x;
	int globy=blockIdx.y * blockDim.y + threadIdx.y;
	int astart=blockIdx.x * T * m;
	int bstart=blockIdx.y * T;
	int cumsum=0;
	
	for(int as = astart, bs= bstart; as<=astart+m-1; as+= blockDim.y, bs+=blockDim.y*n){
		
		buffa[threadIdx.x][threadIdx.y] = a[as+ threadIdx.x * m + threadIdx.y];
		buffb[threadIdx.x][threadIdx.y] = b[bs+threadIdx.x * n +threadIdx.y];
		__syncthreads();
		for(int i=0;i<blockDim.x;i++)
			cumsum+=buffa[threadIdx.x][i]*buffb[i][threadIdx.y];
		__syncthreads();
	}
	c[globx * n + globy] = cumsum;
	
}


int main(int argc,char *argv[]){

int l,m,n;
dim3 sizeGriglia,sizeBlocco;
	
if(argc!=5){
		l=16;
		m=12;
		n=8;
		sizeBlocco.x = T;
		sizeBlocco.y = T;
}
else{
	sscanf(argv[1],"%d",&l);
	sscanf(argv[2],"%d",&m);
	sscanf(argv[3],"%d",&n);
	sscanf(argv[4],"%d",&sizeBlocco.x);
	sizeBlocco.y = T;
	sizeBlocco.y = sizeBlocco.x;
}

sizeGriglia.x = l / sizeBlocco.x;
sizeGriglia.y = n / sizeBlocco.y;

int *ha,*hb,*hc;
allocaEInizializzaMatrice(&ha,l,m);
allocaEInizializzaMatrice(&hb,m,n);
stampaMatrice(ha,l,m);
stampaMatrice(hb,m,n);
hc=new int[l*n];
matMatCPU(ha,hb,hc,l,m,n);
stampaMatrice(hc,l,n);


int *da,*db,*dc;
cudaMalloc(&da,l*m*sizeof(int));
cudaMalloc(&db,m*n*sizeof(int));
cudaMalloc(&dc,l*n*sizeof(int));
cudaMemset(dc,0,l*n*sizeof(int));
cudaMemcpy(da,ha,l*m*sizeof(int),cudaMemcpyHostToDevice);
cudaMemcpy(db,hb,m*n*sizeof(int),cudaMemcpyHostToDevice);
matMatGPUv3<<<sizeGriglia,sizeBlocco,sizeBlocco.x>>>(da,db,dc,l,m,n);
int *copy=new int[l*n];
cudaMemcpy(copy,dc,l*n*sizeof(int),cudaMemcpyDeviceToHost);
stampaMatrice(copy,l,n);




}