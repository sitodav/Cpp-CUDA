#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
using namespace std;


__host__
void inizializzaArray(int *a,int n){
	srand((unsigned int)time(NULL));
	for(int i=0;i<n;i++)
		a[i]=1+rand()%10;
}

__host__
void calcolaProdEsterno(int *a,int *b,int m,int n,int *c){
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			c[i*n+j] = a[i]*b[j];
}

__host__
void stampaMatrice(int *a,int m,int n){
	cout<<"-----------"<<endl;
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++)
			cout<<a[i*n+j]<<" ";
		cout<<endl;
	}
}

__global__
void calcolaProdEsternoGPU (int *a,int *b,int *c,int m,int n,int colEffettive){
	int indRigaThreadGriglia = threadIdx.x + blockDim.x * blockIdx.x;
	int indColThreadGriglia = threadIdx.y + blockDim.y * blockIdx.y;
	
	if(indRigaThreadGriglia>= m || indColThreadGriglia >=n)
		return;
	//ATTENZIONE QUI, SOLO PER LA MATRICE C'E' STATA ALLOCAZIONE PADDED, NON ANCHE PER I VETTORI
	//quindi...
	c[indRigaThreadGriglia * colEffettive + indColThreadGriglia] = a[indRigaThreadGriglia] * b[indColThreadGriglia];
	
}

//programma per il calcolo di prodotto esterno tra 2 vettori
//input : m primo vettore colonna, n vettore riga, dimensioni blocco griglia (n righe threads e n colonne threads)
int main(int argc,char *argv[]){

int n,m;
dim3 dimBlocco; //def constructor

if(argc!=5){
		n=10;
		m=5;
		dimBlocco.x = 4;
		dimBlocco.y = 3;
}	
else {
	sscanf(argv[1],"%d",&m);
	sscanf(argv[2],"%d",&n);
	sscanf(argv[3],"%d",&dimBlocco.x);
	sscanf(argv[4],"%d",&dimBlocco.y);
}

dim3 dimGriglia;
dimGriglia.x = ((m % dimBlocco.x ==0) ? 0 : 1) + m / dimBlocco.x;
dimGriglia.y = ((n%dimBlocco.y == 0) ? 0 : 1) + n / dimBlocco.y;

//strutture dati in memoria heap host
int *h_a,*h_b, *h_c;
h_a= (int *)malloc(m*sizeof(int));
h_b = (int *)malloc(n*sizeof(int));
h_c = (int *)malloc ( n * m * sizeof(int));
inizializzaArray(h_a,m);
inizializzaArray(h_b,n);
calcolaProdEsterno(h_a,h_b,m,n,h_c);
stampaMatrice(h_a,m,1);
stampaMatrice(h_b,1,n);
stampaMatrice(h_c,m,n);	


//strutture dati in memoria gpu
int *d_a,*d_b,*d_c;
size_t pitch;

cudaMalloc(&d_a,m*sizeof(int));
cudaMalloc(&d_b,n*sizeof(int));
cudaMallocPitch(&d_c,&pitch,n*sizeof(int),m);
//copio contenuto array da host a gpu
cudaMemcpy (d_a,h_a,m*sizeof(int),cudaMemcpyHostToDevice) ;
cudaMemcpy(d_b,h_b,n*sizeof(int),cudaMemcpyHostToDevice);
//lancio kernel
calcolaProdEsternoGPU<<<dimGriglia,dimBlocco>>>(d_a,d_b,d_c,m,n,pitch/sizeof(int));
//ricopio su memoria host
int *copia=(int *)malloc(n*m*sizeof(int));
cudaMemcpy2D(copia,n*sizeof(int),d_c,pitch,n*sizeof(int),m,cudaMemcpyDeviceToHost);
//stampo
stampaMatrice(copia,m,n);	
	
}