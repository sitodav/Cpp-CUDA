//programma per calcolo prodotto puntuale tra 2 matrici
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
using namespace std;

//input: dimensioni M,N matrici (quindi questo darà il numero totale di thread nella griglia cuda) , n righe threads in blocco, n colonne threads in blocco
//NB: in cuda x indica la direzione righe, y le colonne


__host__ void calcolaProdPuntualeCPU(int *a,int *b,int *c,int m,int n){
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++){
			c[i*n + j]=a[i*n + j] * b[i*n + j];
		}
}

__host__ void stampaMatrice(int *a,int m,int n){
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++)
			cout<<a[i*n + j]<<" ";
		cout<<endl;
	}
}

__global__ void calcolaProdPuntualeGPU(int *a,int *b,int *c,int m,int n){
	//ottengo indice del thread SULL'INTERA GRIGLIA
	//prima riga poi colonna
	int indRiga = threadIdx.x + blockIdx.x * blockDim.x;
	int indCol = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(indRiga<m && indCol<n)
		c[indRiga * n + indCol] = a[indRiga * n + indCol]*b[indRiga * n + indCol];
}


void inizializzaMatrice(int *a,int m,int n){
	srand((unsigned int)time(NULL));
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			a[i*n + j]=1+rand()%10;
		
	
}

int main(int argc,char *argv[]){
	
	int m,n,totRigheThreadsInGriglia,totColonneThreadsInGriglia,nRigheThreadsInBlocco,nColonneThreadsInBlocco,nRigheBlocchiInGriglia,nColonneBlocchiInGriglia;
	dim3 nBlocchiInGriglia_d3, nThreadsInBlocco_d3; //partono i default constructors
	
	
	if(argc!=5){
		m=5;
		n=5;
		nRigheThreadsInBlocco=2;
		nColonneThreadsInBlocco=2;
	}
	else{
		sscanf(argv[1],"%d",&m); //n righe matrice
		sscanf(argv[2],"%d",&n); //n colonne matrice
		sscanf(argv[3],"%d",&nRigheThreadsInBlocco); //n righe di un blocco (righe di threads)
		sscanf(argv[4],"%d",&nColonneThreadsInBlocco); //n colonne di un blocco (colonne di threads)
	}
	
	//si noti che il totale di righe di threads e di colonne di threads nella griglia è uguale proprio al numero di righe matrice e di colonne (per scelta di distribuzione 1 cella = 1 thread)
	totRigheThreadsInGriglia = m;
	totColonneThreadsInGriglia = n;
	//calcolo numero di blocchi (quindi il numero di righe di blocchi e di colonne di blocchi che costituiscono la nostra griglia cuda)
	nRigheBlocchiInGriglia = totRigheThreadsInGriglia / nRigheThreadsInBlocco;
	if(totRigheThreadsInGriglia % nRigheThreadsInBlocco != 0)
		nRigheBlocchiInGriglia++;
	nColonneBlocchiInGriglia = totColonneThreadsInGriglia / nColonneThreadsInBlocco;
	if(totColonneThreadsInGriglia % nColonneThreadsInBlocco != 0)
		nColonneBlocchiInGriglia++;
	
	nBlocchiInGriglia_d3.x = nRigheBlocchiInGriglia; //x è per le righe in cuda
	nBlocchiInGriglia_d3.y = nColonneBlocchiInGriglia; //y è per le colonne
	
	nThreadsInBlocco_d3.x = nRigheThreadsInBlocco;
	nThreadsInBlocco_d3.y = nColonneThreadsInBlocco;
	
	
	int *h_a,*h_b,*h_c; //le matrici 2d, siccome i size sono decisi a runtime, e siccome per la mappatura ci serve che siano allocate sequenzialmente, sono allocate dinamicamente (heap host)
	//come vettori monodimensionali che simulano l'allocazione per righe della struttura 2d
	int *d_a,*d_b,*d_c;
	
	
	//alloco le strutture sull'heap dell'host
	h_a=(int *)malloc(m*n*sizeof(int));
	h_b=(int *)malloc(m*n*sizeof(int));
	h_c=(int *)malloc(m*n*sizeof(int));
	
	inizializzaMatrice(h_a,m,n);
	inizializzaMatrice(h_b,m,n);
	
	calcolaProdPuntualeCPU(h_a,h_b,h_c,m,n);
	stampaMatrice(h_c,m,n);
	
	//alloco le strutture in memoria gpu
	cudaMalloc((void **)&d_a,m*n*sizeof(int));
	cudaMalloc((void **)&d_b,m*n*sizeof(int));
	cudaMalloc((void **)&d_c,m*n*sizeof(int));
	cudaMemset(d_c,0,m*n*sizeof(int)); //facoltativo
	
	//copio sulla memoria gpu il contenuto delle matrici in memoria ordinaria 
	cudaMemcpy(d_a,h_a,m*n*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,m*n*sizeof(int),cudaMemcpyHostToDevice);
	//lancio kernel
	calcolaProdPuntualeGPU<<< nBlocchiInGriglia_d3, nThreadsInBlocco_d3 >>>(d_a,d_b,d_c,m,n);
	//ricopio il risultato nella memoria host
	int *copiedFromGPU = (int *)malloc(n*m*sizeof(int));
	cudaMemcpy(copiedFromGPU,d_c,n*m*sizeof(int),cudaMemcpyDeviceToHost);
	//stampo
	cout<<"---------------------------------------------"<<endl;
	stampaMatrice(copiedFromGPU,m,n);
}