#include <cuda.h>
#include <iostream>
#include <stdio.h>


using namespace std;


void calcolaOnCPU(int *a,int *b,int *ris,int n){
	for(int i=0;i<n;i++)
		ris[i]=a[i]*b[i];
}

//kernel
__global__ void calcolaOnGPU (int *a,int *b,int *ris,int n){
	//calcolo indice globale del thread nella griglia, usando indice del blocco nella griglia, numero di 
	//thread nel blocco e indice locale del thread nel blocco
	int indiceThreadInGriglia = blockIdx.x * blockDim.x + threadIdx.x;
	//indice globale thread in griglia = indice del blocco in griglia * numero di thread in ciascun blocco + indice locale thread nel blocco
	if(indiceThreadInGriglia < n)
		ris[indiceThreadInGriglia]=a[indiceThreadInGriglia] * b[indiceThreadInGriglia];

}



//questa funzioen prende come ultimi 2 parametri 2 alias (c++) di oggetti dim3, che se non vengono passati
//quindi in ingresso la funzione riceverà parametri di default, vuol dire che il calcolo del prodotto dei 2 array va fatto
//sulla cpu (e quindi i primi parametri ricevuti devono essere nella memoria heap della macchina host)
//se invece sono passati valori diversi da quelli di default allora il calcolo deve essere fatto sulla gpu e quindi in tal caso
//viene richiamato il kernel usando i 2 oggetti dim3 per configurare la griglia
void wrapperCalcolaProdottoPuntuale(int *a,int *b,int *ris,int n,const dim3 & nBlocchi=dim3(0,0,0),const dim3 &nThreads=dim3(0,0,0)){
	
	if(nBlocchi.x==0) //allora abbiamo ricevuto i parametri di default
		calcolaOnCPU(a,b,ris,n);
	else //chiamo il kernel
		calcolaOnGPU<<<nBlocchi,nThreads>>>(a,b,ris,n);
}


int main(int argc,char *argv[]){
	
	srand((unsigned int)time(NULL));
	int nThread;
	int sizeArray;
	int nBlocchi;
	int nThreadPerBlocco;
	dim3 nBlocchi_d3;
	dim3 nThreadPerBlocco_d3;
	
	if(argc!=3){
		sizeArray=200;
		nThread=sizeArray;
		nThreadPerBlocco=5;
	}
	else{
		sscanf(argv[1],"%d",&sizeArray);
		nThread=sizeArray;
		sscanf(argv[2],"%d",&nThreadPerBlocco);
	}
	
	nBlocchi= nThread / nThreadPerBlocco;
	if(nThread % nThreadPerBlocco != 0 )
		nBlocchi++ ;
	
	nBlocchi_d3.x=nBlocchi;
	nThreadPerBlocco_d3.x=nThreadPerBlocco;
	
	//alloco strutture sull'host e le inizializzo
	int *h_a = (int *)malloc(sizeArray * sizeof(int));
	int *h_b = (int *)malloc(sizeArray * sizeof(int));
	int *h_ris = (int *)malloc(sizeArray * sizeof(int));
	
	for(int i=0;i<sizeArray;i++){
		h_a[i] = 1+rand()%(i+1);
		h_b[i] = 1+rand()%(i+1);
	}
	memset(h_ris,0,sizeArray * sizeof(int));
	
	
	//alloco strutture sul device e le inizializzo
	int *d_a,*d_b,*d_ris;
	cudaMalloc((void **)&d_a,sizeArray * sizeof(int));
	cudaMalloc((void **)&d_b,sizeArray * sizeof(int));
	cudaMalloc((void **)&d_ris,sizeArray * sizeof(int));
	cudaMemcpy(d_a,h_a,sizeArray*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,sizeArray*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemset(d_ris,0,sizeof(int)*sizeArray);
	
	
	//calcolo in locale e stampo risultato
	
	wrapperCalcolaProdottoPuntuale(h_a,h_b,h_ris,sizeArray);
	for(int i=0;i<sizeArray;i++)
		cout<<h_ris[i]<<"-";
	cout<<endl<<"-----------------------"<<endl;
	
	//calcolo su gpu
	wrapperCalcolaProdottoPuntuale(d_a,d_b,d_ris,sizeArray,nBlocchi_d3,nThreadPerBlocco_d3);
	//il risultato adesso sta in d_ris nella memoria heap della gpu (device), e quindi non posso accederci direttamente perchè il main sta sull'host
	//quindi me lo riporto qui sulla memoria heap a cui la cpu ha accesso
	int *risFromGPU = (int *)calloc(sizeArray,sizeof(int));
	cudaMemcpy(risFromGPU,d_ris,sizeArray * sizeof(int),cudaMemcpyDeviceToHost);
	
	//stampo il risultato
	for(int i=0;i<sizeArray;i++)
		cout<<risFromGPU[i]<<"-";
	
	
	
	return 0;
}





