#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <time.h>
using namespace std;

//prodotto puntuale tra 2 matrici, usando il padding di memoria (pitch)

//input size matrici (m,n) , dimensioni blocchi (righe di thread, colonne di thread)

__host__ 
void inizializzaCPU(int *a,int m,int n){
	srand((unsigned int)time(NULL));
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			a[i*n+j]=1+rand()%10;
}

__host__
void stampaCPU(int *a,int m,int n){
	cout<<"--------------------------------"<<endl;
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++)
			cout<<a[i*n+j]<<" ";
		cout<<endl;
	}
}

__global__
void calcolaProdPuntuale(int *a,int *b,int *c,int m,int n,int pitch){
	//attenzione a come si accede alle matrici allocate in memoria gpu perchè c'e' il pitch
	int nColonneEffettive = pitch/sizeof(int); //perchè il pitch è la lunghezza effettiva delle righe matrici allocate in memoria gpu, ma in bytes
	
	int iRiga = threadIdx.x + blockIdx.x * blockDim.x; //indice riga del thread GLOBALMENTE alla griglia
	int iCol = threadIdx.y + blockIdx.y * blockDim.y ; //indice colonna del thread GLOBALMENTE alla griglia
	
	
	if(iRiga >= m || iCol >=n)
		return;
	c[iRiga * nColonneEffettive + iCol] = a[iRiga * nColonneEffettive + iCol] * b[iRiga * nColonneEffettive + iCol];
}


int main(int argc,char *argv[]){
	
	
	int m,n;
	dim3 dimBlocco; //default constructor, qui salviamo il numero di righe e di colonne (di thread) in un blocco
	dim3 dimGriglia; //qui salviamo il numero di righe e di colonne (di blocchi) della griglia
	
	if(argc!=5){
		m=5;
		n=5;
		dimBlocco.x=2; //2 righe di thread in un blocco
		dimBlocco.y=2; //2 colonne di thread in un blocco
		//quindi 4 thread in totale per un blocco
	}
	else{
		sscanf(argv[1],"%d",&m);
		sscanf(argv[2],"%d",&n);
		sscanf(argv[3],"%d",& dimBlocco.x); //n righe di thread in blocco
		sscanf(argv[4],"%d",& dimBlocco.y); //n di colonne di thread in blocco
	}
	
	dimGriglia.x = m / dimBlocco.x;
	if(m % dimBlocco.x != 0)
		dimBlocco.x ++;
	dimGriglia.y = n / dimBlocco.y;
	if(n % dimBlocco.y != 0)
		dimBlocco.y ++;
	
	
	//strutture dati su host
	int *h_a,*h_b,*h_c;
	//alloco
	h_a=(int *)malloc(n*m*sizeof(int));
	h_b=(int *)malloc(n*m*sizeof(int));
	h_c=(int *)malloc(n*m*sizeof(int));
	//inizializzo le matrici
	inizializzaCPU(h_a,m,n);
	inizializzaCPU(h_b,m,n);
	stampaCPU(h_a,m,n);
	stampaCPU(h_b,m,n);
	
	
	//per le strutture dati sulla memoria gpu
	int *d_a,*d_b,*d_c;
	size_t pitch; //qui verrà salvato dalla cudaMallocPitch la lunghezza effettiva (paddata) in bytes 
	//ATTENZIONE CHE IL PITCH DEVE ESSERE UNSIGNED LONG
	//alloco memoria sull'heap gpu usando il pitch (padding)
	cudaMallocPitch(&d_a,&pitch,n*sizeof(int),m);
	cudaMallocPitch((void **)&d_b,&pitch,n*sizeof(int),m);
	cudaMallocPitch((void **)&d_c,&pitch,n*sizeof(int),m);
	
	//ora in pitch c'e' la lunghezza effettiva (in bytes) delle righe 
	//ora devo copiare il contenuto delle matrici dalla memoria host a memoria device
	//pero' la memoria device è paddata (pitch) quindi devo usare la cudaMemcpy2D per evitare
	//di inserire dati in quelle che sono le celle di padding, quindi...
	cudaMemcpy2D(d_a,pitch,h_a,n*sizeof(int),n*sizeof(int),m,cudaMemcpyHostToDevice);
	//i parametri sono : destinazione, lunghezza effettiva in bytes delle righe in memoria device (quindi quest'informazione sta in pitch)
	//origine, lunghezza in bytes delle righe nella memoria host (potrebbe esserci il pitch anche qui)
	//numero di bytes da copiare nella direzione orizzontale (lunghezza bytes riga)
	//numero di righe da copiare
	//flag destinazione
	cudaMemcpy2D(d_b,pitch,h_b,n*sizeof(int),n*sizeof(int),m,cudaMemcpyHostToDevice);
	
	
	//lancio il kernel
	calcolaProdPuntuale<<<dimGriglia, dimBlocco>>>(d_a,d_b,d_c,m,n,pitch);
	//ricopio dalla memoria device alla memoria host
	int *copyFromGPU=(int *)malloc(n*m*sizeof(int));
	//usando sempre la cudaMemcpy2D perchè nell'origine ci sono celle di padding
	cudaMemcpy2D(copyFromGPU,n*sizeof(int),d_c,pitch,n*sizeof(int),m,cudaMemcpyDeviceToHost);
	//stampo
	stampaCPU(copyFromGPU,m,n);
	
	
	//libero memoria
	free(h_a);
	free(h_b);
	free(h_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	
}