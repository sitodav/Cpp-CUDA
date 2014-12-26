#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
//ATTENZIONE:
//NB: l'algoritmo qui usato (ed esposto dalla prof) nel caso in cui si voglia usare SHARED MEMORY NON funziona nel caso in cui il raggio sia minore del numero di thread MINIMO che lavorano in un blocco
//allora l'algoritmo non funziona poichè lavorano 2 threads ma bisogna riempire nel lato sinistra 3 pads
//allo stesso modo anche se diamo un numero di thread e blocchi tale che in tutti i blocchi tutti i thread lavorano, ma questo numero di blocchi è < raggio, allo stesso modo non funziona !!




//programma per il calcolo dello stencil di un vettore monodimensionale, come da slide
//senza l'uso dello shared memory e con l'uso di shared memory
//input: n size vettore, t thread per blocco (size blocco) k raggio stencil


using namespace std; //ad esempio se si fissa il numero threads per blocco, e questo ci porta ad avere il blocco finale con in totale 2 thread che lavorano, ed il raggio è 3 ad esempio
__host__ void stencilCPU(int *a,int *b,int n,int k){ 
	for(int i=0;i<n;i++){ 
	int v=0;
		for(int j=-k;j<=k;j++)
			if(i+j>=0 && i+j<n)
				v+=a[i+j];
		b[i]=v;
	}
}

__host__ void stampaArray(int *a,int n){
	cout<<"-------------------------"<<endl;
	for(int i=0;i<n;i++)
		cout<<a[i]<<" ";
	cout<<endl;
}

__host__ void inizializzaArray(int *a,int n){
	for(int i=0;i<n;i++)
		a[i]=1+rand()%10;
}

__global__ void stencilGPU(int *a,int *b,int n,int k){
	int iGlob = threadIdx.x + blockIdx.x * blockDim.x;
	if(iGlob>= n)
		return;
	b[iGlob]=0;
	for(int j=-k;j<=k;j++)
		if(iGlob+j>=0 && iGlob+j<n)
			b[iGlob]+=a[iGlob+j];
	
}

__global__ void stencilGPU_CONSHAREDMEMORY (int *a,int *b,int n,int k){
	extern __shared__ int sh_buffer[]; //questa sintassi prevede che il size di allocazione (dinamica, quindi in heap) in memoria shared, per quest'array venga passato
									//come terzo parametro in <<< >>> nella chiamata al kernel. Questo perchè abbiamo messo extern. Se non lo avessimo 
									//messo avremmo potuto allocare solo staticamente la variabile
	//il size di sh_buffer sarà uguale al numero di thread in un blocco + 2* raggio (k) dello stencil
	
	int globId = threadIdx.x + blockIdx.x * blockDim.x; //indice del thread sulla griglia (quindi indice dell'elemento del vettore a su cui il thread ha lo stencil "centrato")
	int iPadd = threadIdx.x + k; //indice del corrispondente elemento dell'array sh_buffer (che è paddato, cioè ha blockDim.x elementi + k a sinistra e k a destra)

	/* if(globId >= n) 
		return ; */ 
	
	//altrimenti passo a:
		//copio in sh_buffer l'elemento centrale di stencil

	sh_buffer[iPadd] = a[globId];
	//ora solo i primi k thread del blocco, corrispondenti ai primi k elementi dell'array relativi ai thread del blocco
	//devono copiare i primi k elementi relativi alle zone di padding
	
	if(threadIdx.x < k){
		//attenzione va fatto il controllo poichè potremmo trovarci all'estremo sinistro o destro dell'array, quindi le zone di padding di sh_buffer
		//vanno riempite con zeri poichè non hanno corrispettivi nell'array a
		if(globId-k >= 0)
			sh_buffer[iPadd-k] = a[globId-k];
		else sh_buffer[iPadd-k] = 0;
	
		if(globId + blockDim.x <n)
			sh_buffer[iPadd + blockDim.x] = a[globId + blockDim.x];
		else sh_buffer[iPadd + blockDim.x] = 0;
	}
	//sincronizzo i thread (barriera)
	__syncthreads();
	//ora posso effettivamente calcolare la somma 
	int v=0;
	for(int j=-k;j<=k;j++)
		v+=sh_buffer[iPadd+j];
	
	b[globId]= v;
		
}

int main(int argc,char *argv[]){
	srand((unsigned int)time(NULL));
	int n;
	dim3 sizeGriglia;
	dim3 sizeBlocco;
	int k; //raggio stencil
	
	if(argc!=4){
		n=10;
		sizeBlocco.x=4; //4 threads per blocco (BLOCCHI MONODIMENSIONALI)
		k = 3;
	}
	else{
		sscanf(argv[1],"%d",&n);
		sscanf(argv[2],"%d",& sizeBlocco.x);
		sscanf(argv[3],"%d",& k);
	}
	
	sizeGriglia.x = n / sizeBlocco.x ;
	if(n % sizeBlocco.x != 0)
		sizeGriglia.x ++;
	
	
	int *h_a = (int *)malloc(n*sizeof(int));
	inizializzaArray(h_a,n);
	stampaArray(h_a,n);
	int *risCPU= (int *)malloc(n*sizeof(int));
	stencilCPU(h_a,risCPU,n,k);
	stampaArray(risCPU,n);
	
	//in memoria globale device
	int *d_a,*d_b;
	cudaMalloc(&d_a,n*sizeof(int));
	cudaMalloc(&d_b,n*sizeof(int));
	cudaMemcpy(d_a,h_a,n*sizeof(int),cudaMemcpyHostToDevice);
	//lancio kernel
	stencilGPU<<<sizeGriglia,sizeBlocco>>>(d_a,d_b,n,k);
	//ricopio su host
	int *ris = (int *)malloc(n*sizeof(int));
	cudaMemcpy(ris,d_b,n*sizeof(int),cudaMemcpyDeviceToHost);
	stampaArray(ris,n);
	//struttura dati per secondo kernel (usando shared memory questa volta)
	int *d_b2;
	cudaMalloc(&d_b2,n*sizeof(int));
	//chiamo kernel
	//definisco qui il size del buffer contenuto in memoria shared per ogni blocco della griglia
	int sizeBufferShared = (sizeBlocco.x + 2 * k ) * sizeof(int);
	 
	stencilGPU_CONSHAREDMEMORY<<<sizeGriglia,sizeBlocco,sizeBufferShared>>>(d_a,d_b2,n,k);
	cudaThreadSynchronize();
	//ricopio i risultati
	int *ris2= (int *)malloc(n*sizeof(int));
	cudaMemcpy(ris2,d_b2,n*sizeof(int),cudaMemcpyDeviceToHost);
	stampaArray(ris2,n);
	
	
	
}