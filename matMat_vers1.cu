#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>


//programma per il calcolo matrice matrice, usando il primo approccio delle slide ma modificato da me per non avere race condition
//si moltiplica una matrice a size lxm con b size mxn, ottenendo c size lxn
//un elemento, tra gli lxn, di c, di indice (i,j) è ottenuto come prod scalare tra la riga i-esima di a e la colonna j-esima di back
//quindi possiamo assegnare un blocco di thread a ciascuna degli elementi di c, e i thread di quel blocco si occuperanno di calcolare il prodotto scalare
//tra la iesima riga e jesima colonna, quindi ogni blocco effettuerà m prodotti e m-1 somme. Il blocco assegnato sarà quello con indice riga blocco i e indice col j
//a seconda di quanti sono i thread del blocco, e di quanto vale m, ci sarà un map 1 cella di a e b, 1 thread, o più celle di a e b, 1 thread.
//per evitare la race condition, i thread non aggiornano direttamente c(i,j) ma ciascun thread scrive in un suo elemento di un buffer shared memory
//e solo il thread di indice 0 nel blocco alla fine somma tutti gli elementi nello shared buffer, e trascrive la somma in c(i,j) 
using namespace std;

//input: l,m,n, (e quindi il totale di blocchi nella griglia sarà lxn), t (totale di thread per blocco)
//la griglia sarà 2d (in modo tale da mappare gli indici riga e colonna dei blocchi con le celle di c) mentre ciascun blocco sarà monodimensionale
//in quanto i thread di un blocco devono essere mappati su riga a-colonna b.
//t deve essere <=m e per semplicità m%t deve essere uguale a 0

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
__global__ void matMatGPU(int *a,int *b,int *c,int l,int m,int n){
	//ogni elemento della matrice risultato c viene assegnata ad un blocco
	extern __shared__ int buffer[]; //questo verrà allocato per il numero di thread in un blocco
	int stride = m / blockDim.x; // lo stride è la porzione della riga/colonna (assegnata al blocco) che viene assegnata a ciascun thread del blocco
	
	buffer[threadIdx.x] = 0;
	for(int i=0;i<stride;i++){
		if(threadIdx.x * stride + i < m)
			buffer[threadIdx.x]+= (a[blockIdx.x * m + (threadIdx.x * stride +i)] * b[(threadIdx.x*stride +i)*n + blockIdx.y ]);
		else break;
	}
	__syncthreads();
	
	if(threadIdx.x == 0){
			c[blockIdx.x * n + blockIdx.y] = 0;
			for(int i=0;i<blockDim.x;i++)
				c[blockIdx.x * n + blockIdx.y]+=buffer[i];
	}
	
}


int main(int argc,char *argv[]){
	srand((unsigned int)time(NULL));
	int l,m,n,t;
	dim3 sizeGriglia,sizeBlocco;
	
	if(argc!=5){
		l=10;
		m=5;
		n=4;
		t=5;
	}
	else {
		sscanf(argv[1],"%d",&l);
		sscanf(argv[2],"%d",&m);
		sscanf(argv[3],"%d",&n);
		sscanf(argv[4],"%d",&t);
	}
	if(m%t != 0 || t>m){
		cout<<"wrong usage\n";
		exit(1);
	}
	sizeGriglia.x = l;
	sizeGriglia.y = n; //ogni blocco della griglia un elemento della matrice risultato c (size lxn)
	sizeBlocco.x = t; //blocco monodimensionale
	
	int *ha,*hb,*hc;
	allocaEInizializzaMatrice(&ha,l,m);
	allocaEInizializzaMatrice(&hb,m,n);
	hc=new int[l*n];
	
	stampaMatrice(ha,l,m);
	stampaMatrice(hb,m,n);
	matMatCPU(ha,hb,hc,l,m,n);
	stampaMatrice(hc,l,n);
	
	int *da,*db,*dc;
	cudaMalloc(&da,l*m*sizeof(int));
	cudaMalloc(&db,m*n*sizeof(int));
	cudaMalloc(&dc,l*n*sizeof(int));
	cudaMemset(dc,0,l*n*sizeof(int));
	cudaMemcpy(da,ha,l*m*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(db,hb,m*n*sizeof(int),cudaMemcpyHostToDevice);
	
	matMatGPU<<<sizeGriglia,sizeBlocco,sizeof(int)*sizeBlocco.x >>>(da,db,dc,l,m,n);
	int *copia=new int[l*n];
	cudaMemcpy(copia,dc,l*n*sizeof(int),cudaMemcpyDeviceToHost);
	stampaMatrice(dc,l,n);
	
	
	return 0;
}


