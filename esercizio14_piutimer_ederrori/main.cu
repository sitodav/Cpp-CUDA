#include "timer.h"

//esercizio calcolo prod componente per componente matrice vettore
//input : m,n size matrice, dimensioni blocco griglia

__host__
void checkErrore(string msg){
	cudaError_t errore=cudaGetLastError();
	if(errore!=cudaSuccess){
		cout<<msg<<": "<<cudaGetErrorString(errore)<<endl;
		exit(1);
	}
}


__host__
void inizializzaArray(int *a,int n){
	
	for(int i=0;i<n;i++)
		a[i]=1+rand()%10;
}

__host__
void calcolaScalatura(int *a,int *b,int m,int n,int *c){
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			c[i*n+j] = a[i*n+j] * b[j]; 
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
void calcolaScalaturaGPU(int *a,int *b,int *c,int m,int n,int colEffettiveA,int colEffettiveB){
	int iThread,jThread;
	iThread = blockDim.x * blockIdx.x + threadIdx.x;
	jThread = blockDim.y * blockIdx.y + threadIdx.y;
	if(iThread< m && jThread<n)
		c[iThread * colEffettiveB + jThread] = a[iThread * colEffettiveA + jThread] * b[jThread];

}

int main(int argc,char *argv[]){
srand((unsigned int)time(NULL));
dim3 dimBlocco,dimGriglia;
int m,n;
mioTimer timer1; //def constructor

if(argc!=5){
	m=10;
	n=5;
	dimBlocco.x = 3;
	dimBlocco.y = 4;
}
else{
	sscanf(argv[1],"%d",&m);
	sscanf(argv[2],"%d",&n);
	sscanf(argv[3],"%d",& dimBlocco.x);
	sscanf(argv[4],"%d",& dimBlocco.y);
}

dimGriglia.x = (m/dimBlocco.x) + ( (m % dimBlocco.x == 0) ? 0 : 1 );
dimGriglia.y = (n / dimBlocco.y) + ( (n % dimBlocco.y == 0) ? 0 : 1 );

//strutture dati memoria host
int *h_a,*h_b, *h_c;
//alloco
h_a=(int *)malloc(m*n*sizeof(int));
h_b=(int *)malloc(n*sizeof(int));
h_c=(int *)malloc(m*n*sizeof(int));

//uso routine per inizializzare array, per inizializzare matrice
for(int i=0;i<m;i++)
	inizializzaArray(h_a + i*n,n);
//inizializzo l'array vero e proprio
inizializzaArray(h_b,n);
//stampo
stampaMatrice(h_a,m,n);
stampaMatrice(h_b,1,n);
calcolaScalatura(h_a,h_b,m,n,h_c);
stampaMatrice(h_c,m,n);


//sul device
//le strutture dati
int *d_a,*d_b,*d_c;
size_t pitcha,pitchb;
//alloco
cudaMallocPitch(&d_a,&pitcha,n*sizeof(int),m);
cudaMalloc(&d_b,n*sizeof(int));
cudaMallocPitch(&d_c,&pitchb,n*sizeof(int),m);
checkErrore("serie di cudaMalloc e cudaMallocPitch");
//in realtà pitcha e pitchb saranno per forza uguali visto che h_a e h_c sono matrici contenenti stesso tipo di elementi e stesso size
//copio il contenuto della matrice e del vettore da moltiplicare in memoria device
cudaMemcpy2D(d_a,pitcha,h_a,n*sizeof(int),n*sizeof(int),m,cudaMemcpyHostToDevice);
cudaMemcpy(d_b,h_b,n*sizeof(int),cudaMemcpyHostToDevice);
checkErrore("serie di cudaMemcpy e cudaMemcpy2d");
//lancio kernel (dopo aver creato il mio oggetto timer e averlo startato, che usa le primite di cuda per calcolo tempi)

timer1.start();


calcolaScalaturaGPU<<<dimGriglia,dimBlocco>>>(d_a,d_b,d_c,m,n,pitcha/sizeof(int),pitchb/sizeof(int));

timer1.stop();

checkErrore("kernel calcolo scalatura GPU");
//copio su memoria host
int *copia = (int *)malloc(n*m*sizeof(int));
cudaMemcpy2D(copia,n*sizeof(int),d_c,pitchb,n*sizeof(int),m,cudaMemcpyDeviceToHost);



//stampo
stampaMatrice(copia,m,n);
cout<<"tempo impiegato: "<<timer1.getTempoTrascorso()<<endl;

free(h_a);
free(h_b);
free(h_c);
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
	
}
 

