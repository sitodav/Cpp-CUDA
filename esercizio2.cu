#include <cuda.h>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;                                                                                             

//programma che svolge il prodotto puntuale tra 2 array//in input al programma si passa n e nThreadPerBlocco

void inizializzaArrayCPU(int *a,int n){
        for(int i=0;i<n;i++)
                *(a+i)=1+rand()%10;

}

void calcolaProdottoPuntualeOnCPU(int *a,int *b,int *ris,int n){
        for(int i=0;i<n;i++)
                ris[i]=a[i]*b[i];
}

void stampaArrayOnCPU(int a[],int n){
        cout<<endl;
        for(int i=0;i<n;i++)
                cout<<a[i]<<"-";
        cout<<endl;
}

__global__ void calcolaProdottoPuntualeOnGPU(int *a,int *b,int *ris,int n){

//vedi slide del perchè si accede in questo modo..
//l'indice del thread che è conservato nella variable threadId.x èlocale al blocco, mentre a me serve globale rispetto alla griglia, per capire come mappare il thread con la porzione
//di array , quindi...

        int indiceThreadSuGriglia = threadIdx.x + blockIdx.x * blockDim.x;
        //indiceThreadSuGriglia identifica l'indice del'array su cui deve lavorare il thread,
        //pero' siccome puo' capitare che alcuni thread nell'ultimo blocco non debbano lavorare (questo capita se il totale dei thread richiesti per la griglia non è interamente divisibile
        //per il numero di thread per blocchi, quindi ci sara l'ultimo blocco che ha dei thread che non devono lavorare
        if(indiceThreadSuGriglia < n )
                ris[indiceThreadSuGriglia] = a[indiceThreadSuGriglia] * b[indiceThreadSuGriglia];

}

int main(int argc,char *argv[]){

        srand((unsigned int)time(NULL));
        int n; //questo si passa in input,il numero di elementi degli array, e creiamo un thread per ogni elemento di array (n thread in totale per 2 array di n elementi)
        int nThreadPerBlocco; //questo si passa in input

        if(argc!=3){
                n=100;
                nThreadPerBlocco=10;
        }

        else    {
                sscanf(argv[1],"%d",&n);
                sscanf(argv[2],"%d",&nThreadPerBlocco);
        }

        //dichiarazione variabili
        int *h_arraya, *h_arrayb; //array nell'heap della macchina host, da moltiplicare puntualmente
        int *d_arraya, *d_arrayb; //gli array contenuti nell'heap della GPU, da moltiplicare puntualmente
                                                                                                                                                                                   9,0-1         13%

        int *h_risultato_ottenutoOnCPU; //qui salviamo il risultato della moltiplicazione puntuale, calcolato localmente (CPU)
        int *d_risultato_ottenutoOnGPU; //qui salviamo il risultato della moltiplicazione calcolata sul device (GPU)
        int *h_risultatoCopiatoFromGPU; //qui copiamo il contenuto del risultato ottenuto sulla gpu

        //in base al numero totale di thread scelto (n thread per array n dimensionali) e di thread per blocchi eseguo il calcolo del totale di blocchi
        dim3 *nThreadPerBlocco_d3 = new dim3(nThreadPerBlocco,1,1); //uso oggetto di tipo dim3 allocato dinamicamente, al costruttore per la dimensione in x (solo questa visto che stiamo lavorando con blocchi e griglia monodimensionale) passo quanto preso in input

        //calcolo il numero di blocchi
        int nBlocchi= n/nThreadPerBlocco;
        if((n % nThreadPerBlocco )!=0)
                nBlocchi++;

        dim3 *nBlocchi_d3 = new dim3(nBlocchi,1,1);



        //allocazione memoria heap cpu degli array da moltiplicare e inizializzazione e allocazione array risultato
        h_arraya=(int *)malloc(n*sizeof(int));
        h_arrayb=(int *)malloc(n*sizeof(int));
        h_risultato_ottenutoOnCPU=(int *)malloc(n*sizeof(int));
        h_risultatoCopiatoFromGPU=(int *)malloc(n*sizeof(int));
        inizializzaArrayCPU(h_arraya,n);
        inizializzaArrayCPU(h_arrayb,n);
        memset(h_risultato_ottenutoOnCPU,0,n*sizeof(int));



        //allocazione memoria heap gpu
        cudaMalloc((void **)&d_arraya,n*sizeof(int));
        cudaMalloc((void **)&d_arrayb,n*sizeof(int));
        cudaMalloc((void **)&d_risultato_ottenutoOnGPU,n*sizeof(int));
        //copia degli array da moltiplicare sull'heap gpu dall'heap cpu
        cudaMemcpy(d_arraya,h_arraya,n*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_arrayb,h_arrayb,n*sizeof(int),cudaMemcpyHostToDevice);
        //inizializzazione array risultato sulla gpu
        cudaMemset(d_risultato_ottenutoOnGPU,0,n*sizeof(int));


        //stampo array locali CPU, calcolo risultato prod puntuale su CPU e stampo
        stampaArrayOnCPU(h_arraya,n);
        stampaArrayOnCPU(h_arrayb,n);
        calcolaProdottoPuntualeOnCPU(h_arraya,h_arrayb,h_risultato_ottenutoOnCPU,n);
        stampaArrayOnCPU(h_risultato_ottenutoOnCPU,n);


        //calcolo il prodotto puntuale su GPU
        calcolaProdottoPuntualeOnGPU<<<*nBlocchi_d3,*nThreadPerBlocco_d3>>>(d_arraya,d_arrayb,d_risultato_ottenutoOnGPU,n);
        //copio il risultato in memoria host
        cudaMemcpy(h_risultatoCopiatoFromGPU,d_risultato_ottenutoOnGPU,n*sizeof(int),cudaMemcpyDeviceToHost);
        //stampo
        stampaArrayOnCPU(h_risultatoCopiatoFromGPU,n);
}
                                                                                                                                                                                   113,1         Bot
