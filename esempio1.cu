
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

using namespace std;

void inizializzazioneHostArray(int *h_array,int n){
        int i;
        for(i=0;i<n;i++)
                *(h_array+i)=i;
}

int main(int argc,char *argv[]){

int n;
int *h_array, *d_array;

if(argc==1)
                n=20;
else sscanf(argv[1],"%d",&n);


//alloco host array
h_array=(int *)malloc(n*sizeof( int ));
//alloco device array
cudaMalloc((void **)&d_array,n*sizeof(int));

//inizializzo host array
inizializzazioneHostArray(h_array,n);

//trasferisco i dati da host a device
cudaMemcpy(d_array,h_array,n*sizeof(int),cudaMemcpyHostToDevice);


//alloco array per ricevere di nuovo
int *h_array2=(int *)malloc(n*sizeof(int));
cudaMemcpy(h_array,d_array,n*sizeof(int),cudaMemcpyDeviceToHost);

//stampa
int i;
for(i=0;i<n;i++)
{
        printf("%d - %d",h_array[i],h_array[i]);
        printf("\n");
}



//dealloco memoria dell'heap sull'host
free((void *)h_array);
free((void *)h_array2);
//dealloco memoria heap della ram scheda video
}