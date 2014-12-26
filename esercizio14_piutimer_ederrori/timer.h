#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

class mioTimer{
	
	
	public:
		mioTimer(){
			tempoTrascorso = 0.0f;
			isStarted=false;
			cudaEventCreate(&startv);
			cudaEventCreate(&stopv);
		}
		~mioTimer(){
			cudaEventDestroy(&startv);
			cudaEventDestroy(&stopv);
		}
		
		void restart(){
			if(!isStarted)
				return;
			tempoTrascorso=0.0f;
			cudaEventDestroy(&startv);
			cudaEventDestroy(&stopv);
			cudaEventCreate(&startv);
			cudaEventCreate(&stopv);
			start();
		}
		
		void start(){
			if(isStarted)
				return;
			isStarted=true;
			cudaEventRecord(startv,0);
		}
		
		void stop(){
			if(!isStarted)
				return;
			isStarted=false;
			cudaEventRecord(stopv,0);
			cudaEventSynchronize(stopv);
			cudaEventElapsedTime(&tempoTrascorso,startv,stopv);
		}
		
		float getTempoTrascorso(){
			return tempoTrascorso;
		}
		
		private:
		cudaEvent_t startv;
		cudaEvent_t stopv;
		float tempoTrascorso;
		bool isStarted;
};