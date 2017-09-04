/*
Detected 1 CUDA Capable device(s)

Device 0: "GeForce GT 320M"
  CUDA Driver Version / Runtime Version          5.0 / 5.0
  CUDA Capability Major/Minor version number:    1.2
  Total amount of global memory:                 1024 MBytes (1073741824 bytes)
  ( 3) Multiprocessors x (  8) CUDA Cores/MP:    24 CUDA Cores
  GPU Clock rate:                                1100 MHz (1.10 GHz)
  Memory Clock rate:                             790 Mhz
  Memory Bus Width:                              128-bit
  Max Texture Dimension Size (x,y,z)             1D=(8192), 2D=(65536,32768), 3D=(2048,2048,2048)
  Max Layered Texture Size (dim) x layers        1D=(8192) x 512, 2D=(8192,8192) x 512
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       16384 bytes
  Total number of registers available per block: 16384
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           512
  Maximum sizes of each dimension of a block:    512 x 512 x 64
  Maximum sizes of each dimension of a grid:     65535 x 65535 x 1
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             256 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
  Device supports Unified Addressing (UVA):      No
  Device PCI Bus ID / PCI location ID:           2 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 5.0, CUDA Runtime Version = 5.0, NumDevs = 1, Device0 = GeForce GT 320M
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "helper_functions.h"
//#include "helper_cuda.h"
#include <stdio.h>
#include "makeDat.h"

__global__ void freqencyStep1(char *d_dat,int len, int *d_freq)
{//����һ���Ƚ����ݼӺ͵�share memory�У�Ȼ�����ۼӵ��Դ��ϡ�
///����Ҳ�����ַ��������Ƿ���һ��share memory�����á���һ�ַ�������share memory�����ã��ڽ��п����ۼ�ʱ��ֻ��ǰ26���߳����0��127���ۼӡ���
///���������ۼ�ʱ������ۼӵ��Խ����ϣ�Ȼ����д��ʱ�����Ա���bank conflict��
    __shared__ int sfreq[3456];//27*128////share memory���ŷţ�ÿ�߳�27��int.

    for(int i=threadIdx.x ;i< 3456;i += blockDim.x)
        sfreq[i] = 0;////����ա�
    __syncthreads();
    int *myfreq = &sfreq[27*threadIdx.x];
    int gridsize = blockDim.x * gridDim.x;
    for(int i=threadIdx.x + blockIdx.x*blockDim.x; i< len; i += gridsize)
        //if((d_dat[i]>='a')&&(d_dat[i]<='z'))//���ȷ������ֻ��a--z�����԰�ifȥ����
            myfreq[d_dat[i]-'a']++;
    __syncthreads();///���߳�ͳ�Ƶ��Լ���sharememory�С�
	///��һ��ѭ��ʵ���۰�ӡ�
	for(int roll = 64;roll>=1; roll>>=1)
	{
		if(threadIdx.x <roll)
		{
			for(int i=0;i<26;i++)
				myfreq[i] += sfreq[27*(threadIdx.x+roll)+i];
		}
		__syncthreads();
	}
#if 0
    if(threadIdx.x<64)
    {
        for(int i=0;i<26;i++)
           myfreq[i] += sfreq[27*(threadIdx.x+64)+i];
    }
    __syncthreads();
    if(threadIdx.x<32)
    {
        for(int i=0;i<26;i++)
            myfreq[i] += sfreq[27*(threadIdx.x+32)+i];
    }
    __syncthreads();
    if(threadIdx.x<16)
    {
        for(int i=0;i<26;i++)
            myfreq[i] += sfreq[27*(threadIdx.x+16)+i];
    }
    if(threadIdx.x< 8)
    {
        for(int i=0;i<26;i++)
            myfreq[i] += sfreq[27*(threadIdx.x+ 8)+i];
    }
    if(threadIdx.x< 4)
    {
        for(int i=0;i<26;i++)
            myfreq[i] += sfreq[27*(threadIdx.x+ 4)+i];
    }    
    if(threadIdx.x< 2)
    {
        for(int i=0;i<26;i++)
            myfreq[i] += sfreq[27*(threadIdx.x+ 2)+i];
    }
    if(threadIdx.x == 0)
    {
        for(int i=0;i<26;i++)
            myfreq[i] += sfreq[27*(threadIdx.x   )+i];
    }
#endif
	__syncthreads();

    if(threadIdx.x<26)///����Կ�֧��ԭ�Ӽӣ�����ʹ��ԭ�Ӽӣ�ֱ�Ӽӵ��Դ��ϡ�������û�еڶ����� 1.1������֧��ȫ���Դ��32λԭ�Ӳ�����
	    atomicAdd(&d_freq[threadIdx.x],sfreq[threadIdx.x]);

}
__global__ void freqencyMethod2(char *d_dat,int len, int *d_freq)
{//���������Ƚ�����ԭ�Ӽӵ�share memory�У�Ȼ�����ۼӵ��Դ��ϡ�

    __shared__ int sfreq[26];//

    if(threadIdx.x < 26)
        sfreq[threadIdx.x] = 0;////����ա�
    __syncthreads();
    int gridsize = blockDim.x * gridDim.x;
	int pos = 0;
    for(int i=threadIdx.x + blockIdx.x*blockDim.x; i< len; i += gridsize)
	{
		pos = d_dat[i]-'a';
		atomicAdd(&sfreq[pos],1);
	}
	__syncthreads();

    if(threadIdx.x<26)///����Կ�֧��ԭ�Ӽӣ�����ʹ��ԭ�Ӽӣ�ֱ�Ӽӵ��Դ��ϡ�������û�еڶ����� 1.1������֧��ȫ���Դ��32λԭ�Ӳ�����
	    atomicAdd(&d_freq[threadIdx.x],sfreq[threadIdx.x]);

}

void hostCalc(char *dat,int len,int *freqency)
{
    int freque[32];
	memset(freque,0,32*sizeof(int));
	for(int i=0;i<len;i++)
	{
	    if((dat[i]>='a')&&(dat[i]<='z'))
		    freque[dat[i]-'a']++;
	}
	memcpy(freqency,freque,26*sizeof(int));
}


int main(int argc,char **argv)
{
	//makeData("char26.dat",104857600);
	//return 0;
	if(argc<2)
	{
		fprintf(stdout,"usage: a.out datfile\n");
		return -1;
	}
	FILE *fr = NULL;
	if((fr = fopen(argv[1],"r"))==NULL)
	{
		fprintf(stderr,"can't open file %s\n",argv[1]);
		return -1;
	}
	fseek(fr,0,2);
	int len = ftell(fr);
	rewind(fr);
	len = (len-(len&4095))+4096;
	char *dat = new char[len];
	memset(dat,0,len);
	len = fread(dat,1,len,fr);
	fclose(fr);
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		free(dat);
        return -1;
    }
	char *d_dat;
	int *d_freq;
	int gpuFreq[32];
	int cpuFreq[32];
	cudaEvent_t start, stop;
	clock_t t0,t1,t2;
	float cptime,runtime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );//��¼ʱ���
	t0 = clock();
	
	cudaStatus = cudaMalloc((void **)&d_dat,len*sizeof(char));
	if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
		free(dat);
        return -1;
    }
	cudaStatus = cudaMalloc((void **)&d_freq,32*sizeof(int));
	if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
		cudaFree(d_dat);
		free(dat);
        return -1;
    }
	cudaMemcpy(d_dat,dat,len*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemset(d_freq,0,32*sizeof(int));
	cudaEventRecord( stop, 0 );////��¼ʱ���
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &cptime, start, stop );
	t1 = clock();
	
	freqencyStep1<<<256,128>>>(d_dat,len,d_freq);
//	freqencyMethod2<<<256,128>>>(d_dat,len,d_freq);
	cudaStatus = cudaThreadSynchronize();
	if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
		cudaFree(d_freq);
		cudaFree(d_dat);
		free(dat);
        return -1;
    }
	cudaEventRecord( start, 0 );////��¼ʱ���
	cudaEventSynchronize( start );
	cudaEventElapsedTime( &runtime, stop,start );///���������á�
	t2 = clock();

	cudaMemcpy(gpuFreq,d_freq,32*sizeof(int),cudaMemcpyDeviceToHost);

	clock_t ht0 = clock();
	cudaEventRecord( start, 0 );////��¼ʱ���
	cudaEventSynchronize( start );
	hostCalc(dat, len,cpuFreq);
	cudaEventRecord( stop, 0 );////��¼ʱ���
	cudaEventSynchronize( stop );
	clock_t ht1 = clock();
	float hruntime =0.0f;
	cudaEventElapsedTime( &hruntime, start,stop );///���������ʱ�䡣

	cudaFree(d_freq);
	cudaFree(d_dat);
	///check
	if(memcmp(gpuFreq,cpuFreq,26*sizeof(int))!=0)
		fprintf(stdout,"CHECK ERROR\n");
	else
		fprintf(stdout,"CHECK OK\n");

	free(dat);

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("cptime %9.4f ms  runtime %9.4f ms\n",cptime,runtime); 
	printf("t1-t0=%d   t2-t1 = %d \n",t1-t0,t2-t1);
	printf("host run time = %9.4f ms %d \n",hruntime,ht1-ht0);


    return 0;
}
