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
    __syncthreads();

//    myfreq = &d_freq[blockIdx.x * 26 + blockIdx.x];///��һ���Ľ���ȱ��浽�Դ��С�ÿ��block��0��25���̱߳�������
//    if(threadIdx.x<26)
//        myfreq[i] = sfreq[i];
    if(threadIdx.x<26)///����Կ�֧��ԭ�Ӽӣ�����ʹ��ԭ�Ӽӣ�ֱ�Ӽӵ��Դ��ϡ�������û�еڶ����� 1.1������֧��ȫ���Դ��32λԭ�Ӳ�����
	    atomicAdd(&d_freq[threadIdx.x],sfreq[threadIdx.x]);

}
#if 0
__global__ void frequencyStep2(int *d_freq,int *d_swap,int blocksInStep1)//���ǵ�һ��ʹ�õ�block�ĸ�����������٣���ôֱ����26���̼߳��꼴�ɡ�
{////Ӳ����֧��ԭ�Ӽ�ʱ����һ��Ҫ��д���Դ棬Ȼ��ڶ��������ۼӡ��ڶ���ÿ��blockֻʹ��208���̣߳�8��26�̡߳�
    __shared__ int sfreq[256];

    for(int i=threadIdx.x;i<256;i+=blockDim.x) sfreq[i ] = 0;
    __syncthreads();
	int allNumber = blocksInStep1*26;
	int gridSize = blockDim.x*gridDim.x;
    for(int i=threadIdx.x;i<allNumber;i+=gridSize)
	    sfreq[threadIdx.x ] += d_freq[i];
	__syncthreads();
	if(threadIdx.x<104)
	    sfreq[threadIdx.x]+= sfreq[threadIdx.x+104];
	__syncthreads();
	if(threadIdx.x<52)
	    sfreq[threadIdx.x]+= sfreq[threadIdx.x+52];
	__syncthreads();
	if(threadIdx.x<26)
	{
	    sfreq[threadIdx.x]+= sfreq[threadIdx.x+26];
		////��д���Դ��С�
		d_swap[threadIdx.x+blockIdx.x*26] = sfreq[threadIdx.x];
	}
}
////step2��Ҫ���ü��Σ�ֱ�����8��26��Ƶ�����ڡ�
#endif
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

void makeData(char *filename,int len)
{
	if(len<0) {
		fprintf(stdout,"len = %d\n",len);
		return;
	}
	FILE *fp = fopen(filename,"w");
	int len1  = (len-(len&1023)+1024;
	char *dat = new char [len1];
	memset(dat,0,len1);
	srand(0);
	for(int i=0;i<len;i++)
	{
		int x = rand();
		x%=26;
		dat[i] = 'a'+x;
	}
	fwrite(dat,1,len,fp);
	fclose(fp);
}
int main(int argc,char **argv)
{
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
	
	freqencyStep1<<<256,128>>>(d_dat,len,d_freq);
	cudaStatus = cudaThreadSynchronize();
	if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
		cudaFree(d_freq);
		cudaFree(d_dat);
		free(dat);
        return -1;
    }
	cudaMemcpy(gpuFreq,d_freq,32*sizeof(int),cudaMemcpyDeviceToHost);

	hostCalc(dat, len,cpuFreq);
	cudaFree(d_freq);
	cudaFree(d_dat);
	///check
	if(memcmp(gpuFreq,cpuFreq,26*sizeof(int))!=0)
		fprintf(stdout,"CHECK ERROR\n");
	else
		fprintf(stdout,"CHECK OK\n");

	free(dat);

    return 0;
}
