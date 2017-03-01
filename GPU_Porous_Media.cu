#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <time.h>
#include <string>
#include <iostream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;

#include<cuda_runtime.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define PI 3.1415926535897932384626433832795028841971693993751e0
#define MAXTHREADS (1024*1024)
#define blockSize   512
#define REDUCTIONBLOCKS 2048   //归约所允许的最大BLOCKS
#define THEAD  512
#define BLOCKS(a) ((a+2)*(a+2)/THEAD) //允许的最大网格为4098，Thread为512
//#define BLOCKS(a) ((a)*(a)/THEAD+1) 
#define THEADS (BLOCKS*THEAD)
#define DIM 2
#define Q 9
#define MEQ_0(Delta_Rho) (Delta_Rho)
#define MEQ_1(Delta_Rho,vx,vy) ((-2.0)*Delta_Rho+3.0*(vx*vx+vy*vy))
#define MEQ_2(Delta_Rho,vx,vy) (Delta_Rho-3.0*(vx*vx+vy*vy))
#define MEQ_3(vx) (vx)
#define MEQ_4(vx) (-vx)
#define MEQ_5(vy) (vy)
#define MEQ_6(vy) (-vy)
#define MEQ_7(vx,vy) ((vx*vx-vy*vy))
#define MEQ_8(vx,vy) (vx*vy)
#define OMEGA(k) (d_Mni_S[k][0] * (meq[0] - m[0])+\
                  d_Mni_S[k][1] * (meq[1] - m[1])+\
				  d_Mni_S[k][2] * (meq[2] - m[2])+\
				  d_Mni_S[k][3] * (meq[3] - m[3])+\
				  d_Mni_S[k][4] * (meq[4] - m[4])+\
				  d_Mni_S[k][5] * (meq[5] - m[5])+\
				  d_Mni_S[k][6] * (meq[6] - m[6])+\
				  d_Mni_S[k][7] * (meq[7] - m[7])+\
				  d_Mni_S[k][8] * (meq[8] - m[8]))
#define	DELTA_RHO(y,x) (f[0][y][x]+f[1][y][x]+f[2][y][x]+f[3][y][x]+f[4][y][x]+f[5][y][x]+f[6][y][x]+f[7][y][x]+f[8][y][x])
#define	VX(y,x) (f[1][y][x]-f[3][y][x]+f[5][y][x]+f[8][y][x]-f[6][y][x]-f[7][y][x])
#define	VY(y,x) (f[2][y][x]-f[4][y][x]+f[5][y][x]+f[6][y][x]-f[7][y][x]-f[8][y][x])

double     M[Q][Q] = { {  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0, 1.0 },
                       { -4.0,-1.0,-1.0,-1.0,-1.0, 2.0, 2.0,  2.0, 2.0 },
                       {  4.0,-2.0,-2.0,-2.0,-2.0, 1.0, 1.0,  1.0, 1.0 },
                       {  0.0, 1.0, 0.0,-1.0, 0.0, 1.0,-1.0, -1.0, 1.0 },
                       {  0.0,-2.0, 0.0, 2.0, 0.0, 1.0,-1.0, -1.0, 1.0 },
                       {  0.0, 0.0, 1.0, 0.0,-1.0, 1.0, 1.0, -1.0,-1.0 },
                       {  0.0, 0.0,-2.0, 0.0, 2.0, 1.0, 1.0, -1.0,-1.0 },
                       {  0.0, 1.0,-1.0, 1.0,-1.0, 0.0, 0.0,  0.0, 0.0 },
                       {  0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-1.0,  1.0,-1.0 } };

double     MI[Q][Q] = { { 1.0 / 9.0,   -1.0 / 9.0,    1.0 / 9.0 ,       0.0  ,      0.0   ,     0    ,      0     ,        0    ,         0 },
                        { 1.0 / 9.0,   -1.0 / 36.0,  -1.0 / 18.0,   1.0 / 6.0, -1.0 / 6.0 ,     0    ,      0     ,    1.0 / 4.0,         0 },
                        { 1.0 / 9.0,   -1.0 / 36.0,  -1.0 / 18.0,      0.0   ,    0.0     , 1.0 / 6.0, -1.0 / 6.0 ,   -1.0 / 4.0,         0 },
                        { 1.0 / 9.0,   -1.0 / 36.0,  -1.0 / 18.0,  -1.0 / 6.0,  1.0 / 6.0 ,     0    ,      0     ,    1.0 / 4.0,         0 },
                        { 1.0 / 9.0,   -1.0 / 36.0,  -1.0 / 18.0,       0    ,    0.0     ,-1.0 / 6.0,  1.0 / 6.0 ,   -1.0 / 4.0,         0 },
                        { 1.0 / 9.0,    1.0 / 18.0,   1.0 / 36.0,   1.0 / 6.0,  1.0 / 12.0, 1.0 / 6.0,  1.0 / 12.0,        0    ,     1.0 / 4.0 },
                        { 1.0 / 9.0,    1.0 / 18.0,   1.0 / 36.0,  -1.0 / 6.0, -1.0 / 12.0, 1.0 / 6.0,  1.0 / 12.0,        0    ,    -1.0 / 4.0 },
                        { 1.0 / 9.0,    1.0 / 18.0,   1.0 / 36.0,  -1.0 / 6.0, -1.0 / 12.0,-1.0 / 6.0, -1.0 / 12.0,        0    ,     1.0 / 4.0 },
                        { 1.0 / 9.0,    1.0 / 18.0,   1.0 / 36.0,   1.0 / 6.0,  1.0 / 12.0,-1.0 / 6.0, -1.0 / 12.0,        0    ,    -1.0 / 4.0 } };
const int      e_f[Q][2] = { { 0,0 },{ 1,0 },{ 0,1 },{ -1,0 },
                             { 0,-1 },{ 1,1 },{ -1,1 },{ -1,-1 },{ 1,-1 } };
double     rho0=1.0;
double     Mni_S[Q][Q];
__constant__ double d_M[Q][Q];
__constant__ double d_MI[Q][Q];
__constant__ double d_Mni_S[Q][Q];
__constant__ int    d_NX,d_NY,d_ne_f[Q],d_e_f[Q][DIM];
__constant__ double d_dx;
__constant__ double d_dt;
__constant__ double d_Force_x,d_Force_y;
double     dx,dt,Force;
const int      ne_f[Q] = { 0,3,4,1,2,7,8,5,6 };

int *Soid;
int grid[4],NX,NY;
double Volume_Fraction[2];
double *f,*u,*u0,*F,*DeltaRho,niu,*h_error,*h_force_in_x,*h_force_in_y,*h_totalforce_x,*h_totalforce_y;

bool InitCUDA();
void InitSolid();
void Space_Malloc(int Grid);
void Space_Free();
void InitMRT();
__device__  __host__ void Matrix_mult(const double *a, double *b, double *c, int l, int m, int n);
__global__           void Evolution(double *f_d,double *F_d,double *DeltaRho,int *Solid,double *d_u);
__global__           void reduce4(double *g_idata, double *g_odata,  int n);
                     void reduce(double *g_idata, double *g_odata,int nx,int ny);
__global__           void sumtotal(double *d_data,int n);
__global__           void Error(double *uold,double *unew,double*u_u0,double *u_base,int size);//size为数组长度
__global__           void WriteError(double *d_u_u0,double *d_u_L2,double *d_Error,int ith);
__global__           void Drag_Force(double *d_F,int *Solid,double *d_dragforce_x,double *d_dragforce_y);//计算受力
__global__           void WriteForce(double* d_dragforce_x,double* d_dragforce_y,double* d_force_in_x,double* d_force_in_y,int ith);
                   string num2str(double i);
int main(int argc, char *argv[])

{
	string       PathDir;
	int    Error_Enable=1,DevId=0;
	int    *d_Soid;
	double *d_f,  *d_F,*d_u,*d_u0,*d_DeltaRho,*d_dragforce_x/*x方向的力*/,*d_dragforce_y/*y方向的力*/,
	       *d_force_in_x/*储存x方向的力*/,*d_force_in_y/*储存y方向的力*/;
	double Fx,Fy;//驱动力在x和y方向上面的投影
	double *d_u_u0,*d_u_L2, *d_error;
	clock_t time_begin, time_end;
	FILE *LFileWrite,*ERRORWrite,*FORCEWrite;
    /*if(!InitCUDA()) {
            return 0;
    }*/
	//从命令行读入参数
	//this part is used to read the parameters
	sscanf( argv[1], "%d", grid );  // please input the Grid points in x direction 
	sscanf( argv[2], "%d", grid+1 );// please input the Grid points in y direction
	
	//please input the two argument to change the Volume Fraction
	sscanf( argv[3], "%d", grid+2 );
	sscanf( argv[4], "%d", grid+3 );
	
	//input the angele of the flow
	sscanf( argv[5], "%lf", Volume_Fraction );
	
	//input the gradient of the folw
	sscanf( argv[6], "%lf", Volume_Fraction+1 );
	
	sscanf( argv[7], "%lf", &niu          );
	if(argc>8)
	sscanf( argv[8], "%d",  &Error_Enable );
    if(argc>9)
	sscanf( argv[9], "%d",  &DevId );
    if(DevId!=0)
	{
		cudaSetDevice(DevId);
	}
    
	//开辟cpu内存空间
	Space_Malloc(grid[0]);

	
	//开辟gpu设备空间，
	cudaMalloc( (void**)&d_f              , sizeof(double) * Q   * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_F              , sizeof(double) * Q   * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_u              , sizeof(double) * DIM * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_u0             , sizeof(double) * DIM * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_DeltaRho       , sizeof(double) * 1   * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_Soid           , sizeof(int   ) * 1   * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_u_u0           , sizeof(double) * 1   * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_u_L2           , sizeof(double) * 1   * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_error          , sizeof(double) * 1   * 100     * 100    ) ;
	cudaMalloc( (void**)&d_dragforce_x    , sizeof(double) * 1   * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_dragforce_y    , sizeof(double) * 1   * grid[0] * grid[1]) ;
	cudaMalloc( (void**)&d_force_in_x     , sizeof(double) * 1   * 100     * 100    ) ;
	cudaMalloc( (void**)&d_force_in_y     , sizeof(double) * 1   * 100     * 100    ) ;
	cudaCheckErrors("cuda malloc fail\n");
	InitMRT();
	
	//创建文件夹
    ostringstream name;
	name <<"Niu_"<<niu<<"_Force_"<<Volume_Fraction[1]<<"_"<< NX << "_" << NY<<"_"<<Volume_Fraction[0]<<"_"<<grid[2]<<"_"<<grid[3];
	PathDir = name.str();
	mkdir(PathDir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	
	Fx = Force*cos(PI*Volume_Fraction[0]);
	Fy = Force*sin(PI*Volume_Fraction[0]);
	//复制数据到常量内存
	cudaMemcpyToSymbol(d_M[0]    ,M[0]    ,sizeof(double)*Q*Q);
	cudaMemcpyToSymbol(d_MI[0]   ,MI[0]   ,sizeof(double)*Q*Q);
	cudaMemcpyToSymbol(d_Mni_S[0],Mni_S[0],sizeof(double)*Q*Q);
	cudaMemcpyToSymbol(d_ne_f    ,ne_f    ,sizeof(int   )  *Q);
	cudaMemcpyToSymbol(d_e_f[0]  ,e_f[0]  ,sizeof(int   )*2*Q);
	cudaMemcpyToSymbol(d_NX      ,&NX     ,sizeof(int   )    );
	cudaMemcpyToSymbol(d_NY      ,&NY     ,sizeof(int   )    );
	cudaMemcpyToSymbol(d_dx      ,&dx     ,sizeof(double)    );
	cudaMemcpyToSymbol(d_dt      ,&dt     ,sizeof(double)    );
	cudaMemcpyToSymbol(d_Force_x ,&Fx     ,sizeof(double)    );
	cudaMemcpyToSymbol(d_Force_y ,&Fy     ,sizeof(double)    );
	cudaCheckErrors("cuda memcpy fail\n");


	for(int i=0;i<10000;i++)
	{
		h_force_in_x[i]=0.0;
		h_force_in_y[i]=0.0;
		h_error     [i]=2.0;
	}
	
	//复制数据到显存
	cudaMemcpy(d_f ,       f       , sizeof(double)*Q  *NX*NY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F ,       F       , sizeof(double)*Q  *NX*NY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u ,       u       , sizeof(double)*DIM*NX*NY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u0,       u0      , sizeof(double)*DIM*NX*NY, cudaMemcpyHostToDevice);
	cudaMemcpy(d_DeltaRho, DeltaRho, sizeof(double)*NX*NY    , cudaMemcpyHostToDevice);
	cudaMemcpy(d_Soid    , Soid    , sizeof(int   )*NX*NY    , cudaMemcpyHostToDevice);
	cudaMemcpy(d_force_in_x    , h_force_in_x    , sizeof(double   )*10000    , cudaMemcpyHostToDevice);
	cudaMemcpy(d_force_in_y    , h_force_in_y    , sizeof(double   )*10000    , cudaMemcpyHostToDevice);
	cudaMemcpy(d_error    , h_error    , sizeof(double   )*10000    , cudaMemcpyHostToDevice);
	cudaCheckErrors("cuda memcpy from host fail\n");
	time_begin = clock();
	
	//分配线程
	int blocks=BLOCKS(NX);
	printf("the Blocks is : %d \n",blocks);
	for(int s=0;s<5000000;s++)
	{
	Evolution<<<blocks,THEAD>>>(d_f,d_F,d_DeltaRho,d_Soid,d_u);
	   if(s!=0&(s%1000==0)&Error_Enable==1)
	   {
		  // /*计算误差*/
		Error<<<blocks,THEAD>>>(d_u0,d_u,d_u_u0,d_u_L2,NX*NY);
		reduce(d_u_u0,d_u_u0,NX,NY);
		reduce(d_u_L2,d_u_L2,NX,NY);
		WriteError<<<1,1>>>(d_u_u0,d_u_L2,d_error,s/1000);
		cudaMemcpy(h_error      , d_error+(s/1000-1)         ,sizeof(double),cudaMemcpyDeviceToHost);
		
		   // /*计算受力*/
		Drag_Force<<<blocks,THEAD>>>(d_F, d_Soid, d_dragforce_x,d_dragforce_y);
		reduce(d_dragforce_x,d_dragforce_x,NX,NY);
		reduce(d_dragforce_y,d_dragforce_y,NX,NY);
		WriteForce<<<1,1>>>(d_dragforce_x,d_dragforce_y,d_force_in_x,d_force_in_y,s/1000);
		
		if(h_error[0]<1.0e-12)
			break;
	   }
	cudaCheckErrors("cuda Evolution call fail\n");
	}
    
	Drag_Force<<<blocks,THEAD>>>(d_F, d_Soid, d_dragforce_x,d_dragforce_y);
	cudaMemcpy(h_totalforce_y     , d_dragforce_y      ,sizeof(double)    *NX*NY,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_totalforce_x     , d_dragforce_x      ,sizeof(double)    *NX*NY,cudaMemcpyDeviceToHost);
	cudaMemcpy(u0           , d_u             ,sizeof(double)*DIM*NX*NY,cudaMemcpyDeviceToHost);
	cudaMemcpy(DeltaRho     , d_DeltaRho      ,sizeof(double)    *NX*NY,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_error      , d_error         ,sizeof(double)*1*100*100,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_force_in_x , d_force_in_x    ,sizeof(double)*1*100*100,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_force_in_y , d_force_in_y    ,sizeof(double)*1*100*100,cudaMemcpyDeviceToHost);
	cudaCheckErrors("cuda memcpy from device fail\n");
	
	    //输出程序运行时间
	time_end = clock(); 
	printf( "The computing time is: %f seconds\n", 
	(float)( time_end - time_begin ) / CLOCKS_PER_SEC );
	    //写入文件
	
	LFileWrite=fopen((PathDir+"/"+"force.dat").c_str(),"wr");
	
	int index;
	fprintf(LFileWrite,"Title=\"Porous Media Flow\"\n");
	fprintf(LFileWrite,"VARIABLLES=\"X\",\"Y\",\"U\",\"V\",\"fx\",\"fy\"\n");
	fprintf(LFileWrite,"ZONE T=\"BOX\",I=%d,J=%d,F=POINT\n",NX,NY);
	for(int i=0;i<NX;i++)
	{
		for(int j=0;j<NY;j++)
		{
			double x=(double(i)-0.5)*dx;
			double y=(double(j)-0.5)*dx;
			index=i*NX+j;
			fprintf(LFileWrite,"%.16lf  %.16lf  %.16lf  %.16lf  %.16lf  %.16lf  %.16lf\n",
			x,y,u0[index*2],u0[index*2+1],h_totalforce_x[index],h_totalforce_y[index],DeltaRho[index]+1.0);
		}
	}
	if(Error_Enable==1)
	{
	ERRORWrite=fopen((PathDir+"/"+"Error.txt").c_str() ,"wr");
	FORCEWrite=fopen((PathDir+"/"+"Force.txt").c_str() ,"wr");
	for(int i=0;i<10000;i++)
	{
		fprintf(ERRORWrite,"the error of %5d th is : %.13lf \n",(i+1)*1000,h_error[i]);
	    fprintf(FORCEWrite,"%.16lf  %.16lf \n",h_force_in_x[i],h_force_in_y[i]);

	}
	fclose(ERRORWrite);
	fclose(FORCEWrite);
	}
	
	fclose(LFileWrite);
	
       //释放内存空间	
	cudaFree(d_f       );
	cudaFree(d_F       );
	cudaFree(d_u       );
	cudaFree(d_u0      );
	cudaFree(d_DeltaRho);
	cudaFree(d_Soid    );
	cudaFree(d_u_u0    );
	cudaFree(d_u_L2    );
	cudaFree(d_error   );
	cudaFree(d_dragforce_x);
	cudaFree(d_dragforce_y);
	cudaFree(d_force_in_x );
	cudaFree(d_force_in_y );
	Space_Free();
    return 0;
}

void Space_Malloc(int Grid)
{
	Soid          =new int[Grid*Grid];
	f             =new double[Q*Grid*Grid];
	F             =new double[Q*Grid*Grid];
	u             =new double[DIM*Grid*Grid];
	u0            =new double[DIM*Grid*Grid];
	DeltaRho      =new double[DIM*Grid*Grid];
	h_totalforce_x=new double[Grid*Grid];
	h_totalforce_y=new double[Grid*Grid];
	h_error     =new double[1  *100 *100 ];
	h_force_in_x=new double[1  *100 *100 ];
	h_force_in_y=new double[1  *100 *100 ];
}
void Space_Free()
{
	delete []Soid    ;
	delete []f       ;
	delete []F       ;
	delete []u       ;
	delete []u0      ;
	delete []DeltaRho;
	delete []h_error ;
	delete []h_force_in_x;
	delete []h_force_in_y;
	delete []h_totalforce_x;
	delete []h_totalforce_y;
}
bool InitCUDA()
{
    int count;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device./n");
        return false;
    }

    int i;
    
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x./n");
        return false;
    }
    printf("the count of the device is %d,use %d th gpu\n",count,i);
    cudaSetDevice(i);

    return true;
}

void InitSolid()
{
	int i,j;
	int index;
	int dis_x, dis_y,NX=grid[0],NY=grid[1],a = (NX - 2) *grid[2] / grid[3];
	
	for(i=0;i<grid[0];i++)
		for(j=0;j<grid[1];j++)
		{
			index=i*NX+j;
			if (i<(NX - 2) / 2)
				dis_x = (NX - 2) / 2 - i;
			else
				dis_x = i - (NX) / 2;
			if (j<(NX - 2) / 2)
				dis_y = (NY - 2) / 2 - j;
			else
				dis_y = j - (NY) / 2;
			if (dis_x >= (a) && dis_y >= (a))
			{
				Soid[index] = 1;
			}
			else
				Soid[index] = 0;
		}
}
void InitMRT()
{
	InitSolid();
	NX=grid[0],NY=grid[1];
	int i,j,index,indexf;
	double Sdiag[Q],m[Q];
	dx = 1.0/double(NX-2);
	dt = dx ;
	
	Force = Volume_Fraction[1];// Lx / Ly;
	double Sniu = -2.0* dt / (6.0*niu + dt);
	double S_q  = -8.0*(2 + Sniu) / (8 + Sniu); //!!!!!

	Sdiag[7] = -Sniu;
	Sdiag[8] = Sdiag[7];
	Sdiag[1] = Sdiag[7];
	Sdiag[2] = Sdiag[7];

	Sdiag[4] = -S_q;
	Sdiag[6] = Sdiag[4];

	Sdiag[0] = 0.0;
	Sdiag[3] = 1.0;
	Sdiag[5] = 1.0;

	for ( i = 0; i < Q; i++)
		for (j = 0; j < Q; j++)
		{
			Mni_S[i][j] = 0.0;
			Mni_S[i][j] = MI[i][j] * Sdiag[j];

		}
	//数据初始化
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++)
		{
			index          =i*NX+j;
			indexf         =(i*NX+j)*Q;
			u[index*DIM  ] =0.0;
			u[index*DIM+1] =0.0;
			DeltaRho[index]=0.0;
			m[0]=MEQ_0(DeltaRho[index]);
			m[1]=MEQ_1(DeltaRho[index],u[index*DIM  ],u[index*DIM+1] );
			m[2]=MEQ_2(DeltaRho[index],u[index*DIM  ],u[index*DIM+1] );
			m[3]=MEQ_3(u[index*DIM  ]);
			m[4]=MEQ_4(u[index*DIM  ]);
			m[5]=MEQ_5(u[index*DIM+1]);
			m[6]=MEQ_6(u[index*DIM+1]);
			m[7]=MEQ_7(u[index*DIM],u[index*DIM+1]);
			m[8]=MEQ_8(u[index*DIM],u[index*DIM+1]);
			Matrix_mult(MI[0], m, f+indexf, Q, Q, 1);
		}
	
}
__global__ void Evolution(double *f_d,double *F_d,double *DeltaRho,int *Solid,double *d_u){
	int    tid=blockDim.x * blockIdx.x + threadIdx.x;
	double f[Q],m[Q],meq[Q];
	double u_0,u_1,deltarho;
	int    i,j,k,ip,jp;
	if(tid<d_NX*d_NX&Solid[tid]==0)
	{
		
		//Computation of density and velocity 
		
		f[0]=f_d[tid*Q+0];f[1]=f_d[tid*Q+1];f[2]=f_d[tid*Q+2];
		f[3]=f_d[tid*Q+3];f[4]=f_d[tid*Q+4];f[5]=f_d[tid*Q+5];
		f[6]=f_d[tid*Q+6];f[7]=f_d[tid*Q+7];f[8]=f_d[tid*Q+8];
		deltarho=(f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]);
		u_0= (f[1] - f[3] + f[5] + f[8] - f[6] - f[7]);
		u_1= (f[2] - f[4] + f[5] + f[6] - f[7] - f[8]);

		DeltaRho[tid] =deltarho;
		//collision
		Matrix_mult(d_M[0], f, m, Q, Q, 1);
		//第一次外力
		__syncthreads();

		u_1   = (u_1 +0.5*d_Force_y*d_dt);
		u_0   = (u_0 +0.5*d_Force_x*d_dt);
		
		__syncthreads();
		meq[1]= MEQ_1(deltarho,u_0,u_1);
		meq[7]= MEQ_7(u_0,u_1);
		meq[8]= MEQ_8(u_0,u_1);
		meq[4]= MEQ_4(u_0);
		meq[6]= MEQ_6(u_1);
		meq[2]= MEQ_2(deltarho,u_0,u_1);
		//第二次外力
		
		u_1   = (u_1 +0.5*d_Force_y*d_dt);
		u_0   = (u_0 +0.5*d_Force_x*d_dt);
		__syncthreads();
		meq[0]= MEQ_0(deltarho);
		meq[3]= MEQ_3(u_0);
		meq[5]= MEQ_5(u_1);
		__syncthreads();
		F_d[tid*Q+0]=(OMEGA(0)+f[0]);
		F_d[tid*Q+1]=(OMEGA(1)+f[1]);
		F_d[tid*Q+2]=(OMEGA(2)+f[2]);
		F_d[tid*Q+3]=(OMEGA(3)+f[3]);
		F_d[tid*Q+4]=(OMEGA(4)+f[4]);
		F_d[tid*Q+5]=(OMEGA(5)+f[5]);
		F_d[tid*Q+6]=(OMEGA(6)+f[6]);
		F_d[tid*Q+7]=(OMEGA(7)+f[7]);
		F_d[tid*Q+8]=(OMEGA(8)+f[8]);
		
		
		//流动
		__syncthreads();
		for(k=0;k<Q;k++)
		{
			i  =tid/d_NX;
			j  =tid%d_NX;
			ip = (i-d_e_f[k][0]+d_NX)%(d_NX);
			jp = (j-d_e_f[k][1]+d_NY)%(d_NY);
			ip = ip*d_NX+jp;
			// if(Solid[ip]==1)
				// f_d[tid*Q+k]=F_d[(tid*Q+d_ne_f[k])];
			// if(Solid[ip]==0)
				// f_d[tid*Q+k]=F_d[(ip*Q+k)];
			f_d[tid*Q+k] =F_d[(tid*Q+d_ne_f[k])*Solid[ip]+(ip*Q+k)*(1-Solid[ip])];
		}
		d_u[tid*DIM+0]=u_0;
		d_u[tid*DIM+1]=u_1;
	}
	
}

__device__  __host__ void Matrix_mult(const double *a, double *b, double *c, int l, int m, int n) // a is the matrix l*m ,b is the matrix m*n,c is the matrix l*n
{
	int i = 0, j = 0;
	for (i = 0; i<l; i++)
	{
		for (j = 0; j<n; j++)
		{
			c[i*n + j] = 0;
			for (int k = 0; k<m; k++)

				c[i*n + j] += a[i*m + k] * b[k*n + j];
		}

	}
}
__global__ void reduce4(double *g_idata, double *g_odata,  int n)
{
    __shared__ double sdata[blockSize];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	

    double mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i+blockSize];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        __syncthreads();
    }

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}
void reduce(double *g_idata, double *g_odata,int nx,int ny)//规约函数
{
	//当线程足够的时候，直接调用即可
	int blocks       = (nx*ny%THEAD)?(nx*ny/THEAD+1):(nx*ny/THEAD);
	int RemainBlocks = blocks%REDUCTIONBLOCKS;
	int threads      = nx*ny;
	int offset       = 0;

	for(offset=0;offset<threads;offset+=MAXTHREADS)
	{
		if((offset+MAXTHREADS)<=threads)
		{
			reduce4<<<REDUCTIONBLOCKS,THEAD>>>(g_idata+offset,g_odata+offset,MAXTHREADS     );
			reduce4<<<1              ,THEAD>>>(g_odata+offset,g_odata+offset,REDUCTIONBLOCKS);
		}
		else if(RemainBlocks!=0)
		{
			reduce4<<<RemainBlocks,THEAD>>>(g_idata+offset,g_odata+offset,threads-offset);
			reduce4<<<1           ,THEAD>>>(g_odata+offset,g_odata+offset,RemainBlocks  );
		}			
	}
	//将归约结果汇总，求和
	sumtotal<<<1,1>>>(g_odata,threads);
}
__global__ void sumtotal(double *d_data,int n)//将规约的结果求和
{
	int offset=0;
	while((offset+MAXTHREADS)<n)
	{
		d_data[0]+=d_data[offset+MAXTHREADS];
		offset=offset+MAXTHREADS;
	}
}
__global__ void Error(double *uold,double *unew,double*u_u0,double *u_base,int size)
{
	int   tid=blockDim.x * blockIdx.x + threadIdx.x;
	double uold0,uold1,unew0,unew1;
	if(tid<size)
	{
		uold0 = uold[tid*DIM  ];
		uold1 = uold[tid*DIM+1];
		unew0 = unew[tid*DIM];
		unew1 = unew[tid*DIM+1];
		uold[tid*DIM  ] = unew0 ;
		uold[tid*DIM+1] = unew1 ;
		uold0 = sqrt((uold0-unew0)*(uold0-unew0)+(uold1-unew1)*(uold1-unew1));
		unew0 = sqrt(unew0*unew0+unew1*unew1);
		u_u0  [tid]=uold0;
		u_base[tid]=unew0;
		uold  [tid] =unew [tid];
	}
}
__global__           void WriteError(double *d_u_u0,double *d_u_L2,double *d_Error,int ith)
{
	int   tid=threadIdx.x;
	d_Error[ith-1]=d_u_u0[tid]/(d_u_L2[tid]+1.0e-30);
}
__global__ void Drag_Force(double *d_F,int *Solid,double *d_dragforce_x,double *d_dragforce_y)
{
	int    tid=blockDim.x * blockIdx.x + threadIdx.x;
	int    k,ip,jp,i,j;
	if(tid<d_NX*d_NX)
	{
	d_dragforce_x[tid ]=0.0;
	d_dragforce_y[tid ]=0.0;
	}
	__syncthreads();
	if(tid<d_NX*d_NX&Solid[tid]==1)
	{
	d_dragforce_x[tid ]=0.0;
	d_dragforce_y[tid ]=0.0;
		for(k=0;k<Q;k++)
		{
			i  =tid/d_NX;
			j  =tid%d_NX;
			ip = (i+d_e_f[k][0]+d_NX)%(d_NX);
			jp = (j+d_e_f[k][1]+d_NY)%(d_NY);
			ip = ip*d_NX+jp;
			d_dragforce_x[tid] =d_dragforce_x[tid] +  2.0* double(Solid[ip]-1.0)*d_F[ip*Q+d_ne_f[k]]*double(d_e_f[k][0])/d_dt;
			d_dragforce_y[tid] =d_dragforce_y[tid] +  2.0* double(Solid[ip]-1.0)*d_F[ip*Q+d_ne_f[k]]*double(d_e_f[k][1])/d_dt;
		}
	}
}
__global__ void WriteForce(double* d_dragforce_x,double* d_dragforce_y,double* d_force_in_x,double* d_force_in_y,int ith)
{
	int   tid=threadIdx.x;
	d_force_in_x[ith-1]=d_dragforce_x[tid];
	d_force_in_y[ith-1]=d_dragforce_y[tid];
}
string num2str(double i)
{
	stringstream ss;
	ss << i;
	return ss.str();
}