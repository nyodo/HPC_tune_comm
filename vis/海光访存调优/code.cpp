#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib> // for rand()

// 核函数定义（保持不变）
__global__ void gauss_all_seidel_backfor(int mne, int nv, int* nc, double* a_ae, double* f,
                                         int* ne, double* ap, double* con, double* ff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mne)
    {
        double tmp_b = 0.0;
        int j;
        for (j = nc[i]; j <= nc[i + 1] - 1; j++)
        {
            tmp_b += a_ae[j] * f[(nv - 1) * mne + ne[j] - 1];
        }
        ff[i] = (tmp_b + con[i]) / ap[i];
    }
}

int main(int argc, char* argv[])
{
    // 从命令行获取 mne 和 nv
    const int mne = std::atoi(argv[1]); // 例如 1024 或 2048
    const int nv = std::atoi(argv[2]);  // 例如 5

    // 动态分配主机端数组
    int* h_nc = new int[mne + 1];      // nc 数组大小为 mne+1
    double* h_a_ae = new double[mne];  // a_ae 数组大小为 mne
    int* h_ne = new int[mne];          // ne 数组大小为 mne
    double* h_f = new double[nv * mne]; // f 数组大小为 nv * mne
    double* h_ap = new double[mne];    // ap 数组大小为 mne
    double* h_con = new double[mne];   // con 数组大小为 mne
    double* h_ff = new double[mne];    // ff 数组大小为 mne

    // 初始化主机端数组
    for (int i = 0; i <= mne; ++i)
    {
        h_nc[i] = i; // 简单初始化 nc，每个单元对应一个项
    }
    for (int i = 0; i < mne; ++i)
    {
        h_ne[i] = (i % mne) + 1; // ne 索引在 [1, mne] 范围内
        h_a_ae[i] = 1.0;         // a_ae 初始化为 1.0
        h_ap[i] = 2.0;           // ap 初始化为 2.0，避免除以零
        h_con[i] = 1.0;          // con 初始化为 1.0
        h_ff[i] = 0.0;           // ff 初始化为 0.0
    }
    for (int i = 0; i < nv * mne; ++i)
    {
        h_f[i] = (double)(i % 100) / 10.0; // f 初始化为一些变化的值
    }

    // 分配设备端内存
    int *d_nc, *d_ne;
    double *d_a_ae, *d_f, *d_ap, *d_con, *d_ff;
    hipMalloc(&d_nc, (mne + 1) * sizeof(int));
    hipMalloc(&d_ne, mne * sizeof(int));
    hipMalloc(&d_a_ae, mne * sizeof(double));
    hipMalloc(&d_f, nv * mne * sizeof(double));
    hipMalloc(&d_ap, mne * sizeof(double));
    hipMalloc(&d_con, mne * sizeof(double));
    hipMalloc(&d_ff, mne * sizeof(double));

    // 将主机数据拷贝到设备
    hipMemcpy(d_nc, h_nc, (mne + 1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_ne, h_ne, mne * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_a_ae, h_a_ae, mne * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_f, h_f, nv * mne * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_ap, h_ap, mne * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_con, h_con, mne * sizeof(double), hipMemcpyHostToDevice);

    // 设置线程块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (mne + threadsPerBlock - 1) / threadsPerBlock;

    // 启动核函数
    hipLaunchKernelGGL(gauss_all_seidel_backfor, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       mne, nv, d_nc, d_a_ae, d_f, d_ne, d_ap, d_con, d_ff);

    // 等待设备完成计算
    hipDeviceSynchronize();

    // 将结果从设备拷回主机
    hipMemcpy(h_ff, d_ff, mne * sizeof(double), hipMemcpyDeviceToHost);

    // 打印部分结果以验证
    printf("%d %d\n",mne,nv);
    std::cout << "前10个结果: ";
    for (int i = 0; i < 10 && i < mne; ++i)
    {
        std::cout << h_ff[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备端内存
    hipFree(d_nc);
    hipFree(d_ne);
    hipFree(d_a_ae);
    hipFree(d_f);
    hipFree(d_ap);
    hipFree(d_con);
    hipFree(d_ff);

    // 释放主机端动态分配的内存
    delete[] h_nc;
    delete[] h_a_ae;
    delete[] h_ne;
    delete[] h_f;
    delete[] h_ap;
    delete[] h_con;
    delete[] h_ff;

    return 0;
}