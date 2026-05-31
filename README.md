# 并行程序设计：多线程编程实验

本仓库包含了并行程序设计第二次实验（多线程编程）的核心源代码。
本次实验以普通高斯消去法（LU分解）为载体，深度融合了底层向量化指令集（x86 AVX2 / ARM NEON）与共享内存多线程技术（Pthread / OpenMP），并完成了跨体系架构的性能对比。

## 代码文件架构与核心优化点

本仓库包含以下 3 份核心源代码文件，分别针对不同的并行范式与硬件平台进行了深度定制：

### 1. `lu_pthread.cpp` (x86 平台: Pthread 优化版)
- **底层向量化**：采用 256-bit AVX2 指令集（`_mm256_loadu_ps` 等）榨取单核浮点吞吐极限。
- **静态线程池架构**：抛弃了在 $O(N)$ 外层循环中频繁创建/销毁线程的做法，通过 `pthread_create` 在程序伊始构建静态线程池。
- **任务分配策略**：采用 **静态循环划分** (`i = k + 1 + t_id; i += NUM_THREADS`)。该策略实现了完美的负载均衡，避免了全局任务队列的锁竞争，且在物理内存上天然隔离，**彻底规避了多核写入时的伪共享问题**。
- **同步机制**：使用 `pthread_barrier_t` 实施低开销的系统级屏障同步。

### 2. `lu_omp_avx.cpp` (x86 平台: OpenMP 扩展性测试版)
- **调度策略探索**：为了应对高斯消去步计算量呈二次方递减的“负载不均”特征，在内层消去循环中采用了 `schedule(guided)` 引导调度/动态调度机制。
- **扩展性探究**：内置了 1~20 线程的自动遍历测算逻辑，用于生成不同并发度下的阿姆达尔性能退化曲线。

### 3. `lu_neon_pthread.cpp` (ARM 平台: 鲲鹏服务器适配版)
- **跨架构移植**：为满足进阶要求，将 x86 的 AVX 指令全套翻译重构为 **ARM NEON 128-bit 向量化指令**（如 `vld1q_f32`, `vdivq_f32`, `vsubq_f32`）。
- **多维度测试**：内置了不同矩阵规模（N=512, 1024, 2048）的缩放性自动化测试逻辑，用于验证大规模矩阵下 $O(N^3)$ 复杂度的耗时斜率。

---

## 编译与运行指南

本项目严格区分 x86 架构与 ARM 架构。请根据当前硬件环境选择相应的编译指令：

### 本地 x86 PC 环境 (Intel/AMD)

**测试 Pthread 静态线程池版本：**

```bash
g++ lu_pthread.cpp -O2 -mavx2 -pthread -o run_pthread
./run_pthread
```

**测试 OpenMP 扩展性版本：**

```bash
g++ lu_omp_avx.cpp -O2 -mavx2 -fopenmp -o run_omp
./run_omp
```
### 鲲鹏服务器环境 (ARM aarch64)

**测试 ARM NEON 跨平台版本：**
```bash
g++ lu_neon_pthread.cpp -O2 -march=armv8-a -pthread -o run_neon
./run_neon
```


# 并行程序设计：MPI 分布式内存编程实验

本仓库包含了并行程序设计第三次实验（MPI 编程）的核心源代码。
本次实验以 $O(N^3)$ 的普通高斯消去法（LU分解）为载体，在 **鲲鹏 ARM 服务器集群** 上完成了从基础纯 MPI 到顶级超算混合编程架构的全面探索。

## 代码文件架构与核心优化点

本仓库共包含 3 份核心 C++ 源代码文件，分别对应实验的不同递进阶段：

### 1. `lu_mpi.cpp` (基础纯 MPI 与缩放性测试)
- **任务划分策略**：摒弃了极易导致负载不均的块划分，采用**一维循环划分**。天然保证了消去后期的负载绝对均衡，且完美契合行主序，彻底消除了写竞争与伪共享。
- **实验目标**：内置动态规模参数解析，用于测试 1、2、4、8 进程的缩放性，并成功复现且分析了 8 进程超载下的通信风暴的底层体系结构异常。

### 2. `lu_mpi_omp_neon.cpp` (进阶：MPI + OpenMP + NEON 三合一)
- **架构重构**：针对纯 MPI 的跨核通信风暴，重构为**“MPI 管网络、OpenMP 管节点、NEON 管指令”**的混合并行架构。
- **底层向量化**：结合 ARM 鲲鹏平台，手工编写 128-bit NEON 向量指令 (`vld1q_f32`, `vsubq_f32`)。
- **实验表现**：通过分配 `2个进程 × 4个线程`，将通信开销降至最低，成功将 $N=1024$ 的耗时从 9150ms 极限压缩至 **37.88 ms**。

### 3. `lu_mpi_async.cpp` (进阶探索：非阻塞流水线通信)
- **通信隐藏**：将传统的强阻塞广播 `MPI_Bcast` 替换为**非阻塞通信 `MPI_Ibcast` + `MPI_Wait`**。
- **流水线优化**：在发起异步硬件网络传输的间隙，让 CPU 提前进行循环边界更新等预处理，实现了通信与计算的部分重叠。实测在极小时间窗内依然压榨出了约 12.6% 的额外性能（降至 33.09 ms）。

---

## 编译与运行指南

本实验代码高度绑定 **ARM aarch64 (鲲鹏)** 平台及底层的 MPI 环境。请在 Linux 服务器终端下执行以下指令：

### 1. 编译所有文件
请使用 MPI 专用的 C++ 编译器包装器 `mpicxx`，并开启 ARMv8 优化与 OpenMP 支持：
```bash
mpicxx lu_mpi.cpp -O2 -o run_mpi
mpicxx lu_mpi_omp_neon.cpp -O2 -march=armv8-a -fopenmp -o run_hybrid
mpicxx lu_mpi_async.cpp -O2 -march=armv8-a -fopenmp -o run_async
```

### 2.运行测试示例
测试纯 MPI 缩放性 (以 N=1024, 4进程为例)：

```bash
mpirun -np 4 ./run_mpi 1024
```
测试混合编程极致性能 (2个MPI进程，每个进程4个OpenMP线程)：

```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./run_hybrid
```

测试非阻塞通信重叠 (异步广播)：

```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./run_async
```
