#include <iostream>
#include <chrono>
#include <pthread.h>
#include <cstdlib>
#include <vector> 

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#else
#error "该代码必须在 ARM 架构服务器上编译运行！"
#endif

using namespace std;

const int MAX_N = 2048; 
alignas(16) float m[MAX_N][MAX_N];

int current_N; 

const int NUM_THREADS = 8;

pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

struct ThreadParam {
    int t_id;
};

void init_matrix() {
    for (int i = 0; i < current_N; i++) {
        m[i][i] = 1.0;
        for (int j = i + 1; j < current_N; j++) m[i][j] = rand() % 10;
        for (int j = 0; j < i; j++) m[i][j] = 0;
    }
    for (int k = 0; k < current_N; k++) {
        for (int i = k + 1; i < current_N; i++) {
            for (int j = 0; j < current_N; j++) m[i][j] += m[k][j];
        }
    }
}

void* threadFunc(void* param) {
    ThreadParam* p = (ThreadParam*)param;
    int t_id = p->t_id;

    for (int k = 0; k < current_N; ++k) {
        
        if (t_id == 0) {
            float pivot = m[k][k];
            float32x4_t vt = vdupq_n_f32(pivot);
            int j = k + 1;

            for (; j + 4 <= current_N; j += 4) {
                float32x4_t va = vld1q_f32(&m[k][j]); 

                va = vdivq_f32(va, vt);
                vst1q_f32(&m[k][j], va); 
            }
            for (; j < current_N; ++j) m[k][j] = m[k][j] / pivot;
            m[k][k] = 1.0;
        }
        
        pthread_barrier_wait(&barrier_Divsion);

        for (int i = k + 1 + t_id; i < current_N; i += NUM_THREADS) {
            float factor = m[i][k];
            float32x4_t vaik = vdupq_n_f32(factor);
            int j = k + 1;
            
            for (; j + 4 <= current_N; j += 4) {
                float32x4_t vakj = vld1q_f32(&m[k][j]);
                float32x4_t vaij = vld1q_f32(&m[i][j]);            
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                
                vst1q_f32(&m[i][j], vaij);
            }
            for (; j < current_N; ++j) m[i][j] = m[i][j] - factor * m[k][j];
            m[i][k] = 0.0;
        }

        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}

int main() {
    cout << "--- ARM NEON + Pthread 静态线程池 (多规模自动化测试) ---" << endl;
    cout << "工作线程数为: " << NUM_THREADS << endl << endl;

    vector<int> test_sizes = {512, 1024, 2048};

    for (int n : test_sizes) {
        current_N = n;
        cout << "正在测试矩阵规模: " << current_N << " x " << current_N << " ..." << endl;

        init_matrix();

        // 为当前的测试初始化 Barrier
        pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
        pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

        pthread_t handles[NUM_THREADS];
        ThreadParam params[NUM_THREADS];

        auto start = chrono::high_resolution_clock::now();

        // 创建线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&params[t_id]);
        }

        // 回收线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        auto end = chrono::high_resolution_clock::now();
        double time = chrono::duration<double, milli>(end - start).count();
        
        cout << "-> ARM 平台 Pthread 平均耗时: " << time << " ms\n" << endl;

        // 销毁 Barrier，准备下一轮不同规模的测试
        pthread_barrier_destroy(&barrier_Divsion);
        pthread_barrier_destroy(&barrier_Elimination);
    }

    cout << "测试完毕" << endl;
    return 0;
}
