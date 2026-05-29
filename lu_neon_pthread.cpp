#include <iostream>
#include <chrono>
#include <pthread.h>
#include <cstdlib>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#else
#error "该代码必须在 ARM 架构服务器上编译运行！"
#endif

using namespace std;

const int N = 2048; 
alignas(16) float m[N][N];

const int NUM_THREADS = 8;

pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

struct ThreadParam {
    int t_id;
};

void init_matrix() {
    for (int i = 0; i < N; i++) {
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) m[i][j] = rand() % 10;
        for (int j = 0; j < i; j++) m[i][j] = 0;
    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) m[i][j] += m[k][j];
        }
    }
}

void* threadFunc(void* param) {
    ThreadParam* p = (ThreadParam*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        
        if (t_id == 0) {
            float pivot = m[k][k];
            float32x4_t vt = vdupq_n_f32(pivot);
            int j = k + 1;

            for (; j + 4 <= N; j += 4) {
                float32x4_t va = vld1q_f32(&m[k][j]); 

                va = vdivq_f32(va, vt);
                vst1q_f32(&m[k][j], va); 
            }
            for (; j < N; ++j) m[k][j] = m[k][j] / pivot;
            m[k][k] = 1.0;
        }
        
        pthread_barrier_wait(&barrier_Divsion);

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            float factor = m[i][k];
            float32x4_t vaik = vdupq_n_f32(factor);
            int j = k + 1;
            
            for (; j + 4 <= N; j += 4) {
                float32x4_t vakj = vld1q_f32(&m[k][j]);
                float32x4_t vaij = vld1q_f32(&m[i][j]);            
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                
                vst1q_f32(&m[i][j], vaij);
            }
            for (; j < N; ++j) m[i][j] = m[i][j] - factor * m[k][j];
            m[i][k] = 0.0;
        }

        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}

int main() {
    cout << "矩阵规模: " << N << " x " << N << " (ARM NEON + Pthread静态线程池)" << endl;
    cout << "工作线程数: " << NUM_THREADS << endl;

    init_matrix();

    pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    pthread_t handles[NUM_THREADS];
    ThreadParam params[NUM_THREADS];

    auto start = chrono::high_resolution_clock::now();

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc, (void*)&params[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration<double, milli>(end - start).count();
    cout << "ARM 平台 Pthread 平均耗时: " << time << " ms" << endl;

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

    return 0;
}