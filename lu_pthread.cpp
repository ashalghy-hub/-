#include <iostream>
#include <chrono>
#include <immintrin.h> 
#include <pthread.h>
#include <cstdlib>

using namespace std;

const int N = 2048; 
alignas(32) float m[N][N];

const int NUM_THREADS = 4;

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
            __m256 vt = _mm256_set1_ps(pivot);
            int j = k + 1;
            for (; j + 8 <= N; j += 8) {
                __m256 va = _mm256_loadu_ps(&m[k][j]);
                va = _mm256_div_ps(va, vt);
                _mm256_storeu_ps(&m[k][j], va);
            }
            for (; j < N; ++j) m[k][j] = m[k][j] / pivot;
            m[k][k] = 1.0;
        }
        
        pthread_barrier_wait(&barrier_Divsion);
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            float factor = m[i][k];
            __m256 vaik = _mm256_set1_ps(factor);
            int j = k + 1;
            for (; j + 8 <= N; j += 8) {
                __m256 vakj = _mm256_loadu_ps(&m[k][j]);
                __m256 vaij = _mm256_loadu_ps(&m[i][j]);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(&m[i][j], vaij);
            }
            for (; j < N; ++j) m[i][j] = m[i][j] - factor * m[k][j];
            m[i][k] = 0.0;
        }
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}

/*int main() {
    system("chcp 65001 > nul");
    cout << "矩阵规模: " << N << " x " << N << " (AVX + Pthread静态线程池)" << endl;
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
    cout << "Pthread 平均耗时: " << time << " ms" << endl;

    // 4. 销毁 Barrier
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

    return 0;
}*/

int main() {
    system("chcp 65001 > nul");
    init_matrix();
    pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);
    
    cout << "开始连续运行，准备让 VTune 抓取..." << endl;
    
    for(int run = 0; run < 5; run++) { // 连续跑5次！
        pthread_t handles[NUM_THREADS];
        ThreadParam params[NUM_THREADS];
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&params[t_id]);
        }
        for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
    }
    
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);
    return 0;
}