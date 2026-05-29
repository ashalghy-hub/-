#include <iostream>
#include <chrono>
#include <immintrin.h> 
#include <omp.h>
#include <cstdlib>
#include<vector>

using namespace std;
const int N = 2048; 
alignas(32) float m[N][N];

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

void LU_OpenMP_AVX() {
    #pragma omp parallel
    {
        for (int k = 0; k < N; ++k) {           
            #pragma omp single
            {
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

            #pragma omp for schedule(guided)
            for (int i = k + 1; i < N; ++i) {
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
           
        }
    }
}

int main() {
    system("chcp 65001 > nul");
    cout << "矩阵规模: " << N << " x " << N << " (AVX + OpenMP)" << endl;
    
    vector<int> thread_counts = {1, 2, 4, 8, 12, 16, 20};
    
    for (int t : thread_counts) {
        omp_set_num_threads(t);
        double total_time = 0;
        int runs = 5;
        for(int r = 0; r < runs; ++r) {
            init_matrix();
            auto start = chrono::high_resolution_clock::now();
            LU_OpenMP_AVX();
            auto end = chrono::high_resolution_clock::now();
            total_time += chrono::duration<double, milli>(end - start).count();
        }
        
        cout << "[ " << t << " 个线程 ] 平均耗时: " << total_time / runs << " ms" << endl;
    }

    return 0;
}