#include <iostream>
#include <vector>

using namespace std;

void youhua(int n, const vector<double>& A, const vector<double>& x, vector<double>& y) {
    for (int i = 0; i < n; ++i) y[i] = 0.0;
    for(int j=0;j<n;++j){
        for(int i=0;i<n;++i){
            y[i]+=A[j*n+i]*x[j];
        }
    }
}

int main() {
    int n = 2000;
    int re = 100;

    vector<double> A(n * n);
    vector<double> x(n);
    vector<double> y_y(n, 0.0);

    for (int j = 0; j < n; ++j) {
        x[j] = 1.0; 
        for (int i = 0; i < n; ++i) {
            A[j * n + i] = j + i; 
        }
    }

    for (int r = 0; r < re; ++r) {
        youhua(n, A, x, y_y);
    }

    return 0;
}