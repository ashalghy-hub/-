#include<iostream>
#include <vector>
using namespace std;

const int N = 200000;     
const int REPEAT = 1000;  

long long yh(const vector<int>& data) {
    long long sum0 = 0, sum1 = 0;
    int i = 0;
    for (; i + 1 < N; i += 2) {
        sum0 += data[i];
        sum1 += data[i + 1];
    }


    return sum0 + sum1;
}

int main(){
    vector<int> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = i % 100; 
    }
    long long result = 0;
    for(int i=0;i<REPEAT;++i){
        result = yh(data);
    }
    return 0;
}