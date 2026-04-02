#include<iostream>
#include <vector>
using namespace std;

const int N = 200000;     
const int REPEAT = 1000;  

long long pf(const vector<int>& data) {
    long long sum = 0;
    int i = 0;
    for (; i < N; i++) {
        sum += data[i];
    }
    return sum;
}

int main(){
    vector<int> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = i % 100; 
    }
    long long result = 0;
    for(int i=0;i<REPEAT;++i){
        result = pf(data);
    }
    return 0;
}