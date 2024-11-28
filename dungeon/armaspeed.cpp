#include <iostream>
#include <armadillo>
#include <chrono>
using namespace std;


int main() 
{
    chrono::steady_clock sc;
    int n = 500;
    arma::Mat<double> A = arma::randu(n, n);
    arma::Mat<double> B = arma::randu(n, n);
    auto start = sc.now();     // start timer

    for (int i = 0; i < 20; i++)
    {
        arma::Mat<double> C = A * B;
    }
    auto end = sc.now();
    auto time_span = static_cast<chrono::duration<double>>(end - start);
    cout << "Operation took: " << time_span.count() << " seconds !!!";
    return 0;
}
