#include "../include/sonne.hpp"
#include <bitset>
#include <cmath>
#include <iostream>
#include <limits>
#include <armadillo>
#include <sstream>

template <typename T> void printVector(std::vector<T> &x) {
  for (auto &i : x) {
    std::cout << i << " ";
  }
  std::cout << "\n";
}


int main() {

    int n = 10;

   // Initialize Armadillo
    arma::arma_rng::set_seed_random();  // Set a random seed for reproducibility

    // Create a random 100x100 matrix
    arma::mat A = arma::randu<arma::mat>(n, n);

    // Matrices to hold the SVD results
    arma::mat U;  // Orthogonal matrix
    arma::vec S;  // Singular values (diagonal matrix as a vector)
    arma::mat V;  // Orthogonal matrix

    // Compute the singular value decomposition
    arma::svd(U, S, V, A);

    // Output results
    std::cout << "Matrix A (n x n):\n" << A << "\n";
    std::cout << "\nSingular Values (S):\n" << S << "\n";
    std::cout << "\nMatrix U (n x n):\n" << U << "\n";
    std::cout << "\nMatrix V (n x n):\n" << V << "\n";

  return 0;
}