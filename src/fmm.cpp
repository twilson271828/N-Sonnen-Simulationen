#include "../include/fmm.hpp"


arma::Mat<double> fmm::get_force_matrix(std::vector<sonne> &bodies, double theta, double G, double dt){

    int n = bodies.size();
    arma::Mat<double> A = arma::randu(n, n);

    for (auto &m : bodies){

        
        if (m.get_mass() == 0){
            throw std::invalid_argument("Mass cannot be zero");
        }
    }
    return A;


}