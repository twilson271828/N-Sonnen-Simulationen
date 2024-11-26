#include "../include/fmm.hpp"


arma::mat fmm::get_force_matrix(std::vector<sonne> &bodies, double theta, double G, double dt){

    for (auto &m : bodies){

        
        if (m.get_mass() == 0){
            throw std::invalid_argument("Mass cannot be zero");
        }
    }


}