#include "sonne.hpp"
#include <vector>


class fmm{

    private:
    std::vector<sonne> bodies;
    std::vector<double> center;
    std::vector<fmm> children; 
    long size;
    
     

    public:

    void compute_forces(std::vector<sonne> &bodies, double theta, double G, double dt);


}