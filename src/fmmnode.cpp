#include "../include/fmmnode.hpp"

fmmnode::fmmnode(){
    this->is_empty = true;
    this->is_leaf = true;
    this->size = 0;
    this->center = {0, 0, 0};
    this->children = std::vector<fmmnode>(8);
}
fmmnode::fmmnode(std::vector<sonne> &bodies){
    this->bodies = bodies;
    
    if (bodies.size() == 0){
        this->is_empty = true;
    }
    this->size = bodies.size();
    this->center = {0, 0, 0};
    for (auto &m : bodies){
        this->center[0] += m.get_position()[0];
        this->center[1] += m.get_position()[1];
        this->center[2] += m.get_position()[2];
    }
    this->center[0] /= this->size;
    this->center[1] /= this->size;
    this->center[2] /= this->size;
    this->children = std::vector<fmmnode>(8);
}
    