#include "../include/fmmnode.hpp"

fmmnode::fmmnode(std::vector<sonne> &bodies, double theta, double G, double dt){
    this->bodies = bodies;
    this->theta = theta;
    this->G = G;
    this->dt = dt;
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
    this->is_leaf = true;
    this->is_empty = false;
    this->is_far = false;
    this->is_near = false;
    this->is_intermediate = false;
    this->is_root = false;
    this->is_external = false;
    this->is_internal = false;
    this->is_empty = false;
    this->is_far = false;
    this->is_near = false;
    this->is_intermediate = false;
    this->is_root = false;
    this->is_external = false;
    this->is_internal = false;
    this->is_empty = false;
    this->is_far = false;
    this->is_near = false;
    this->is_intermediate = false;
    this->is_root = false;
    this->is_external = false;
    this->is_internal = false;
    this->is_empty = false;
    this->is_far = false;
    this->is_near = false;
    this->is_intermediate = false;
    this->is_root = false;
    this->is_external = false;
    this->is_internal = false;
    this->is_empty = false;
    this->is_far = false;
    this->is_near = false;
    this->is_intermediate = false;
    this->is_root = false;
    this->is_external = false;
    this->is_internal = false;
    this->is_empty = false;
    this->is_far = false;
    this->is_near = false;