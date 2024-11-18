#include "../include/sonne.hpp"

// Update position and velocity based on force
void sonne::update(double dt){
    // Update position and velocity based on force
    for (int i = 0; i < 3; i++){
        m_vel[i] += m_force[i] * dt / m_mass;
        m_pos[i] += m_vel[i] * dt;
    }
}