#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>
#include <array>


class sonne{

    private:

    std::array<double, 3> m_pos;
    std::array<double, 3> m_vel;
    std::array<double, 3> m_acc;
    std::array<double,3> m_force;
    double m_mass;
    double m_radius;

    public:
    // Getters
    inline const std::array<double, 3>& get_position() const { return m_pos; }
    inline const std::array<double, 3>& get_velocity() const { return m_vel; }
    inline const std::array<double, 3>& get_acceleration() const { return m_acc; }
    inline const std::array<double, 3>& get_force() const { return m_force; }
    inline double get_mass() const { return m_mass; }
    inline double get_radius() const { return m_radius; }
    friend std::ostream &operator<<(std::ostream &out, const sonne &body);
};

inline std::ostream &operator<<(std::ostream &os, const sonne &body) {
    os << "Position: " << body.m_pos[0] << " " << body.m_pos[1] << " " << body.m_pos[2] << std::endl;
    os << "Velocity: " << body.m_vel[0] << " " << body.m_vel[1] << " " << body.m_vel[2] << std::endl;
    os << "Acceleration: " << body.m_acc[0] << " " << body.m_acc[1] << " " << body.m_acc[2] << std::endl;
    os << "Force: " << body.m_force[0] << " " << body.m_force[1] << " " << body.m_force[2] << std::endl;
    os << "Mass: " << body.m_mass << std::endl;
    os << "Radius: " << body.m_radius << std::endl;

  return os;
}