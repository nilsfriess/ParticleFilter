#ifndef PARTICLE_HH
#define PARTICLE_HH

#include <iostream>
#include <limits>
#include <vector>

namespace smcpf {
template <typename PT> class Particle {
private:
  PT value;
  double weight;

public:
  Particle() : weight(std::numeric_limits<double>::quiet_NaN()) {}

  Particle(PT val, double w) : value(val), weight(w) {}

  void setValue(PT value) { this->value = value; }
  void setWeight(double weight) { this->weight = weight; }

  template <typename T>
  friend std::ostream &operator<<(std::ostream &os, const Particle<T> &p);
};
  
template <typename PT>
std::ostream &operator<<(std::ostream &os, const Particle<PT> &p) {
  return os << p.value(0,0) << ", " << p.value(1,0) << " (" << p.weight << ")";
}
} // namespace smcpf

#endif // PARTICLE_HH
