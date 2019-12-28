#ifndef PARTICLE_HH
#define PARTICLE_HH

#include <limits>
#include <memory>

namespace smcpf {
template <class PT> class Particle {
private:
  PT m_value;
  double m_weight{std::numeric_limits<double>::quiet_NaN()};

  PT m_prev_value;
  double m_prev_weight;

public:
  Particle() = default;

  Particle(PT t_value, double t_weight)
      : m_value(t_value), m_weight(t_weight) {}

  void set_value(PT t_value) { m_value = t_value; }
  void set_weight(double t_weight) { m_weight = t_weight; }

  PT get_value() const { return m_value; }
  double get_weight() const { return m_weight; }

  Particle<PT> get_previous() const {
    return Particle<PT>(m_prev_value, m_prev_weight);
  }

  void set_previous_value(PT t_val) {
    m_prev_value = t_val;
  }

  void set_previous_weight(double t_weight) {
    m_prev_weight = t_weight;
  }
};
} // namespace smcpf

#endif // PARTICLE_HH
