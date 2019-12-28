#ifndef HISTORY_HH
#define HISTORY_HH

#include <ostream>
#include <vector>

#include "particle.hh"

namespace smcpf {
template <class PT> class History {
private:
  struct HistoryElement {
    std::vector<Particle<PT>> m_particles;
    PT m_mean;
    PT m_weighted_mean;
    bool m_was_resampled;
  };

  HistoryElement m_current;
  std::vector<HistoryElement> m_history;

public:
  void set_particles(std::vector<Particle<PT>> &t_particles) {
    m_current.m_particles = t_particles;
  }

  void set_means(PT t_mean, PT t_weighted_mean) {
    m_current.m_mean = t_mean;
    m_current.m_weighted_mean = t_weighted_mean;
  }

  void set_resampled(bool t_resampled) {
    m_current.m_was_resampled = t_resampled;
  }

  void flush() {
    m_history.push_back(m_current);
    m_current = HistoryElement();
  }

  void write_all(std::ostream &t_out, char t_separator = ',') {
    // TODO: Write the current history to the output stream in a csv style
    // format using t_separator
  }
};
} // namespace smcpf

#endif // HISTORY_HH
