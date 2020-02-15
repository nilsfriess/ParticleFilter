#ifndef HISTORY_HH
#define HISTORY_HH

#include <ostream>
#include <vector>
#include <algorithm>

#include "particle.hh"

namespace smcpf {
template <class PT> class History {
private:
  struct HistoryElement {
    PT m_mean;
    PT m_weighted_mean;
    bool m_was_resampled = false;
    double m_time = 0.0;
  };

  HistoryElement m_current;
  std::vector<HistoryElement> m_history;

public:
  void set_means(PT t_mean, PT t_weighted_mean) {
    m_current.m_mean = t_mean;
    m_current.m_weighted_mean = t_weighted_mean;
  }

  void set_time(double t_time) {
    m_current.m_time = t_time;
  }

  void set_current_resampled() {
    m_current.m_was_resampled = true;
  }

  void flush() {
    m_history.push_back(m_current);
    m_current = HistoryElement();
  }

  template <class PTWriter>
  void write_all(std::ostream &t_out, PTWriter t_writer, char t_separator = ',') {
    // TODO: Write the current history to the output stream in a csv style
    // format using t_separator
    std::for_each(m_history.begin(), m_history.end(),
                  [&](const HistoryElement &t_hist) {
		    t_out << t_hist.m_time << t_separator
			  << t_writer(t_hist.m_mean) << t_separator
			  << t_writer(t_hist.m_weighted_mean) << t_separator
			  << t_hist.m_was_resampled << t_separator
			  << '\n';
                  });
  }
};
} // namespace smcpf

#endif // HISTORY_HH
