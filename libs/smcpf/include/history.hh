#ifndef HISTORY_HH
#define HISTORY_HH

#include <algorithm>
#include <ostream>
#include <tuple>
#include <vector>

namespace smcpf {
template <class PT, typename... Args> class History {
private:
  struct HistoryElement {
    PT m_mean;
    PT m_weighted_mean;
    bool m_was_resampled = false;
    std::tuple<Args...> m_args;
  };

  HistoryElement m_current;
  std::vector<HistoryElement> m_history;

public:
  void set_means(PT t_mean, PT t_weighted_mean) {
    m_current.m_mean = t_mean;
    m_current.m_weighted_mean = t_weighted_mean;
  }

  void set_current_resampled() { m_current.m_was_resampled = true; }

  /* This method appends the current history element to the
   * list of all history elements and clears the state. This
   * is called after the particles are evolved and possibly
   * resampling has been performed.
   */
  void flush() {
    m_history.push_back(m_current);
    m_current = HistoryElement();
  }

  void set_args(Args... t_args) {
    m_current.m_args = std::make_tuple(t_args...);
  }

  /* This methods writes all the save history elements to the ostream t_out
   * using a csv style format with the given seperator. The template parameters
   * PTWriter and ArgWriter are both assumed to be functors that do the
   * following:
   * - PTWriter: Functor that takes a value of type PT and converts it to
   * string.
   * - ArgWriter: Functor that takes a tuple of the variadic arguments and
   * converts them to string.
   * In other words, they convert particles and Args... to std::strings.
   */
  template <class PTWriter, class ArgWriter>
  void write_all(std::ostream &t_out, PTWriter t_writer, ArgWriter t_awriter,
                 char t_separator = ',') {
    std::for_each(m_history.begin(), m_history.end(),
                  [&](const HistoryElement &t_hist) {
                    t_out << t_writer(t_hist.m_mean) << t_separator
                          << t_writer(t_hist.m_weighted_mean) << t_separator
                          << (t_hist.m_was_resampled ? "1" : "0") << t_separator
                          << t_awriter(t_hist.m_args) << '\n';
                  });
  }
};
} // namespace smcpf

#endif // HISTORY_HH
