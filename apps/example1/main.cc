#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>

#include <boost/math/distributions/normal.hpp>
using boost::math::normal;
using boost::math::pdf;

#include <model.hh>
#include <particle.hh>
#include <particlefilter.hh>

using smcpf::Model;
using smcpf::Particle;

/* The particles and observations are real numbers, ie. doubles.
 * There is one additional argument, namely the index of the observations, ie.
 * an int.
 */
class ExampleModel : public Model<double, double, int> {
private:
  std::mt19937 m_gen;
  std::normal_distribution<> m_prior{0, 0.5};

  std::deque<std::pair<double, double>> m_observations;

  /* This method loads the observations from a csv file that should
   * contain two columns: the number of the observation and the actual value of
   * the observation.
   */
  void load_observations(const std::string &t_filename) {
    const auto read_line = [](std::istream &str) {
      std::vector<double> result;
      std::string line;
      std::getline(str, line);

      std::stringstream lineStream(line);
      std::string cell;

      while (std::getline(lineStream, cell, ',')) {
        result.push_back(std::stod(cell));
      }
      return result;
    };

    std::ifstream file(t_filename);
    while (true) {
      auto line = read_line(file);
      if (line.empty())
        break;
      m_observations.push_back(std::make_pair(line[0], line[1]));
    }
  }

public:
  ExampleModel() {
    std::random_device rd{};
    m_gen = std::mt19937{rd()};

    load_observations("obs_ex1.csv");
  }

  virtual double zero_particle() override { return 0.0; }

  virtual void sample_prior([
      [maybe_unused]] Particle<double> &t_particle) override {
    t_particle.set_value(m_prior(m_gen));
  }

  /* The weight update method. This model uses a bootstrap filter, ie.
   * the weight update equation is simply the old weight times the
   * pdf p(y | x), evaluated at the current observation. In this specific case
   * the pdf is a Gaussian, centered at x^2 / 20 with variance 1, where x is
   * the current particle's value.
   */
  virtual double update_weight(
      [[maybe_unused]] const Particle<double> &t_particle_before_sampling,
      const Particle<double> &t_particle_after_sampling,
      const double &t_observation, [[maybe_unused]] int t_step) override {
    auto pval = t_particle_after_sampling.get_value();
    auto mean = (pval * pval) / 20.0;
    auto var = 1.0;

    normal n{mean, var};
    return pdf(n, t_observation);
  }

  /* This method samples from the proposal. In this case, the proposal is the
   * system's prior, i.e. x ~ p(x | x'), x' being the previous state. p(x | x')
   * is again a Gaussian, now centered at 1/2 * x' + (25 * x')/ (1 + x'^2) + 8
   * cos(1.2 + k) with variance 10, where k is the index of the current
   * observation.
   */
  virtual double sample_proposal(const Particle<double> &t_particle,
                                 [[maybe_unused]] const double &t_observation,
                                 int t_step) override {
    auto pval = t_particle.get_value();
    auto mean = pval / 2.0 + (25 * pval) / (1 + pval * pval) +
                8 * std::cos(1.2 * (t_step));
    auto var = 10.0;

    std::normal_distribution<> proposal{mean, var};
    return proposal(m_gen);
  }

  /*
   * This method is used to simulate that the observations are only available
   * sequentially. Everytime it gets called, it returns a new observation from
   * the previously read observations. If all observations have been processed,
   * the method returns {}, i.e. an empty std::option.
   */
  std::optional<std::pair<double, double>> next_observation() {
    if (m_observations.empty()) {
      return {};
    } else {
      auto obs = m_observations.front();
      m_observations.pop_front();
      return obs;
    }
  }
};

int main() {
  constexpr long int N = 400;
  const auto seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  typedef smcpf::ParticleFilter<double, // Type of particle (Real number)
                                double, // Type of observation (Real number)
                                N,      // Number of particles
                                true,   // Enable history
                                false,  // Disable parallel algorithms
                                int>    // index of observation
      PF;

  ExampleModel model;

  /* Setup particle filter to use systematic resampling and resample when
   * effective sampling size is less than 50% of N.
   */
  PF pf(&model, smcpf::ResamplingStrategy::RESAMPLING_SYSTEMATIC, 0.8, seed);

  // Here, the actual particle filtering is performed
  while (auto obs = model.next_observation()) {
    pf.evolve(obs.value().second, obs.value().first);
  }

  /* The results are written to a .csv file and can be plotted using e.g.
   * gnuplot
   */
  std::ofstream csv_data;
  csv_data.open("output_ex1.csv");

  const auto pt_writer = [](const double &t_particle) {
    return std::to_string(t_particle);
  };

  const auto arg_writer = [](const std::tuple<int> &t_args) {
    return std::to_string(std::get<0>(t_args));
  };

  pf.write_history(csv_data, pt_writer, arg_writer);

  csv_data.close();
}
