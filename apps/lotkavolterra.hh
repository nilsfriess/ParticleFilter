#ifndef LOTKAVOLTERRA_HH
#define LOTKAVOLTERRA_HH

#include <deque>
#include <fstream>
#include <string>

#include <model.hh>
#include <particle.hh>

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;

#include "functors.hh"

using namespace smcpf;

class LotkaVolterra : public Model<arma::dvec, double> {
private:
  const double m_alpha, m_beta, m_gamma, m_delta;

  BivariateNormal m_prior;

  // Constants for computing the proposal
  const arma::dmat m_sys_cov{{0.5, 0}, {0, 0.5}};
  const double m_obs_cov{1};
  const arma::dmat m_obs_mat{{1, 0}};

  const arma::dmat m_model_cov =
      arma::inv(m_sys_cov.i() + m_obs_mat.t() * (1. / m_obs_cov * m_obs_mat));

  std::deque<std::pair<double, double>> m_observations;

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

  arma::dvec evolve(const Particle<arma::dvec> &t_particle) const {
    // function representing the lotka volterra equations
    typedef boost::array<double, 2> state_type;
    const auto f = [=](const state_type &t_x, state_type &t_dxdt,
                       [[maybe_unused]] double t_time) {
      t_dxdt[0] = m_alpha * t_x[0] - m_beta * t_x[0] * t_x[1];
      t_dxdt[1] = m_delta * t_x[0] * t_x[1] - m_gamma * t_x[1];
    };

    state_type x = {t_particle.get_value()(0, 0),
                    t_particle.get_value()(1, 0)}; // initial conditions
    integrate(f, x, 0.0, 1.0, 0.1);
    return {x[0], x[1]};
  }

  double extract_predator(const Particle<arma::dvec> &t_particle) const {
    return t_particle.get_value()(0, 0);
  }

public:
  LotkaVolterra(double t_alpha, double t_beta, double t_gamma, double t_delta)
      : m_alpha(t_alpha), m_beta(t_beta), m_gamma(t_gamma), m_delta(t_delta) {
    // initialise prior
    const arma::dvec mu{{5}, {2}};
    const arma::dmat sigma{{1, 0}, {0, 0.5}};
    m_prior = BivariateNormal(mu, sigma);
    load_observations("obs.csv");
  }

  virtual void sample_prior(Particle<arma::dvec> &t_particle) const override {
    t_particle.set_value(m_prior());
  };

  inline arma::dvec zero_particle() const override { return {0, 0}; }

  // weight update formula
  double update_weight(const Particle<arma::dvec> &t_curr_particle,
                       const double &t_observation,
                       [[maybe_unused]]double t_time) const override {
    const arma::dvec mu = m_obs_mat * evolve(t_curr_particle);
    const arma::dmat sigma = m_obs_mat*m_sys_cov*m_obs_mat.t() + m_obs_cov;

    return stats::dnorm(t_observation, mu(0), sigma(0,0));
  }

  arma::dvec sample_proposal(const Particle<arma::dvec> &t_curr_particle,
                             const double &t_observation,
                             [[maybe_unused]] double t_time) const override {
    const auto mu_model =
        m_model_cov * (m_sys_cov.i() * evolve(t_curr_particle) +
                       m_obs_mat.t() * (1. / m_obs_cov * t_observation));
    return BivariateNormal(mu_model, m_model_cov)();
  }

  std::optional<std::pair<double, double>> next_observation() {
    if (m_observations.empty()) {
      return {};
    } else {
      auto obs = m_observations.front();
      m_observations.pop_front();
      return obs;
    }
  }

  typedef arma::dvec ParticleType;
  typedef double ObservationType;
};

#endif // LOTKAVOLTERRA_HH
