#ifndef LOTKAVOLTERRA_HH
#define LOTKAVOLTERRA_HH

#include <model.hh>
#include <particle.hh>

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;

#include "functors.hh"

using namespace smcpf;

class LotkaVolterra : public Model<matrix, double> {
private:
  BivaraiteGaussian m_prior; // Functor for sampling from 2D gaussian

  const double m_alpha, m_beta, m_gamma, m_delta;

  const double observation_variance = 0.8;

  typedef boost::array<double, 2> state_type;

public:
  matrix evolve(const Particle<matrix> &t_particle) const {
    // function representing the lotka volterra equations
    const auto f = [=](const state_type &t_x, state_type &t_dxdt,
                       [[maybe_unused]] double t_time) {
      t_dxdt[0] = m_alpha * t_x[0] - m_beta * t_x[0] * t_x[1];
      t_dxdt[1] = m_delta * t_x[0] * t_x[1] - m_gamma * t_x[1];
    };

    state_type x = {t_particle.get_value()(0, 0),
                    t_particle.get_value()(1, 0)}; // initial conditions
    integrate(f, x, 0.0, 1.0, 0.1);
    return matrix{{x[0]}, {x[1]}};
  }

  LotkaVolterra(double t_alpha, double t_beta, double t_gamma, double t_delta)
      : m_alpha(t_alpha), m_beta(t_beta), m_gamma(t_gamma), m_delta(t_delta) {
    // initialise prior
    const matrix mu = {{5}, {2}};
    const matrix sigma = {{1, 0}, {0, 0.5}};
    m_prior = BivaraiteGaussian(mu, sigma);
  }

  inline void sample_prior(Particle<matrix> &t_particle) const override {
    t_particle.set_value(m_prior());
  }

  inline matrix zero_particle() const override { return matrix(2, 1, 0.0); }

  double observation_density(const Particle<matrix> &t_particle,
                             const double &t_observation,
                             [[maybe_unused]] double t_time) const override {
    // Gaussian PDF, centered at the observation and evaluated at the
    // particles predator value
    return stats::dnorm(t_observation, t_particle.get_value()(0, 0),
                        observation_variance);
  }

  double transition_density(const Particle<matrix> &t_particle_prev,
                            const Particle<matrix> &t_particle_curr,
                            [[maybe_unused]] double t_time) const override {
    BivaraiteGaussian transition_kernel({{t_particle_prev.get_value()(0, 0)},
                                         {t_particle_prev.get_value()(1, 0)}},
                                        {{1, 0}, {0, 1}});
    auto evolved_value = evolve(t_particle_curr);
    return transition_kernel.density(evolved_value);
  }

  typedef matrix ParticleType;
};

#endif // LOTKAVOLTERRA_HH
