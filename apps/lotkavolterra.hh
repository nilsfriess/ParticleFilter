#ifndef LOTKAVOLTERRA_HH
#define LOTKAVOLTERRA_HH

#include <model.hh>
#include <particle.hh>

#include "functors.hh"

using namespace smcpf;

class LotkaVolterra : public Model<matrix, double> {
private:
  BivaraiteGaussian m_prior; // Functor for sampling from 2D gaussian

  const double m_alpha, m_beta, m_gamma, m_delta;

  const double observation_variance = 0.2;

public:
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

  double observation_density(const Particle<matrix> t_particle,
                             double t_observation,
                             [[maybe_unused]] double t_time) const override {
    // Gaussian PDF, centered at the observation and evaluated at the
    // particles predator value
    const auto diff = std::abs(t_particle.get_value()(0, 0) - t_observation);
    return diff + stats::rnorm(0, observation_variance);
  }

  double transition_density(const Particle<PT> t_particle_prev,
			    const Particle<PT> t_particle_curr,
			    [[maybe_unused]] double t_time) const override {
    
  }

  typedef matrix ParticleType;
};

#endif // LOTKAVOLTERRA_HH
