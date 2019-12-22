#ifndef LOTKAVOLTERRA_HH
#define LOTKAVOLTERRA_HH

#include <model.hh>
#include <particle.hh>

#include "functors.hh"

using namespace smcpf;

class LotkaVolterra : public Model<matrix> {
private:
  BivaraiteGaussian prior; // Functor for sampling from 2D gaussian

  double a, b, c, d;

public:
  LotkaVolterra(double alpha, double beta, double gamma, double delta)
      : a(alpha), b(beta), c(gamma), d(delta) {
    // initialise prior
    const matrix mu = {{5}, {2}};
    const matrix sigma = {{1, 0}, {0, 0.5}};
    prior = BivaraiteGaussian(mu, sigma);
  }

  inline void samplePrior(Particle<matrix> &particle) const override {
    particle.setValue(prior());
  }

  typedef matrix ParticleType;
};

#endif // LOTKAVOLTERRA_HH
