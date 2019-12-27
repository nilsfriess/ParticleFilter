#include <iostream>
#include <memory>
#include <thread>

#include "helper.hh"

#include <particlefilter.hh>

#include "functors.hh"
#include "lotkavolterra.hh"

int main() {
  constexpr long int N = 10000;

  typedef blaze::DynamicMatrix<double> matrix;
  typedef ParticleFilter<LotkaVolterra::ParticleType, // Type of one particle
                         double,                      // Type of observations
                         N,                           // Number of Particles
                         BivaraiteGaussian>           // Functor that represents the prior
      PF;

  matrix mu = {{5}, {2}};
  matrix sigma = {{1, 0}, {0, 0.5}};
  auto proposal = BivaraiteGaussian(mu, sigma);

  // Constructor takes paramaters for the predator prey model
  LotkaVolterra model(0.5, 2, 1, 0.5);
  PF pf(std::make_unique<LotkaVolterra>(model), proposal);
  pf.create_particles();

  for (double i = 5.; i < 150; i++) {
    mu = {{i}, {4}};
    sigma = {{1, 0}, {0, 0.5}};
    proposal = BivaraiteGaussian(mu, sigma);

    pf.update_proposal(proposal);

    pf.sample_proposal();

    std::cout << "Value of particle = " << pf(0).get_value()(0,0) << '\n';
    std::cout << "Density =           " << model.observation_density(pf(0), 100, 0) << '\n';
  }

  std::cout << "Resampling is " << (pf.resampling_necessary() ? "" : "not ") << "necessary" << '\n';
}
