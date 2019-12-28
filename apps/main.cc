#include <iostream>
#include <memory>
#include <thread>

#include "helper.hh"

#include <particlefilter.hh>

#include "functors.hh"
#include "lotkavolterra.hh"

int main() {
  constexpr long int N = 1000;

  typedef blaze::DynamicMatrix<double> matrix;
  typedef ParticleFilter<LotkaVolterra::ParticleType, // Type of one particle
                         double,                      // Type of observations
                         N,                           // Number of Particles
                         BivaraiteGaussian,           // Functor that represents the prior
			 true>                        // should run parallel or not
    PF;

  matrix mu = {{5}, {8}};
  matrix sigma = {{1, 0}, {0, 0.5}};
  auto proposal = BivaraiteGaussian(mu, sigma);

  // Constructor takes paramaters for the predator prey model
  PF pf(std::make_unique<LotkaVolterra>(0.5, 2, 1, 0.5), proposal);
  pf.create_particles();
  pf.enable_history();

  std::cout << "Mean before = " << pf.weighted_mean() << '\n';
  pf.evolve(6, 0);
  std::cout << "Mean after  = " << pf.weighted_mean() << '\n';
  
  std::cout << "Resampling is " << (pf.resampling_necessary() ? "" : "not ") << "necessary" << '\n';
}
