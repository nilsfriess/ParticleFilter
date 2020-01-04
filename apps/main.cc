#include <iostream>
#include <memory>
#include <thread>

#include "helper.hh"

#include <particlefilter.hh>

#include "functors.hh"
#include "lotkavolterra.hh"

int main() {
  constexpr long int N = 300;

  typedef blaze::DynamicMatrix<double> matrix;
  typedef ParticleFilter<LotkaVolterra::ParticleType, // Type of one particle
                         double,                      // Type of observations
                         N,                           // Number of Particles
                         BivaraiteGaussian,           // Functor that represents the prior
			 true>                       // should run parallel or not
    PF;

  matrix mu = {{5}, {8}};
  matrix sigma = {{0.1, 0}, {0, 0.1}};
  auto proposal = BivaraiteGaussian(mu, sigma);

  // Constructor takes paramaters for the predator prey model
  auto model = LotkaVolterra(7./3., 1./3., 5./3., 1.);
  PF pf(&model, proposal);
  pf.enable_history();
  
  while(auto obs = model.next_observation()) {
    std::cout << "Time = " << obs.value().first << ", Value = " << obs.value().second << '\n';
    pf.evolve(obs.value().second, obs.value().first);
    std::cout << pf.weighted_mean() << '\n';
  }

  // std::cout << "Mean before = " << pf.mean() << '\n';
  // pf.evolve(6, 0);
  // std::cout << "Mean after  = " << pf.mean() << '\n';
  
  // std::cout << "Resampling is " << (pf.resampling_necessary() ? "" : "not ") << "necessary" << '\n';
}
