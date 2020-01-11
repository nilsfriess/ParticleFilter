#include <iostream>
#include <memory>
#include <thread>
#include <sstream>
#include <armadillo>

#include "helper.hh"

#include <particlefilter.hh>

#include "functors.hh"
#include "lotkavolterra.hh"

int main() {
  constexpr long int N = 20;

  typedef ParticleFilter<LotkaVolterra::ParticleType,    // Type of one particle
                         LotkaVolterra::ObservationType, // Type of observations
                         N,                              // Number of Particles
                         true>                           // save history?
      PF;

  // Constructor takes paramaters for the predator prey model
  auto model = LotkaVolterra(5./3., 1./3., 5./3., 1.);
  PF pf(&model);

  while (auto obs = model.next_observation()) {
    std::cout << "Time = " << obs.value().first
               << ", Value = " << obs.value().second << '\n';
    pf.evolve(obs.value().second, obs.value().first);
    std::cout << pf.weighted_mean() << '\n';
    pf.resampling_necessary();
  }

  std::ofstream csv_data;
  csv_data.open("output.csv");
  pf.write_history(csv_data, [](const arma::vec &t_particle){
  				std::ostringstream ret;
  				ret << t_particle(0)
  				    << ','
  				    << t_particle(1);
  				return ret.str();
  			      });

  csv_data.close();
  // std::cout << "Mean before = " << pf.mean() << '\n';
  // pf.evolve(6, 0);
  // std::cout << "Mean after  = " << pf.mean() << '\n';
  
  // std::cout << "Resampling is " << (pf.resampling_necessary() ? "" : "not ") << "necessary" << '\n';
}
