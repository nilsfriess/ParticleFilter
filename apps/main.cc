#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include <armadillo>

#include "helper.hh"

#include <particlefilter.hh>

#include "functors.hh"
#include "lotkavolterra.hh"

int main() {
  constexpr long int N = 5000;

  typedef smcpf::ParticleFilter<
      LotkaVolterra::ParticleType,    // Type of one particle
      LotkaVolterra::ObservationType, // Type of observations
      N,                              // Number of Particles
      false, false>                     // save history?
      PF;

  // Constructor takes paramaters for the predator prey model
  auto model = LotkaVolterra(7. / 3., 1. / 3., 5. / 3., 1.);

  const auto seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  PF pf(&model, smcpf::ResamplingStrategy::RESAMPLING_SYSTEMATIC, 0.5, seed);

  auto start = std::chrono::high_resolution_clock::now();
  
  while (auto obs = model.next_observation()) {
    // std::cout << "Time = " << obs.value().first
    //           << ", Value = " << obs.value().second << '\n';
    pf.evolve(obs.value().second, obs.value().first);
    // std::cout << pf.weighted_mean()[1] << '\n';
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = finish - start;
  std::cout << "Time = " << duration.count() << std::endl;

  // std::ofstream csv_data;
  // csv_data.open("output.csv");
  // pf.write_history(csv_data, [](const arma::vec &t_particle) {
  //   std::ostringstream ret;
  //   ret << t_particle(0) << ',' << t_particle(1);
  //   return ret.str();
  // });

  // csv_data.close();
  // std::cout << "Mean before = " << pf.mean() << '\n';
  // pf.evolve(6, 0);
  // std::cout << "Mean after  = " << pf.mean() << '\n';

  // std::cout << "Resampling is " << (pf.resampling_necessary() ? "" : "not ")
  // << "necessary" << '\n';
}
