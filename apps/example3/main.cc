#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include <armadillo>

#include <particlefilter.hh>

#include "functors.hh"
#include "lotkavolterra.hh"

int main() {
  constexpr long int N = 50;

  typedef smcpf::ParticleFilter<
      LotkaVolterra::ParticleType,    // Type of one particle
      LotkaVolterra::ObservationType, // Type of observations
      N,                              // Number of Particles
      true,                           // Save history?
      false,                          // Run parallel?
      double>                         // Time type
      PF;

  auto model = LotkaVolterra(1., 0.1, 1., 0.1);

  const auto seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  PF pf(&model, smcpf::ResamplingStrategy::RESAMPLING_SYSTEMATIC, 0.5, seed);

  auto start = std::chrono::high_resolution_clock::now();

  while (auto obs = model.next_observation()) {
    pf.evolve(obs.value().second, obs.value().first);
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = finish - start;
  std::cout << "Time = " << duration.count() << std::endl;

  std::ofstream csv_data;
  csv_data.open("output_ex3.csv");

  const auto pt_writer = [](const arma::vec &t_particle) {
    std::ostringstream ret;
    ret << t_particle(0) << ',' << t_particle(1);
    return ret.str();
  };

  const auto arg_writer = [](const std::tuple<double> &t_args) {
    return std::to_string(std::get<0>(t_args));
  };

  pf.write_history(csv_data, pt_writer, arg_writer);

  csv_data.close();
}

