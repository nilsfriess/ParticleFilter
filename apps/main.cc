#include <iostream>
#include <memory>

#include "helper.hh"

#include <particlefilter.hh>

#include "functors.hh"
#include "lotkavolterra.hh"

int main() {
  constexpr long int N = 2000; // Number of particles

  typedef blaze::DynamicMatrix<double> matrix;
  typedef ParticleFilter<LotkaVolterra::ParticleType, N, BivaraiteGaussian> PF;

  const matrix mu = {{5}, {2}};
  const matrix sigma = {{1, 0}, {0, 0.5}};
  auto proposal = BivaraiteGaussian(mu, sigma);

  // Constructor takes paramaters for the predator prey model
  LotkaVolterra model(0.5, 2, 1, 0.5);
  PF pf(std::make_unique<LotkaVolterra>(model), proposal);
  pf.create_particles();

  std::cout << pf.weighted_mean() << std::endl;
  std::cout << pf.mean() << std::endl;
}
