#include <iostream>
#include <memory>

#include "helper.hh"

#include <particlefilter.hh>

#include "functors.hh"
#include "lotkavolterra.hh"


int main() {
  constexpr int N = 20000; // Number of particles

  typedef blaze::DynamicMatrix<double> matrix;
  typedef ParticleFilter<LotkaVolterra::ParticleType, N, BivaraiteGaussian> PF;

  const matrix mu = {{5}, {2}};
  const matrix sigma = {{1, 0}, {0, 0.5}};
  auto proposal = BivaraiteGaussian(mu, sigma);

  // Constructor takes paramaters for the predator prey model
  auto model = std::make_shared<LotkaVolterra>(0.5, 2, 1, 0.5);
  PF pf(model, proposal);
  pf.createParticles();

  for (int i = 0; i < 100; i++)
    std::cout << pf(i) << std::endl;
}
