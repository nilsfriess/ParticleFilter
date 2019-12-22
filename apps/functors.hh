#ifndef FUNCTORS_HH
#define FUNCTORS_HH

#include <vector>

#include <blaze/Blaze.h>

#define STATS_ENABLE_BLAZE_WRAPPERS
#include <stats.hpp>

typedef blaze::DynamicMatrix<double> matrix;

struct BivaraiteGaussian {
  matrix _mu;
  matrix _sigma;

  BivaraiteGaussian(matrix mu, matrix sigma) : _mu(mu), _sigma(sigma) {}

  // Default constructor sets mu to the zero vector and sigma the identity
  // matrix, respectively
  BivaraiteGaussian() : BivaraiteGaussian({{0}, {0}}, {{1, 0}, {0, 1}}) {}

  matrix operator()() const { return stats::rmvnorm(_mu, _sigma); }
};

#endif // FUNCTORS_HH
