#ifndef FUNCTORS_HH
#define FUNCTORS_HH

#define STATS_ENABLE_BLAZE_WRAPPERS
#include <blaze/Blaze.h>
#include <stats.hpp>

typedef blaze::DynamicMatrix<double> matrix;

struct BivaraiteGaussian {
  matrix m_mu{{0}, {0}};
  matrix m_sigma{{1, 0}, {0, 1}};

  BivaraiteGaussian(matrix t_mu, matrix t_sigma)
      : m_mu(t_mu), m_sigma(t_sigma) {}
  BivaraiteGaussian() {}

  matrix operator()() const { return stats::rmvnorm(m_mu, m_sigma); }
};

#endif // FUNCTORS_HH
