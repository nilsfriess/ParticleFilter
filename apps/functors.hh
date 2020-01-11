#ifndef FUNCTORS_HH
#define FUNCTORS_HH

#define STATS_ENABLE_ARMA_WRAPPERS
#include <armadillo>
#include <stats.hpp>

struct BivariateNormal {
  arma::dvec m_mu{0, 0};
  arma::dmat m_sigma{{1, 0}, {0, 1}};

  BivariateNormal(arma::dvec t_mu, arma::dmat t_sigma)
      : m_mu(t_mu), m_sigma(t_sigma) {}
  BivariateNormal() {}

  arma::dvec operator()() const { return stats::rmvnorm(m_mu, m_sigma); }

  double density(arma::dvec t_x) const {
    return stats::dmvnorm(t_x, m_mu, m_sigma);
  }
};

#endif // FUNCTORS_HH
