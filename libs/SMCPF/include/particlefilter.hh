#ifndef PARTICLE_FILTER_HH
#define PARTICLE_FILTER_HH

#include <memory>
#include <numeric>
#include <vector>

namespace smcpf {

template <typename PT> class Model;
template <typename PT> class Particle;

enum ResamplingStrategy {
  RESAMPLING_NONE = 0,
  RESAMPLING_STRATIFIED,
  RESAMPLING_SYSTEMATIC
};

// PT: type of single particle (e.g. double, std::vector<double>)
template <typename PT, long int N, class ProposalFunctor> class ParticleFilter {
private:
  // every particle also contains its associated weight
  std::array<Particle<PT>, N> m_particles;

  std::unique_ptr<Model<PT>> m_model;
  ProposalFunctor m_proposal;

  ResamplingStrategy m_strategy;
  double m_treshhold;

public:
  ParticleFilter(
      std::unique_ptr<Model<PT>> t_model, ProposalFunctor t_proposal,
      ResamplingStrategy t_strategy = ResamplingStrategy::RESAMPLING_SYSTEMATIC,
      double t_treshhold = 0.5)
      : m_model(std::move(t_model)), m_proposal(t_proposal),
        m_strategy(t_strategy), m_treshhold(t_treshhold) {}

  void create_particles() {
    for (auto &particle : m_particles) {
      m_model->sample_prior(particle);
      particle.set_weight(1. / N);
    }
  }

  // Compute the unweighted mean of the current set of particles
  inline PT mean() const {
    auto sum = m_model->zero_particle();
    for (const auto &particle : m_particles)
      sum += particle.get_value();
    return 1. / N * sum;
  }

  // Compute the unweighted mean of the current set of particles
  inline PT weighted_mean() const {
    auto sum = m_model->zero_particle();
    for (const auto &particle : m_particles)
      sum += particle.get_weight() * particle.get_value();
    auto total_weights = std::accumulate(
        m_particles.begin(), m_particles.end(), 0.0,
        [](double sum, Particle<PT> p) { return sum + p.get_weight(); });
    return sum / total_weights;
  }

  void set_resampling_strategy(ResamplingStrategy t_strategy) {
    m_strategy = t_strategy;
  }

  void setResamplingTreshhold(double t_treshhold) { m_treshhold = t_treshhold; }

  void updateProposal(ProposalFunctor &t_proposal) { m_proposal = t_proposal; }

  Particle<PT> operator()(unsigned int i) { return m_particles[i]; }
};
} // namespace smcpf

#endif // PARTICLE_FILTER_HH
