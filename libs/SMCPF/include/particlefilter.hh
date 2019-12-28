#ifndef PARTICLE_FILTER_HH
#define PARTICLE_FILTER_HH

#include <algorithm>
#include <execution>
#include <memory>
#include <numeric>
#include <vector>

namespace smcpf {

template <class PT, class OT> class Model;
template <class PT> class Particle;

enum ResamplingStrategy {
  RESAMPLING_NONE = 0,
  RESAMPLING_STRATIFIED,
  RESAMPLING_SYSTEMATIC
};

// PT: type of single particle (e.g. double, std::vector<double>)
template <class PT, class OT, long int N, class ProposalFunctor,
          bool parallel = false>
class ParticleFilter {
private:
  // every particle also contains its associated weight
  std::array<Particle<PT>, N> m_particles;

  std::unique_ptr<Model<PT, OT>> m_model;

  ProposalFunctor m_proposal;

  ResamplingStrategy m_strategy;
  double m_treshhold;

public:
  ParticleFilter(
      std::unique_ptr<Model<PT, OT>> t_model, ProposalFunctor t_proposal,
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

  void sample_proposal() {
    const auto sample = [&](Particle<PT> &particle) {
      // save current particle state; then update its value
      particle.set_previous_value(particle.get_value());
      particle.set_previous_weight(particle.get_weight());
      particle.set_value(m_proposal());
    };

    if constexpr (parallel) {
      std::for_each(std::execution::par, m_particles.begin(), m_particles.end(),
                    sample);
    } else {
      std::for_each(std::execution::seq, m_particles.begin(), m_particles.end(),
                    sample);
    }
  }

  void normalise_weights() {
    const auto total_weights = std::accumulate(
        m_particles.begin(), m_particles.end(), 0.0,
        [](double sum, const Particle<PT> &p) { return sum + p.get_weight(); });

    const auto scalar_mult = [total_weights](Particle<PT> &particle) {
      particle.set_weight(1. / total_weights * particle.get_weight());
    };

    if constexpr (parallel) {
      std::for_each(std::execution::par, m_particles.begin(), m_particles.end(),
                    scalar_mult);
    } else {
      std::for_each(std::execution::seq, m_particles.begin(), m_particles.end(),
                    scalar_mult);
    }
  }

  void evolve(OT t_observation, double t_time) {
    sample_proposal();

    const auto transform_weight = [&](Particle<PT> &curr_particle) {
      curr_particle.set_weight(
          curr_particle.get_previous().get_weight() *
          m_model->observation_density(curr_particle, t_observation, t_time) *
          m_model->transition_density(curr_particle.get_previous(),
                                      curr_particle, t_time) /
          m_proposal.density(curr_particle.get_value()));
    };

    if constexpr (parallel) {
      std::for_each(std::execution::par, m_particles.begin(), m_particles.end(),
                    transform_weight);
    } else {
      std::for_each(std::execution::seq, m_particles.begin(), m_particles.end(),
                    transform_weight);
    }

    normalise_weights();
  }

  bool resampling_necessary() {
    normalise_weights();
    // compute effective sampling size
    double sum = 0;
    for (const auto &particle : m_particles) {
      sum += particle.get_weight() * particle.get_weight();
    }

    // resample only if ess is below treshhold
    return (1. / sum) < m_treshhold * N;
  }

  // Compute the unweighted mean of the current set of particles
  inline PT mean() const {
    auto sum = m_model->zero_particle();
    for (const auto &particle : m_particles)
      sum += particle.get_value();
    return 1. / N * sum;
  }

  // Compute the weighted mean of the current set of particles
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

  void set_resampling_treshhold(double t_treshhold) {
    m_treshhold = t_treshhold;
  }

  void update_proposal(ProposalFunctor &t_proposal) { m_proposal = t_proposal; }

  Particle<PT> &operator()(unsigned int i) { return m_particles[i]; }
};
} // namespace smcpf

#endif // PARTICLE_FILTER_HH
