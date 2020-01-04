#ifndef PARTICLE_FILTER_HH
#define PARTICLE_FILTER_HH

#include <algorithm>
#include <execution>
#include <memory>
#include <numeric>
#include <vector>

#include "history.hh"
#include "model.hh"
#include "particle.hh"

namespace smcpf {

enum ResamplingStrategy {
  RESAMPLING_NONE = 0,
  RESAMPLING_STRATIFIED,
  RESAMPLING_SYSTEMATIC
};

// PT: type of single particle (e.g. double, std::vector<double>)
// OT: type of observation
// N: number of particles
// ProposalFunctor: Functor representing the Proposal distribution TODO: CHANGE
// TO ABSTRACT INTERFACE parallel: indicates if parallel versions of eg.
// std::for_each should be used
template <class PT, class OT, size_t N, class ProposalFunctor,
          bool parallel = false>
class ParticleFilter {
private:
  // every particle also contains its associated weight
  std::array<Particle<PT>, N> m_particles;

  Model<PT, OT> *m_model;

  ProposalFunctor m_proposal;

  ResamplingStrategy m_strategy;
  double m_treshhold;

  bool m_save_history = false;
  History<PT> m_history;

public:
  ParticleFilter(
      Model<PT, OT> *t_model, ProposalFunctor t_proposal,
      ResamplingStrategy t_strategy = ResamplingStrategy::RESAMPLING_SYSTEMATIC,
      double t_treshhold = 0.5)
      : m_model(t_model), m_proposal(t_proposal), m_strategy(t_strategy),
        m_treshhold(t_treshhold) {
    // RAII: Create initial set of particles by drawing from the prior so that
    // the particle filter is ready to use after constructing it
    for (auto &particle : m_particles) {
      m_model->sample_prior(particle);
      particle.set_weight(1. / N);
    }
  }

  // disable copying or moving particle filters
  ParticleFilter(const ParticleFilter&) = delete;
  ParticleFilter(const ParticleFilter&&) = delete;
  ~ParticleFilter() = default;

  void sample_proposal() {
    const auto sample = [&](Particle<PT> &particle) {
      // save current particle state; then update its value
      particle.set_previous_value(particle.get_value());
      particle.set_previous_weight(particle.get_weight());
      particle.set_value(m_proposal());
    };

    if constexpr (parallel) {
      std::for_each(std::execution::par_unseq, m_particles.begin(), m_particles.end(),
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
      std::for_each(std::execution::par_unseq, m_particles.begin(), m_particles.end(),
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
          // see paper for derivation of this update formula
          curr_particle.get_previous().get_weight() *
          m_model->observation_density(curr_particle, t_observation, t_time) *
          m_model->transition_density(curr_particle.get_previous(),
                                      curr_particle, t_time) /
          m_proposal.density(curr_particle.get_value()));
    };

    if constexpr (parallel) {
      std::for_each(std::execution::par_unseq, m_particles.begin(), m_particles.end(),
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

  void resample() {}

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

  // Once history is enabled, it cannot be disabled again
  void enable_history() { m_save_history = true; }

  Particle<PT> &operator()(unsigned int i) { return m_particles[i]; }
};
} // namespace smcpf

#endif // PARTICLE_FILTER_HH
