#ifndef PARTICLE_FILTER_HH
#define PARTICLE_FILTER_HH

#include <algorithm>
#include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>
#include <random>

#include "history.hh"
#include "model.hh"
#include "particle.hh"

namespace smcpf {

enum ResamplingStrategy {
  RESAMPLING_NONE = 0,
  RESAMPLING_STRATIFIED,
  RESAMPLING_SYSTEMATIC
};

template <class PT, class OT, size_t N, bool enable_history = false,
          bool parallel = false>
class ParticleFilter {
private:
  // every particle also contains its associated weight
  std::array<Particle<PT>, N> m_particles;

  Model<PT, OT> *m_model;

  ResamplingStrategy m_strategy;
  double m_treshhold;

  History<PT> m_history;

  std::mt19937 m_gen;

public:
  explicit ParticleFilter(
      Model<PT, OT> *t_model,
      ResamplingStrategy t_strategy = ResamplingStrategy::RESAMPLING_SYSTEMATIC,
      double t_treshhold = 0.5)
      : m_model(t_model), m_strategy(t_strategy), m_treshhold(t_treshhold),
        m_gen() {
    // RAII: Create initial set of particles by drawing from the prior so that
    // the particle filter is ready to use after constructing it
    for (auto &particle : m_particles) {
      m_model->sample_prior(particle);
      particle.set_weight(1. / N);
    }

    if constexpr (enable_history) {
      m_history.set_means(mean(), weighted_mean());
      m_history.set_time(-1);
      m_history.flush();
    }
  }

  // disable copying or moving particle filters
  ParticleFilter(const ParticleFilter &) = delete;
  ParticleFilter(const ParticleFilter &&) = delete;
  ~ParticleFilter() = default;

  void normalise_weights() {
    const auto total_weights = std::accumulate(
        m_particles.begin(), m_particles.end(), 0.0,
        [](double acc, const Particle<PT> &p) { return acc + p.get_weight(); });

    const auto scalar_mult = [total_weights](Particle<PT> &particle) {
      particle.set_weight(1. / total_weights * particle.get_weight());
    };

    if constexpr (parallel) {
      std::for_each(std::execution::par_unseq, m_particles.begin(),
                    m_particles.end(), scalar_mult);
    } else {
      std::for_each(std::execution::seq, m_particles.begin(), m_particles.end(),
                    scalar_mult);
    }
  }

  void evolve(OT t_observation, double t_time) {
    const auto transform_weight = [&](Particle<PT> &curr_particle) {
      // update particle by sampling from proposal distribution
      curr_particle.set_value(
          m_model->sample_proposal(curr_particle, t_observation, t_time));

      // update weight according to the function defined in the model
      curr_particle.set_weight(
          curr_particle.get_weight() *
          m_model->update_weight(curr_particle, t_observation, t_time));
    };

    if constexpr (parallel) {
      std::for_each(std::execution::par_unseq, m_particles.begin(),
                    m_particles.end(), transform_weight);
    } else {
      std::for_each(std::execution::seq, m_particles.begin(), m_particles.end(),
                    transform_weight);
    }

    if (resampling_necessary()) resample();

    if constexpr (enable_history) {
      m_history.set_means(mean(), weighted_mean());
      m_history.set_time(t_time);
      m_history.flush();
    }
  }

  bool resampling_necessary() {
    normalise_weights();
    // compute effective sampling size
    double squared_sum =
        std::accumulate(m_particles.begin(), m_particles.end(), 0.0,
                        [](double s, const auto &particle) {
                          const auto weight = particle.get_weight();
                          return s + (weight * weight);
                        });
    // resample only if ess is below treshhold
    return (1. / squared_sum) < m_treshhold * N;
  }

  void resample() {
    std::vector<double> cum_sum_weights(N);
    cum_sum_weights[0] =  m_particles[0].get_weight();
    for (unsigned i = 1; i < N; ++i) {
      cum_sum_weights[i] = cum_sum_weights[i - 1] + m_particles[i].get_weight();
    }

    std::uniform_real_distribution<> dis(0., 1.);
    double u_init = dis(m_gen);

    std::vector<unsigned> indices(N);
    
    unsigned i = 0;
    for (unsigned j = 1; j <= N; j++) {
      auto u = (u_init + j - 1) / N;
      while (u > cum_sum_weights[i])
        ++i;
      indices[j-1] = i;
    }

    auto old_particles = m_particles;
    for (unsigned k=0; k<N; k++) {
      m_particles[k].set_value(old_particles[indices[k]].get_value());
      m_particles[k].set_weight(1. / N);
    }
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
        [](double acc, Particle<PT> p) { return acc + p.get_weight(); });
    return sum / total_weights;
  }

  void set_resampling_strategy(ResamplingStrategy t_strategy) {
    m_strategy = t_strategy;
  }

  void set_resampling_treshhold(double t_treshhold) {
    m_treshhold = t_treshhold;
  }

  template <class PTWriter>
  void write_history(std::ostream &t_out, PTWriter t_writer,
                     char t_separator = ',') {
    m_history.write_all(t_out, t_writer, t_separator);
  }

  Particle<PT> &operator()(unsigned int i) { return m_particles[i]; }
};
} // namespace smcpf

#endif // PARTICLE_FILTER_HH
