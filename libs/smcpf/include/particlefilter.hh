#ifndef PARTICLE_FILTER_HH
#define PARTICLE_FILTER_HH

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <iostream>

#ifdef PF_USE_PARALLEL
#include <execution>
#endif

#include "history.hh"
#include "model.hh"
#include "particle.hh"

namespace smcpf {

enum class ResamplingStrategy { RESAMPLING_NONE, RESAMPLING_SYSTEMATIC };

template <class PT, class OT, size_t N, bool enable_history = false,
          bool parallel = false, typename... Args>
class ParticleFilter {
private:
  // every particle also contains its associated weight
  std::array<Particle<PT>, N> m_particles;

  Model<PT, OT, Args...> *m_model;

  ResamplingStrategy m_strategy;
  double m_treshhold;

  History<PT, Args...> m_history;

  std::mt19937 m_gen; // rng used in resampling

public:
  explicit ParticleFilter(
      Model<PT, OT, Args...> *t_model,
      ResamplingStrategy t_strategy = ResamplingStrategy::RESAMPLING_SYSTEMATIC,
      double t_treshhold = 0.5, double t_seed = 0)
      : m_model(t_model), m_strategy(t_strategy), m_treshhold(t_treshhold),
        m_gen(t_seed) {
    // Create initial set of particles by drawing from the prior so that
    // the particle filter is ready to use after constructing it.
    for (auto &particle : m_particles) {
      m_model->sample_prior(particle);
      particle.set_weight(1. / N);
    }

    if constexpr (enable_history) {
      m_history.set_means(mean(), weighted_mean());
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

    const auto normalise_particle = [total_weights](Particle<PT> &particle) {
      particle.set_weight(1. / total_weights * particle.get_weight());
    };

#ifdef PF_USE_PARELLEL 
    std::for_each(std::execution::par_unseq, m_particles.begin(),
                    m_particles.end(), normalise_particle);
#else
    std::cout << "TEST normalise" << '\n';
      std::for_each(m_particles.begin(), m_particles.end(),
                    normalise_particle);
#endif
  }

  /* Performs one step of the particle filter algorithm. This is the only method
   * that needs to be called at every step. Using the model, a new set of
   * particles is sampled from the proposal and the weights are updated
   * accordingly. Depending on the template value parallel, the particles are
   * evolved in parallel.
   * If the particles should be resampled, this is automatically
   * performed and, depending on the value of the template parameter history,
   * the current sample mean and possibly additional info from the parameter
   * pack ...args is passed to the history object.
   */
  void evolve(OT t_observation, Args... t_args) {
    const auto transform_weight = [&](Particle<PT> &curr_particle) {
      const auto particle_before_sampling = curr_particle;
      // update particle by sampling from proposal distribution
      curr_particle.set_value(
          m_model->sample_proposal(curr_particle, t_observation, t_args...));

      // update weight according to the function defined in the model
      curr_particle.set_weight(
          curr_particle.get_weight() *
          m_model->update_weight(particle_before_sampling, curr_particle,
                                 t_observation, t_args...));
    };

#ifdef PF_USE_PARALLEL
    std::for_each(std::execution::par_unseq, m_particles.begin(),
                    m_particles.end(), transform_weight);
#else
    std::cout << "TEST evolve" << '\n';
      std::for_each(m_particles.begin(), m_particles.end(),
                    transform_weight);
#endif

    if (resampling_necessary()) {
      resample();
      m_history.set_current_resampled();
    }

    if constexpr (enable_history) {
      m_history.set_means(mean(), weighted_mean());
      m_history.set_args(t_args...);
      m_history.flush();
    }
  }

  /* This method computes an estimate of the effective sampling size and
   * returns a boolean specifying whether or not to resample the current set of
   * particles. This can be adjusted with the resampling treshhold m_treshhold.
   */
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

  /* Performs the resampling step. Note that this method does not check
   * whether resampling is necessary. This is checked in the evolve method.
   * Therefore this method does not need to be called explicitly when resampling
   * should be performed.
   */
  void resample() {
    /* NOTE:
     * Currently only systematic resampling is implemented.
     */
    switch (m_strategy) {

    case ResamplingStrategy::RESAMPLING_SYSTEMATIC: {
      std::vector<double> cum_sum_weights(N);
      cum_sum_weights[0] = m_particles[0].get_weight();
      for (unsigned i = 1; i < N; ++i) {
        cum_sum_weights[i] =
            cum_sum_weights[i - 1] + m_particles[i].get_weight();
      }

      std::uniform_real_distribution<> dis(0., 1.);
      double u_init = dis(m_gen);

      std::vector<unsigned> indices(N);

      unsigned i = 0;
      for (unsigned j = 1; j <= N; j++) {
        auto u = (u_init + j - 1) / N;
        while (u > cum_sum_weights[i])
          ++i;
        indices[j - 1] = i;
      }

      auto old_particles = m_particles;
      for (unsigned k = 0; k < N; k++) {
        m_particles[k].set_value(old_particles[indices[k]].get_value());
        m_particles[k].set_weight(1. / N);
      }
      break;
    }
    case ResamplingStrategy::RESAMPLING_NONE:
    default:
      // Do nothing
      break;
    }
  }

  /* Computes the unweighted mean of the current set of particles.
   * This can be useful in testing an debugging since in certain cases
   * (e.g. when no resampling is performed) the particle weights might take
   * values of approximately zero, leading to uniformative results when
   * computing the weighted mean. This mean, however, does not represent
   * the actual mean of the current approximated posterior.
   */
  inline PT mean() const {
    auto sum = m_model->zero_particle();
    for (const auto &particle : m_particles)
      sum += particle.get_value();
    return 1. / N * sum;
  }

  /* Computes the weighted mean of the current set of particles.
   * Similarly, other methods can be implemented to compute approximations
   * of other quantities of interest.
   */
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

  /* This method uses the history object to write some info on the particles
   * (e.g. the sample mean, wheter resampling was performed at some time step,
   * additional info using the Args... argument pack) to the output stream t_out
   * in a csv style format separated using the t_separator. PTWriter and
   * ArgWriter are functors that should return a std::string representation for
   * a particle (PTWriter) and an Args... pack (ArgWriter). For particles taking
   * values in the reals this might simply be the identity but for
   * multi-dimensional particle types some other custom output format can be
   * used.
   */
  template <class PTWriter, class ArgWriter>
  void write_history(std::ostream &t_out, PTWriter t_writer,
                     ArgWriter t_awriter, char t_separator = ',') {
    m_history.write_all(t_out, t_writer, t_awriter, t_separator);
  }

  Particle<PT> &operator()(unsigned int i) const { return m_particles[i]; }
};
} // namespace smcpf

#endif // PARTICLE_FILTER_HH
