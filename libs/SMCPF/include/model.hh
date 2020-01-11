#ifndef MODEL_HH
#define MODEL_HH

#include <optional>

#include "particle.hh"

namespace smcpf {
template <class PT, class OT> class Model {
public:
  virtual void sample_prior(Particle<PT> &) const = 0;

  // should return a particle with value zero
  // (eg. the zero vector, or just zero in 1D)
  virtual PT zero_particle() const = 0;

  // weight update formula
  virtual double update_weight(const Particle<PT> &t_curr_particle,
                               const OT &t_observation,
                               double t_time) const = 0;

  virtual PT sample_proposal(const Particle<PT> &t_curr_particle,
                                 const OT &t_observation,
                                 double t_time) const = 0;

  virtual ~Model() {}
  // t_curr_particle.set_weight(
  //     // see paper for derivation of this update formula
  //     curr_particle.get_previous().get_weight() *
  //     (m_model->observation_density(curr_particle, t_observation, t_time) *
  //      m_model->transition_density(curr_particle.get_previous(),
  //                                  curr_particle, t_time)) /
  //     m_proposal.density(curr_particle.get_value()));

  // virtual double observation_density(const Particle<PT> &t_particle,
  //                                    const OT &t_observation,
  //                                    double t_time) const;

  // virtual double transition_density(const Particle<PT> &t_particle_prev,
  //                                   const Particle<PT> &t_particle_curr,
  //                                   double t_time) const;

  // virtual std::optional<std::pair<double, OT> > next_observation() const;
};

} // namespace smcpf

#endif // MODEL_HH
