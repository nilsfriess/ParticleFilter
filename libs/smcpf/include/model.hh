#ifndef MODEL_HH
#define MODEL_HH

#include "particle.hh"

/* This abstract base class can be used to implement a model for
 * the particle filter. The four methods declared below must be
 * implemented by every model that derives from this class as
 * they automatically get called by the particle filter.
 */
namespace smcpf {
template <class PT, class OT, typename... Args> class Model {
public:
  /* This method should sample from the models prior and store
   * the sampled value in the given particle using t_particle.set_value(...).
   */
  virtual void sample_prior(Particle<PT> &t_particle) = 0;

  /* This method should return a particle with value zero (eg. the zero vector,
   * or just zero in 1D)
   */
  virtual PT zero_particle() = 0;

  /* This method does *not* alter the particle itself, it rather should return
   * the number which the weight of the particle gets multiplied with.
   */
  virtual double update_weight(const Particle<PT> &t_particle_before_sampling,
                               const Particle<PT> &t_particle_after_sampling,
                               const OT &t_observation, Args... t_args) = 0;

  /* This method also does *not* alter the particle. It shoudl return the
   * sampled value only.
   */
  virtual PT sample_proposal(const Particle<PT> &t_particle,
                             const OT &t_observation, Args... t_args) = 0;

  virtual ~Model() {}
};

} // namespace smcpf

#endif // MODEL_HH
