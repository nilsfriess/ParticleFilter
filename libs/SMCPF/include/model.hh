#ifndef MODEL_HH
#define MODEL_HH

namespace smcpf {
template <class PT> class Particle;

template <class PT, class OT> class Model {
public:
  virtual void sample_prior(Particle<PT> &) const = 0;

  // should return a particle with value zero (eg. the zero vector, or just zero
  // in 1D)
  virtual PT zero_particle() const = 0;

  virtual double observation_density(const Particle<PT> &t_particle,
                                     const OT &t_observation,
                                     double t_time) const = 0;

  virtual double transition_density(const Particle<PT> &t_particle_prev,
                                    const Particle<PT> &t_particle_curr,
                                    double t_time) const = 0;
};

} // namespace smcpf

#endif // MODEL_HH
