#ifndef MODEL_HH
#define MODEL_HH

namespace smcpf {
template <typename PT> class Particle;
  
template <typename PT> class Model {
public:
  virtual void sample_prior(Particle<PT> &) const = 0;

  // should return a particle with value zero (eg. the zero vector, or just zero
  // in 1D)
  virtual PT zero_particle() const = 0;
};

} // namespace smcpf

#endif // MODEL_HH
