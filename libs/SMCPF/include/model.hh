#ifndef MODEL_HH
#define MODEL_HH

#include <vector>

#include "particle.hh"

namespace smcpf {
  template <class PT> class Model {
public:
    virtual void samplePrior(Particle<PT> &) const = 0;
};

} // namespace smcpf

#endif // MODEL_HH
