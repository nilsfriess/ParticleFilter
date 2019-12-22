#ifndef PARTICLE_FILTER_HH
#define PARTICLE_FILTER_HH

#include <memory>
#include <vector>

#include "model.hh"
#include "particle.hh"

namespace smcpf {

enum ResamplingStrategy {
  RESAMPLING_NONE = 0,
  RESAMPLING_STRATIFIED,
  RESAMPLING_SYSTEMATIC
};

// PT: type of single particle (e.g. double, std::vector<double>)
template <typename PT, int N, class ProposalFunctor> class ParticleFilter {
private:
  std::array<Particle<PT>, N>
      particles; // every particle also contains its associated weight

  std::shared_ptr<Model<PT>> model;

  // functor that represents the proposal distribution
  ProposalFunctor proposal;

  ResamplingStrategy rstrat;
  double rTreshhold;

public:
  ParticleFilter(
      std::shared_ptr<Model<PT>> m, ProposalFunctor p,
      ResamplingStrategy strategy = ResamplingStrategy::RESAMPLING_SYSTEMATIC,
      double treshhold = 0.5)
      : model(m), proposal(p), rstrat(strategy), rTreshhold(treshhold) {}

  void createParticles() {
    for (auto &particle : particles) {
      model->samplePrior(particle);
      particle.setWeight(1. / N);
    }
  }

  void setResamplingStrategy(ResamplingStrategy strat) { this->rstrat = strat; }

  void setResamplingTreshhold(double treshhold) {
    this->rTreshhold = treshhold;
  }

  void updateProposal(ProposalFunctor &pf) { this->proposal = pf; }

  Particle<PT> operator()(unsigned int i) { return particles[i]; }
};
} // namespace smcpf

#endif // PARTICLE_FILTER_HH
