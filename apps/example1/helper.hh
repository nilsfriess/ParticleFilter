#ifndef HELPER_HH
#define HELPER_HH

#include <cmath>

template <typename T> T normal_pdf(T x, T m, T s) {
  constexpr double pi = std::atan(1) * 4;
  constexpr T inv_sqrt_2pi = 1.0 / std::sqrt(2 * pi);

  T a = (x - m) / s;

  return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
}

#endif // HELPER_HH
