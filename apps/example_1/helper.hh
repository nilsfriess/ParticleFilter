#ifndef HELPER_HH
#define HELPER_HH

#include <iostream>
#include <vector>

std::ostream &operator<<(std::ostream &os, const std::vector<double> &v) {
  os << "[";
  for (auto e: v) {
    os << " " << e;
  }
  os << "]";
  return os;
}

#endif // HELPER_HH
