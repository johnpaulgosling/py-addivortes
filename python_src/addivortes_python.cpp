// Hot-path optimisations aligned with R AddiVortes 0.7.1:
// 1. Incremental cell reassignment with cached winning distance keys
// 2. Active-dimension-only Euclidean distance (row-major centres)
// 3. Specialised all-Euclidean assign path
// 4. Single-pass residual aggregation reused for MH and mu redraw
// 5. Preallocated scratch buffers outside the j/iter loops
// 6. Deferred posterior packaging (compact C++ store, Python lists at end)
// 7. Same NN kernel + flattened posterior traversal for predict
// 8. Binary-column masks and precomputed categorical column maps
//
// Python adaptations vs R:
// - Centres are row-major (nC x d): centres[c * d + di]
// - Dimension indices are 0-based
// - MH uses selection_prob / log_structure_and_selection
// - RNG is mt19937_64

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

bool in_vector(int value, const std::vector<int>& values) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

double period_shift(double value, double limit) {
  while (value >= limit) {
    value -= 2.0 * limit;
  }
  while (value < -limit) {
    value += 2.0 * limit;
  }
  return value;
}

bool is_last_member_column(int index, const std::vector<int>& members) {
  return index == static_cast<int>(members.size()) - 1 || members[index + 1] != members[index];
}

double uniform01(std::mt19937_64& rng) {
  static constexpr double min_open = std::numeric_limits<double>::min();
  std::uniform_real_distribution<double> dist(min_open, 1.0);
  return dist(rng);
}

bool all_euclidean_metric(const std::vector<int>& metric) {
  for (int m : metric) {
    if (m != 0) {
      return false;
    }
  }
  return true;
}

struct ReducedMetric {
  std::vector<int> metric;
  std::vector<int> member_counts;
};

ReducedMetric make_reduced_metric(const std::vector<int>& metric, const std::vector<int>& members) {
  if (metric.size() != members.size()) {
    throw std::invalid_argument("metric and members must have the same length.");
  }

  ReducedMetric reduced;
  int idx = 0;
  while (idx < static_cast<int>(members.size())) {
    const int current_member = members[idx];
    int count = 0;
    while (idx + count < static_cast<int>(members.size()) && members[idx + count] == current_member) {
      ++count;
    }
    reduced.metric.push_back(metric[idx]);
    reduced.member_counts.push_back(count);
    idx += count;
  }
  return reduced;
}

std::vector<int> compute_ncats_from_data(const double* x,
                                         int n,
                                         int p,
                                         const std::vector<int>& metric) {
  std::vector<int> ncats;
  for (int col = 0; col < p; ++col) {
    if (metric[col] != 2) {
      continue;
    }
    int max_val = 0;
    for (int row = 0; row < n; ++row) {
      const double value = x[static_cast<size_t>(row) * p + col];
      if (value > static_cast<double>(max_val)) {
        max_val = static_cast<int>(std::llround(value));
      }
    }
    if (max_val <= 0) {
      throw std::invalid_argument("Categorical columns must use positive integer codes.");
    }
    ncats.push_back(max_val);
  }
  return ncats;
}

// ---------------------------------------------------------------------------
// Distance kernels
// ---------------------------------------------------------------------------

double euclidean_distance(const double* first, const double* second, int size) {
  double total = 0.0;
  for (int idx = 0; idx < size; ++idx) {
    const double diff = first[idx] - second[idx];
    total += diff * diff;
  }
  return total;
}

double spherical_distance(const double* first, const double* second, int size) {
  if (size == 1) {
    const double a1 = std::abs(first[0] - second[0]);
    const double a2 = 2.0 * kPi - a1;
    return std::min(a1, a2) * std::min(a1, a2);
  }

  double angle_diff = std::cos(first[size - 1] - second[size - 1]);
  for (int idx = size - 2; idx >= 0; --idx) {
    double internal = std::sin(first[idx]) * std::sin(second[idx]) +
                      std::cos(first[idx]) * std::cos(second[idx]) * angle_diff;
    internal = std::clamp(internal, -1.0, 1.0);
    angle_diff = (idx == 0) ? std::acos(internal) : internal;
  }
  return angle_diff * angle_diff;
}

// Eskin distance (Eskin et al., 2002): for each mismatched category add 2 / n_c^2.
double categorical_distance(const double* first,
                            const double* second,
                            int size,
                            const std::vector<int>& ncats,
                            int cat_offset) {
  if (cat_offset + size > static_cast<int>(ncats.size())) {
    throw std::invalid_argument("Categorical group size exceeds ncats length.");
  }
  double total = 0.0;
  for (int idx = 0; idx < size; ++idx) {
    const double left = first[idx];
    const double right = second[idx];
    if (std::floor(left) != left || std::floor(right) != right) {
      throw std::invalid_argument("Categorical coordinates must be integer-valued.");
    }
    if (left != right) {
      const double n_cat = static_cast<double>(ncats[cat_offset + idx]);
      if (n_cat <= 0.0) {
        throw std::invalid_argument("ncats entries must be positive.");
      }
      total += 2.0 / (n_cat * n_cat);
    }
  }
  return total;
}

double calc_distance(const double* first,
                     const double* second,
                     int /*p*/,
                     const std::vector<int>& member_counts,
                     const std::vector<int>& metric,
                     const std::vector<int>& ncats) {
  int offset = 0;
  int cat_offset = 0;
  double total = 0.0;
  for (int group = 0; group < static_cast<int>(member_counts.size()); ++group) {
    const int size = member_counts[group];
    if (metric[group] == 0) {
      total += euclidean_distance(first + offset, second + offset, size);
    } else if (metric[group] == 1) {
      total += spherical_distance(first + offset, second + offset, size);
    } else if (metric[group] == 2) {
      total += categorical_distance(first + offset, second + offset, size, ncats, cat_offset);
      cat_offset += size;
    } else {
      throw std::invalid_argument("Unsupported metric type in distance calculation.");
    }
    offset += size;
  }
  return total;
}

// ---------------------------------------------------------------------------
// Assignment cache + incremental reassignment
// ---------------------------------------------------------------------------

enum class AssignmentDelta {
  CentreAdded,
  CentreRemoved,
  CentreMoved,
  FullRecompute
};

struct AssignmentCache {
  std::vector<int> assignment;    // 0-based centre index per observation
  std::vector<double> best_keys;  // winning comparison key per observation
};

// Row-major centres: centres[c * d + di]
static inline double euclidean_key_active(const double* active,
                                         const double* centres,
                                         int /*nC*/,
                                         int c,
                                         int d) {
  double key = 0.0;
  const double* centre_row = centres + static_cast<size_t>(c) * static_cast<size_t>(d);
  for (int di = 0; di < d; ++di) {
    const double diff = active[di] - centre_row[di];
    key += diff * diff;
  }
  return key;
}

static void assign_full_euclidean(const double* x_row,
                                  int n,
                                  int p,
                                  const double* centres,
                                  int nC,
                                  int d,
                                  const int* dim0,
                                  AssignmentCache& out,
                                  std::vector<double>& active_scratch) {
  out.assignment.resize(n);
  out.best_keys.resize(n);
  active_scratch.resize(d);
  for (int obs = 0; obs < n; ++obs) {
    const double* row = x_row + static_cast<size_t>(obs) * p;
    for (int di = 0; di < d; ++di) {
      active_scratch[di] = row[dim0[di]];
    }
    double best = std::numeric_limits<double>::infinity();
    int best_c = 0;
    for (int c = 0; c < nC; ++c) {
      const double key = euclidean_key_active(active_scratch.data(), centres, nC, c, d);
      if (key < best) {
        best = key;
        best_c = c;
      }
    }
    out.assignment[obs] = best_c;
    out.best_keys[obs] = best;
  }
}

static void reassign_added_euclidean(const double* x_row,
                                     int n,
                                     int p,
                                     const double* centres,
                                     int nC,
                                     int d,
                                     const int* dim0,
                                     const AssignmentCache& prev,
                                     AssignmentCache& out,
                                     std::vector<double>& active_scratch) {
  out.assignment = prev.assignment;
  out.best_keys = prev.best_keys;
  active_scratch.resize(d);
  const int added = nC - 1;
  for (int obs = 0; obs < n; ++obs) {
    const double* row = x_row + static_cast<size_t>(obs) * p;
    for (int di = 0; di < d; ++di) {
      active_scratch[di] = row[dim0[di]];
    }
    const double key = euclidean_key_active(active_scratch.data(), centres, nC, added, d);
    if (key < out.best_keys[obs]) {
      out.best_keys[obs] = key;
      out.assignment[obs] = added;
    }
  }
}

static void reassign_moved_euclidean(const double* x_row,
                                     int n,
                                     int p,
                                     const double* centres,
                                     int nC,
                                     int d,
                                     const int* dim0,
                                     int moved,
                                     const AssignmentCache& prev,
                                     AssignmentCache& out,
                                     std::vector<double>& active_scratch) {
  out.assignment = prev.assignment;
  out.best_keys = prev.best_keys;
  active_scratch.resize(d);
  for (int obs = 0; obs < n; ++obs) {
    const double* row = x_row + static_cast<size_t>(obs) * p;
    for (int di = 0; di < d; ++di) {
      active_scratch[di] = row[dim0[di]];
    }
    const int incumbent = prev.assignment[obs];
    if (incumbent == moved) {
      double best = std::numeric_limits<double>::infinity();
      int best_c = 0;
      for (int c = 0; c < nC; ++c) {
        const double key = euclidean_key_active(active_scratch.data(), centres, nC, c, d);
        if (key < best) {
          best = key;
          best_c = c;
        }
      }
      out.assignment[obs] = best_c;
      out.best_keys[obs] = best;
    } else {
      const double key = euclidean_key_active(active_scratch.data(), centres, nC, moved, d);
      if (key < out.best_keys[obs] || (key == out.best_keys[obs] && moved < incumbent)) {
        out.best_keys[obs] = key;
        out.assignment[obs] = moved;
      }
    }
  }
}

static void reassign_removed_euclidean(const double* x_row,
                                       int n,
                                       int p,
                                       const double* centres,
                                       int nC,
                                       int d,
                                       const int* dim0,
                                       int removed,
                                       const AssignmentCache& prev,
                                       AssignmentCache& out,
                                       std::vector<double>& active_scratch) {
  out.assignment.resize(n);
  out.best_keys.resize(n);
  active_scratch.resize(d);
  for (int obs = 0; obs < n; ++obs) {
    const int incumbent = prev.assignment[obs];
    if (incumbent == removed) {
      const double* row = x_row + static_cast<size_t>(obs) * p;
      for (int di = 0; di < d; ++di) {
        active_scratch[di] = row[dim0[di]];
      }
      double best = std::numeric_limits<double>::infinity();
      int best_c = 0;
      for (int c = 0; c < nC; ++c) {
        const double key = euclidean_key_active(active_scratch.data(), centres, nC, c, d);
        if (key < best) {
          best = key;
          best_c = c;
        }
      }
      out.assignment[obs] = best_c;
      out.best_keys[obs] = best;
    } else {
      out.assignment[obs] = (incumbent > removed) ? incumbent - 1 : incumbent;
      out.best_keys[obs] = prev.best_keys[obs];
    }
  }
}

static void assign_full_general(const double* x_row,
                                int n,
                                int p,
                                const double* centres,
                                int nC,
                                int d,
                                const int* dim0,
                                const std::vector<int>& metric,
                                const std::vector<int>& members,
                                const std::vector<int>& cats,
                                AssignmentCache& out,
                                std::vector<double>& synth) {
  out.assignment.resize(n);
  out.best_keys.resize(n);
  synth.resize(p);
  for (int obs = 0; obs < n; ++obs) {
    const double* row = x_row + static_cast<size_t>(obs) * p;
    std::memcpy(synth.data(), row, static_cast<size_t>(p) * sizeof(double));
    double best = std::numeric_limits<double>::infinity();
    int best_c = 0;
    for (int c = 0; c < nC; ++c) {
      for (int di = 0; di < d; ++di) {
        synth[dim0[di]] = centres[static_cast<size_t>(c) * d + di];
      }
      const double key = calc_distance(row, synth.data(), p, members, metric, cats);
      if (key < best) {
        best = key;
        best_c = c;
      }
    }
    out.assignment[obs] = best_c;
    out.best_keys[obs] = best;
  }
}

static void reassign_added_general(const double* x_row,
                                   int n,
                                   int p,
                                   const double* centres,
                                   int nC,
                                   int d,
                                   const int* dim0,
                                   const std::vector<int>& metric,
                                   const std::vector<int>& members,
                                   const std::vector<int>& cats,
                                   const AssignmentCache& prev,
                                   AssignmentCache& out,
                                   std::vector<double>& synth) {
  out.assignment = prev.assignment;
  out.best_keys = prev.best_keys;
  synth.resize(p);
  const int added = nC - 1;
  for (int obs = 0; obs < n; ++obs) {
    const double* row = x_row + static_cast<size_t>(obs) * p;
    std::memcpy(synth.data(), row, static_cast<size_t>(p) * sizeof(double));
    for (int di = 0; di < d; ++di) {
      synth[dim0[di]] = centres[static_cast<size_t>(added) * d + di];
    }
    const double key = calc_distance(row, synth.data(), p, members, metric, cats);
    if (key < out.best_keys[obs]) {
      out.best_keys[obs] = key;
      out.assignment[obs] = added;
    }
  }
}

static void reassign_moved_general(const double* x_row,
                                   int n,
                                   int p,
                                   const double* centres,
                                   int nC,
                                   int d,
                                   const int* dim0,
                                   int moved,
                                   const std::vector<int>& metric,
                                   const std::vector<int>& members,
                                   const std::vector<int>& cats,
                                   const AssignmentCache& prev,
                                   AssignmentCache& out,
                                   std::vector<double>& synth) {
  out.assignment = prev.assignment;
  out.best_keys = prev.best_keys;
  synth.resize(p);
  for (int obs = 0; obs < n; ++obs) {
    const double* row = x_row + static_cast<size_t>(obs) * p;
    std::memcpy(synth.data(), row, static_cast<size_t>(p) * sizeof(double));
    const int incumbent = prev.assignment[obs];
    if (incumbent == moved) {
      double best = std::numeric_limits<double>::infinity();
      int best_c = 0;
      for (int c = 0; c < nC; ++c) {
        for (int di = 0; di < d; ++di) {
          synth[dim0[di]] = centres[static_cast<size_t>(c) * d + di];
        }
        const double key = calc_distance(row, synth.data(), p, members, metric, cats);
        if (key < best) {
          best = key;
          best_c = c;
        }
      }
      out.assignment[obs] = best_c;
      out.best_keys[obs] = best;
    } else {
      for (int di = 0; di < d; ++di) {
        synth[dim0[di]] = centres[static_cast<size_t>(moved) * d + di];
      }
      const double key = calc_distance(row, synth.data(), p, members, metric, cats);
      if (key < out.best_keys[obs] || (key == out.best_keys[obs] && moved < incumbent)) {
        out.best_keys[obs] = key;
        out.assignment[obs] = moved;
      }
    }
  }
}

static void reassign_removed_general(const double* x_row,
                                     int n,
                                     int p,
                                     const double* centres,
                                     int nC,
                                     int d,
                                     const int* dim0,
                                     int removed,
                                     const std::vector<int>& metric,
                                     const std::vector<int>& members,
                                     const std::vector<int>& cats,
                                     const AssignmentCache& prev,
                                     AssignmentCache& out,
                                     std::vector<double>& synth) {
  out.assignment.resize(n);
  out.best_keys.resize(n);
  synth.resize(p);
  for (int obs = 0; obs < n; ++obs) {
    const int incumbent = prev.assignment[obs];
    if (incumbent == removed) {
      const double* row = x_row + static_cast<size_t>(obs) * p;
      std::memcpy(synth.data(), row, static_cast<size_t>(p) * sizeof(double));
      double best = std::numeric_limits<double>::infinity();
      int best_c = 0;
      for (int c = 0; c < nC; ++c) {
        for (int di = 0; di < d; ++di) {
          synth[dim0[di]] = centres[static_cast<size_t>(c) * d + di];
        }
        const double key = calc_distance(row, synth.data(), p, members, metric, cats);
        if (key < best) {
          best = key;
          best_c = c;
        }
      }
      out.assignment[obs] = best_c;
      out.best_keys[obs] = best;
    } else {
      out.assignment[obs] = (incumbent > removed) ? incumbent - 1 : incumbent;
      out.best_keys[obs] = prev.best_keys[obs];
    }
  }
}

struct AssignScratch {
  std::vector<double> active;
  std::vector<double> synth;
  std::vector<int> dim0;
};

static void reassign(const double* x_row,
                     int n,
                     int p,
                     const double* centres,
                     int nC,
                     int d,
                     const std::vector<int>& dim,  // 0-based
                     AssignmentDelta delta,
                     int touched,
                     bool euclidean,
                     const std::vector<int>& metric,
                     const std::vector<int>& members,
                     const std::vector<int>& cats,
                     const AssignmentCache& prev,
                     AssignmentCache& out,
                     AssignScratch& scratch) {
  scratch.dim0.resize(d);
  for (int i = 0; i < d; ++i) {
    scratch.dim0[i] = dim[i];
  }

  const bool cold = prev.assignment.size() != static_cast<size_t>(n) ||
                    prev.best_keys.size() != static_cast<size_t>(n);
  if (cold || delta == AssignmentDelta::FullRecompute) {
    if (euclidean) {
      assign_full_euclidean(x_row, n, p, centres, nC, d, scratch.dim0.data(), out, scratch.active);
    } else {
      assign_full_general(x_row, n, p, centres, nC, d, scratch.dim0.data(), metric, members, cats,
                          out, scratch.synth);
    }
    return;
  }

  if (euclidean) {
    switch (delta) {
      case AssignmentDelta::CentreAdded:
        reassign_added_euclidean(x_row, n, p, centres, nC, d, scratch.dim0.data(), prev, out,
                                 scratch.active);
        break;
      case AssignmentDelta::CentreMoved:
        reassign_moved_euclidean(x_row, n, p, centres, nC, d, scratch.dim0.data(), touched, prev,
                                 out, scratch.active);
        break;
      case AssignmentDelta::CentreRemoved:
        reassign_removed_euclidean(x_row, n, p, centres, nC, d, scratch.dim0.data(), touched, prev,
                                   out, scratch.active);
        break;
      case AssignmentDelta::FullRecompute:
        assign_full_euclidean(x_row, n, p, centres, nC, d, scratch.dim0.data(), out, scratch.active);
        break;
    }
  } else {
    switch (delta) {
      case AssignmentDelta::CentreAdded:
        reassign_added_general(x_row, n, p, centres, nC, d, scratch.dim0.data(), metric, members,
                               cats, prev, out, scratch.synth);
        break;
      case AssignmentDelta::CentreMoved:
        reassign_moved_general(x_row, n, p, centres, nC, d, scratch.dim0.data(), touched, metric,
                               members, cats, prev, out, scratch.synth);
        break;
      case AssignmentDelta::CentreRemoved:
        reassign_removed_general(x_row, n, p, centres, nC, d, scratch.dim0.data(), touched, metric,
                                 members, cats, prev, out, scratch.synth);
        break;
      case AssignmentDelta::FullRecompute:
        assign_full_general(x_row, n, p, centres, nC, d, scratch.dim0.data(), metric, members, cats,
                            out, scratch.synth);
        break;
    }
  }
}

static void aggregate_residuals_both(const std::vector<double>& R_j,
                                     const std::vector<int>& idx_old,
                                     int nC_old,
                                     const std::vector<int>& idx_new,
                                     int nC_new,
                                     std::vector<double>& R_old,
                                     std::vector<int>& n_old,
                                     std::vector<double>& R_new,
                                     std::vector<int>& n_new) {
  R_old.assign(nC_old, 0.0);
  n_old.assign(nC_old, 0);
  R_new.assign(nC_new, 0.0);
  n_new.assign(nC_new, 0);
  const int n = static_cast<int>(R_j.size());
  for (int obs = 0; obs < n; ++obs) {
    R_old[idx_old[obs]] += R_j[obs];
    n_old[idx_old[obs]]++;
    R_new[idx_new[obs]] += R_j[obs];
    n_new[idx_new[obs]]++;
  }
}

// ---------------------------------------------------------------------------
// Progress helpers
// ---------------------------------------------------------------------------

void render_progress_bar(const std::string& label, int current, int total, int width = 30) {
  if (total <= 0) {
    return;
  }
  if (current < 0) {
    current = 0;
  }
  if (current > total) {
    current = total;
  }

  const double fraction = static_cast<double>(current) / static_cast<double>(total);
  int filled = static_cast<int>(std::round(fraction * width));
  if (filled > width) {
    filled = width;
  }
  const int percent = static_cast<int>(std::round(fraction * 100.0));

  std::ostringstream out;
  out << '\r' << label << " [";
  for (int idx = 0; idx < width; ++idx) {
    out << (idx < filled ? '#' : '-');
  }
  out << "] " << std::setw(3) << percent << "% (" << current << "/" << total << ")";
  if (current >= total) {
    out << '\n';
  }

  std::cerr << out.str() << std::flush;
}

void maybe_progress(const std::string& label,
                    int current,
                    int total,
                    int width,
                    int& last_filled,
                    bool show_progress) {
  if (!show_progress) {
    return;
  }
  int filled = 0;
  if (total > 0) {
    filled = static_cast<int>(static_cast<double>(current) / static_cast<double>(total) * width + 1e-12);
    if (filled > width) {
      filled = width;
    }
  }
  if (current == 1 || current == total || filled != last_filled) {
    render_progress_bar(label, current, total, width);
    last_filled = filled;
  }
}

// ---------------------------------------------------------------------------
// MH / mu helpers (Python selection_prob formulation)
// ---------------------------------------------------------------------------

double tessellation_log_likelihood_component(const std::vector<double>& r_cell,
                                             const std::vector<int>& n_cell,
                                             double sigma_squared,
                                             double sigma_squared_mu) {
  double sum_log = 0.0;
  double sum_r = 0.0;
  for (int idx = 0; idx < static_cast<int>(n_cell.size()); ++idx) {
    const double den = n_cell[idx] * sigma_squared_mu + sigma_squared;
    sum_log += std::log(den);
    sum_r += (r_cell[idx] * r_cell[idx]) / den;
  }
  return -0.5 * sum_log + (sigma_squared_mu / (2.0 * sigma_squared)) * sum_r;
}

double selection_prob(const std::string& modification, int b, int d, int p) {
  if (modification == "AD") {
    if (d == p) {
      return 0.0;
    }
    return (d == 1) ? 0.4 : 0.2;
  }
  if (modification == "RD") {
    if (d <= 1) {
      return 0.0;
    }
    return (d == p) ? 0.4 : 0.2;
  }
  if (modification == "AC") {
    return (b == 1) ? 0.4 : 0.2;
  }
  if (modification == "RC") {
    return (b > 1) ? 0.2 : 0.0;
  }
  if (modification == "Change") {
    return (d == p) ? 0.2 : 0.1;
  }
  if (modification == "Swap") {
    return (d < p) ? 0.1 : 0.0;
  }
  return 0.0;
}

std::string reverse_modification(const std::string& modification) {
  if (modification == "AD") {
    return "RD";
  }
  if (modification == "RD") {
    return "AD";
  }
  if (modification == "AC") {
    return "RC";
  }
  if (modification == "RC") {
    return "AC";
  }
  return modification;
}

double log_structure_and_selection(int d_new,
                                   int n_centres_new,
                                   double sigma_squared,
                                   double omega,
                                   double lambda_rate,
                                   int p,
                                   const std::string& modification) {
  double acc = 0.0;
  int b_old = n_centres_new;
  int d_old = d_new;

  if (modification == "AD") {
    d_old = d_new - 1;
    acc += std::log(static_cast<double>(p - d_old)) - std::log(static_cast<double>(d_old)) +
           std::log(omega) - std::log(static_cast<double>(p) - omega);
  } else if (modification == "RD") {
    d_old = d_new + 1;
    acc += std::log(static_cast<double>(d_old - 1)) - std::log(static_cast<double>(p - d_old + 1)) +
           std::log(static_cast<double>(p) - omega) - std::log(omega);
  } else if (modification == "AC") {
    b_old = n_centres_new - 1;
    acc += std::log(lambda_rate) - std::log(static_cast<double>(b_old)) + 0.5 * std::log(sigma_squared);
  } else if (modification == "RC") {
    b_old = n_centres_new + 1;
    acc += std::log(static_cast<double>(b_old - 1)) - std::log(lambda_rate) - 0.5 * std::log(sigma_squared);
  } else {
    return acc;
  }

  const std::string reverse = reverse_modification(modification);
  const double q_forward = selection_prob(modification, b_old, d_old, p);
  const double q_reverse = selection_prob(reverse, n_centres_new, d_new, p);
  acc += std::log(q_reverse) - std::log(q_forward);
  return acc;
}

double log_acceptance_components(const std::vector<double>& r_old,
                                 const std::vector<int>& n_old,
                                 const std::vector<double>& r_new,
                                 const std::vector<int>& n_new,
                                 int d_new,
                                 int n_centres_new,
                                 double sigma_squared,
                                 double sigma_squared_mu,
                                 double omega,
                                 double lambda_rate,
                                 int p,
                                 const std::string& modification) {
  const double old_log_lik =
      tessellation_log_likelihood_component(r_old, n_old, sigma_squared, sigma_squared_mu);
  const double new_log_lik =
      tessellation_log_likelihood_component(r_new, n_new, sigma_squared, sigma_squared_mu);
  return new_log_lik - old_log_lik +
         log_structure_and_selection(
             d_new, n_centres_new, sigma_squared, omega, lambda_rate, p, modification);
}

static void sample_mu_into(const std::vector<double>& r_cell,
                           const std::vector<int>& n_cell,
                           double sigma_squared_mu,
                           double sigma_squared,
                           std::vector<double>& result,
                           std::mt19937_64& rng) {
  const int n_cells = static_cast<int>(r_cell.size());
  result.resize(n_cells);
  std::normal_distribution<double> normal(0.0, 1.0);
  for (int idx = 0; idx < n_cells; ++idx) {
    const double den = sigma_squared_mu * n_cell[idx] + sigma_squared;
    const double mean = (sigma_squared_mu * r_cell[idx]) / den;
    const double sd = std::sqrt((sigma_squared * sigma_squared_mu) / den);
    result[idx] = mean + normal(rng) * sd;
  }
}

// ---------------------------------------------------------------------------
// Tessellation proposals (with AssignmentDelta)
// ---------------------------------------------------------------------------

struct ProposalResult {
  std::vector<double> tess;  // row-major nC x d
  int n_centres = 0;
  std::vector<int> dim;  // 0-based
  std::string modification = "Change";
  AssignmentDelta delta = AssignmentDelta::CentreMoved;
  int touched = 0;  // moved/removed centre index (0-based)
};

static ProposalResult propose_internal(const std::vector<double>& tess,
                                       int n_centres,
                                       int d,
                                       const std::vector<int>& dim,
                                       int p,
                                       const std::vector<double>& proposal_sd,
                                       const std::vector<double>& proposal_mu,
                                       const std::vector<int>& metric,
                                       const std::vector<int>& members,
                                       const std::vector<int>& ncats,
                                       const std::vector<int>& cat_index_of_col,
                                       std::mt19937_64& rng) {
  ProposalResult result;
  result.tess = tess;
  result.n_centres = n_centres;
  result.dim = dim;
  result.modification = "Change";
  result.delta = AssignmentDelta::CentreMoved;
  result.touched = 0;

  const double choice = uniform01(rng);
  std::normal_distribution<double> normal(0.0, 1.0);

  auto sample_global_coordinate = [&](int global_dim) {
    if (metric[global_dim] == 2) {
      const int cat_idx = cat_index_of_col[global_dim];
      if (cat_idx < 0 || cat_idx >= static_cast<int>(ncats.size()) || ncats[cat_idx] <= 0) {
        throw std::invalid_argument("Invalid categorical level count for proposal.");
      }
      std::uniform_int_distribution<int> cat_dist(1, ncats[cat_idx]);
      return static_cast<double>(cat_dist(rng));
    }
    double value = proposal_mu[global_dim] + normal(rng) * proposal_sd[global_dim];
    if (metric[global_dim] == 1 && is_last_member_column(global_dim, members)) {
      value = period_shift(value, kPi);
    }
    return value;
  };

  if ((choice < 0.2 && d != p) || (d == 1 && d != p && choice < 0.4)) {
    result.modification = "AD";
    result.delta = AssignmentDelta::FullRecompute;
    int new_dim = 0;
    std::uniform_int_distribution<int> dim_dist(0, p - 1);
    do {
      new_dim = dim_dist(rng);
    } while (in_vector(new_dim, result.dim));

    result.dim.push_back(new_dim);
    result.tess.assign(static_cast<size_t>(n_centres) * (d + 1), 0.0);
    for (int row = 0; row < n_centres; ++row) {
      for (int col = 0; col < d; ++col) {
        result.tess[static_cast<size_t>(row) * (d + 1) + col] = tess[static_cast<size_t>(row) * d + col];
      }
      result.tess[static_cast<size_t>(row) * (d + 1) + d] = sample_global_coordinate(new_dim);
    }
  } else if (choice < 0.4 && d > 1) {
    result.modification = "RD";
    result.delta = AssignmentDelta::FullRecompute;
    std::uniform_int_distribution<int> remove_dist(0, d - 1);
    const int remove_idx = remove_dist(rng);
    result.dim.erase(result.dim.begin() + remove_idx);
    result.tess.assign(static_cast<size_t>(n_centres) * (d - 1), 0.0);
    for (int row = 0; row < n_centres; ++row) {
      int out_col = 0;
      for (int col = 0; col < d; ++col) {
        if (col == remove_idx) {
          continue;
        }
        result.tess[static_cast<size_t>(row) * (d - 1) + out_col] = tess[static_cast<size_t>(row) * d + col];
        ++out_col;
      }
    }
  } else if (choice < 0.6 || (choice < 0.8 && n_centres == 1)) {
    result.modification = "AC";
    result.delta = AssignmentDelta::CentreAdded;
    result.n_centres = n_centres + 1;
    result.tess.assign(static_cast<size_t>(result.n_centres) * d, 0.0);
    for (int row = 0; row < n_centres; ++row) {
      for (int col = 0; col < d; ++col) {
        result.tess[static_cast<size_t>(row) * d + col] = tess[static_cast<size_t>(row) * d + col];
      }
    }
    for (int col = 0; col < d; ++col) {
      result.tess[static_cast<size_t>(n_centres) * d + col] = sample_global_coordinate(result.dim[col]);
    }
  } else if (choice < 0.8 && n_centres > 1) {
    result.modification = "RC";
    result.delta = AssignmentDelta::CentreRemoved;
    std::uniform_int_distribution<int> remove_dist(0, n_centres - 1);
    const int remove_row = remove_dist(rng);
    result.touched = remove_row;
    result.n_centres = n_centres - 1;
    // MUST clear before push_back — previously tess was copied from old then
    // append compacted centres, so first nC*d elements were wrong.
    result.tess.clear();
    result.tess.reserve(static_cast<size_t>(result.n_centres) * d);
    for (int row = 0; row < n_centres; ++row) {
      if (row == remove_row) {
        continue;
      }
      for (int col = 0; col < d; ++col) {
        result.tess.push_back(tess[static_cast<size_t>(row) * d + col]);
      }
    }
  } else if (choice < 0.9 || d == p) {
    result.modification = "Change";
    result.delta = AssignmentDelta::CentreMoved;
    std::uniform_int_distribution<int> centre_dist(0, n_centres - 1);
    const int centre = centre_dist(rng);
    result.touched = centre;
    for (int col = 0; col < d; ++col) {
      result.tess[static_cast<size_t>(centre) * d + col] = sample_global_coordinate(result.dim[col]);
    }
  } else {
    result.modification = "Swap";
    result.delta = AssignmentDelta::FullRecompute;
    std::uniform_int_distribution<int> local_dim_dist(0, d - 1);
    std::uniform_int_distribution<int> global_dim_dist(0, p - 1);
    const int local_dim = local_dim_dist(rng);
    int new_dim = 0;
    do {
      new_dim = global_dim_dist(rng);
    } while (in_vector(new_dim, result.dim));
    result.dim[local_dim] = new_dim;
    for (int row = 0; row < n_centres; ++row) {
      result.tess[static_cast<size_t>(row) * d + local_dim] = sample_global_coordinate(new_dim);
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Compact posterior store (deferred Python packaging)
// ---------------------------------------------------------------------------

struct StoredTess {
  std::vector<double> centres;  // row-major nC x d
  int n_centres = 0;
  int d = 0;
  std::vector<int> dim;  // 0-based
  std::vector<double> mu;
};

struct StoredDraw {
  std::vector<StoredTess> tessellations;
  double sigma = 0.0;
};

struct FlatPosterior {
  int num_samples = 0;
  int m = 0;
  std::vector<double> centres;  // concatenated row-major blocks
  std::vector<double> mus;
  std::vector<int> dims;  // 0-based, concatenated
  std::vector<int> centre_off;
  std::vector<int> mu_off;
  std::vector<int> dim_off;
  std::vector<int> n_centres;
  std::vector<int> d;
};

// ---------------------------------------------------------------------------
// Array helpers
// ---------------------------------------------------------------------------

std::vector<double> copy_double_array(py::handle handle,
                                      int expected_ndim,
                                      std::vector<ssize_t>* shape = nullptr) {
  py::array_t<double, py::array::c_style | py::array::forcecast> arr =
      py::cast<py::array_t<double>>(handle);
  py::buffer_info info = arr.request();
  if (info.ndim != expected_ndim) {
    throw std::invalid_argument("Unexpected array dimensionality.");
  }
  if (shape != nullptr) {
    shape->assign(info.shape.begin(), info.shape.end());
  }
  const auto* ptr = static_cast<const double*>(info.ptr);
  return std::vector<double>(ptr, ptr + info.size);
}

std::vector<int> copy_int_array(py::handle handle) {
  py::array_t<int, py::array::c_style | py::array::forcecast> arr = py::cast<py::array_t<int>>(handle);
  py::buffer_info info = arr.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("Expected a one-dimensional integer array.");
  }
  const auto* ptr = static_cast<const int*>(info.ptr);
  return std::vector<int>(ptr, ptr + info.size);
}

py::array_t<double> make_matrix(const std::vector<double>& values, int rows, int cols) {
  py::array_t<double> arr({rows, cols});
  auto out = arr.mutable_unchecked<2>();
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      out(row, col) = values[static_cast<size_t>(row) * cols + col];
    }
  }
  return arr;
}

py::array_t<double> make_vector(const std::vector<double>& values) {
  py::array_t<double> arr(static_cast<py::ssize_t>(values.size()));
  auto out = arr.mutable_unchecked<1>();
  for (int idx = 0; idx < static_cast<int>(values.size()); ++idx) {
    out(idx) = values[idx];
  }
  return arr;
}

py::array_t<int> make_int_vector(const std::vector<int>& values) {
  py::array_t<int> arr(static_cast<py::ssize_t>(values.size()));
  auto out = arr.mutable_unchecked<1>();
  for (int idx = 0; idx < static_cast<int>(values.size()); ++idx) {
    out(idx) = values[idx];
  }
  return arr;
}

FlatPosterior flatten_posterior(const py::list& posterior_tess,
                                const py::list& posterior_dim,
                                const py::list& posterior_pred) {
  FlatPosterior flat;
  flat.num_samples = static_cast<int>(posterior_tess.size());
  if (flat.num_samples == 0) {
    return flat;
  }
  if (static_cast<int>(posterior_dim.size()) != flat.num_samples ||
      static_cast<int>(posterior_pred.size()) != flat.num_samples) {
    throw std::invalid_argument("Posterior tess/dim/pred lists must have equal length.");
  }

  py::list first_sample = py::cast<py::list>(posterior_tess[0]);
  flat.m = static_cast<int>(first_sample.size());
  const int total = flat.num_samples * flat.m;
  flat.centre_off.resize(total);
  flat.mu_off.resize(total);
  flat.dim_off.resize(total);
  flat.n_centres.resize(total);
  flat.d.resize(total);

  int c_off = 0;
  int m_off = 0;
  int d_off = 0;
  for (int s = 0; s < flat.num_samples; ++s) {
    py::list sample_tess = py::cast<py::list>(posterior_tess[s]);
    py::list sample_dim = py::cast<py::list>(posterior_dim[s]);
    py::list sample_pred = py::cast<py::list>(posterior_pred[s]);
    if (static_cast<int>(sample_tess.size()) != flat.m ||
        static_cast<int>(sample_dim.size()) != flat.m ||
        static_cast<int>(sample_pred.size()) != flat.m) {
      throw std::invalid_argument("Posterior sample has inconsistent tessellation counts.");
    }
    for (int j = 0; j < flat.m; ++j) {
      const int idx = s * flat.m + j;
      std::vector<ssize_t> tess_shape;
      std::vector<double> centres = copy_double_array(sample_tess[j], 2, &tess_shape);
      const int nC = static_cast<int>(tess_shape[0]);
      const int d = static_cast<int>(tess_shape[1]);
      std::vector<int> dims = copy_int_array(sample_dim[j]);
      std::vector<double> mus = copy_double_array(sample_pred[j], 1);
      if (static_cast<int>(dims.size()) != d) {
        throw std::invalid_argument("Tessellation has mismatched dim length.");
      }
      if (static_cast<int>(mus.size()) != nC) {
        throw std::invalid_argument("Tessellation has mismatched pred length.");
      }

      flat.n_centres[idx] = nC;
      flat.d[idx] = d;
      flat.centre_off[idx] = c_off;
      flat.mu_off[idx] = m_off;
      flat.dim_off[idx] = d_off;

      flat.centres.resize(c_off + nC * d);
      if (nC * d > 0) {
        std::memcpy(flat.centres.data() + c_off, centres.data(),
                    static_cast<size_t>(nC) * d * sizeof(double));
      }
      flat.mus.resize(m_off + nC);
      if (nC > 0) {
        std::memcpy(flat.mus.data() + m_off, mus.data(), static_cast<size_t>(nC) * sizeof(double));
      }
      flat.dims.resize(d_off + d);
      if (d > 0) {
        std::memcpy(flat.dims.data() + d_off, dims.data(), static_cast<size_t>(d) * sizeof(int));
      }
      c_off += nC * d;
      m_off += nC;
      d_off += d;
    }
  }
  return flat;
}

}  // namespace

py::dict run_mcmc(py::array_t<double, py::array::c_style | py::array::forcecast> x_scaled_arr,
                  py::array_t<double, py::array::c_style | py::array::forcecast> y_scaled_arr,
                  py::array_t<int, py::array::c_style | py::array::forcecast> metric_arr,
                  py::array_t<int, py::array::c_style | py::array::forcecast> member_arr,
                  int m,
                  int total_iter,
                  int burn_in,
                  int thinning,
                  double nu,
                  double lambda,
                  double sigma_squared_mu,
                  double omega,
                  double lambda_rate,
                  py::array_t<double, py::array::c_style | py::array::forcecast> proposal_sd_arr,
                  py::array_t<double, py::array::c_style | py::array::forcecast> proposal_mu_arr,
                  py::list init_tess,
                  py::list init_dim,
                  py::list init_pred,
                  py::array_t<int, py::array::c_style | py::array::forcecast> binary_cols_arr,
                  double cat_scaling,
                  std::uint64_t seed,
                  bool verbose) {
  py::buffer_info x_info = x_scaled_arr.request();
  py::buffer_info y_info = y_scaled_arr.request();
  if (x_info.ndim != 2 || y_info.ndim != 1) {
    throw std::invalid_argument("x_scaled must be 2D and y_scaled must be 1D.");
  }
  const int n = static_cast<int>(x_info.shape[0]);
  const int p = static_cast<int>(x_info.shape[1]);
  if (static_cast<int>(y_info.shape[0]) != n) {
    throw std::invalid_argument("x_scaled and y_scaled have incompatible row counts.");
  }

  const auto* x_scaled = static_cast<const double*>(x_info.ptr);
  const auto* y_scaled = static_cast<const double*>(y_info.ptr);
  std::vector<int> metric = copy_int_array(metric_arr);
  std::vector<int> members = copy_int_array(member_arr);
  std::vector<double> proposal_sd = copy_double_array(proposal_sd_arr, 1);
  std::vector<double> proposal_mu = copy_double_array(proposal_mu_arr, 1);
  std::vector<int> binary_cols = copy_int_array(binary_cols_arr);
  const ReducedMetric reduced = make_reduced_metric(metric, members);
  const std::vector<int> ncats = compute_ncats_from_data(x_scaled, n, p, metric);

  if (static_cast<int>(metric.size()) != p || static_cast<int>(members.size()) != p ||
      static_cast<int>(proposal_sd.size()) != p || static_cast<int>(proposal_mu.size()) != p) {
    throw std::invalid_argument("Per-feature arrays must match x_scaled column count.");
  }
  if (static_cast<int>(init_tess.size()) != m || static_cast<int>(init_dim.size()) != m ||
      static_cast<int>(init_pred.size()) != m) {
    throw std::invalid_argument("Initial state lists must have length n_tessellations.");
  }

  std::vector<char> is_binary(p, 0);
  for (int col : binary_cols) {
    if (col >= 0 && col < p) {
      is_binary[col] = 1;
    }
  }

  std::vector<int> cat_index_of_col(p, -1);
  {
    int cat_i = 0;
    for (int col = 0; col < p; ++col) {
      if (metric[col] == 2) {
        cat_index_of_col[col] = cat_i++;
      }
    }
  }

  const bool euclidean = all_euclidean_metric(reduced.metric);

  std::vector<std::vector<double>> tess(m);
  std::vector<int> tess_n_centres(m);
  std::vector<int> tess_dim_count(m);
  std::vector<std::vector<int>> dim(m);
  std::vector<std::vector<double>> pred(m);

  for (int idx = 0; idx < m; ++idx) {
    std::vector<ssize_t> tess_shape;
    tess[idx] = copy_double_array(init_tess[idx], 2, &tess_shape);
    tess_n_centres[idx] = static_cast<int>(tess_shape[0]);
    tess_dim_count[idx] = static_cast<int>(tess_shape[1]);
    dim[idx] = copy_int_array(init_dim[idx]);
    pred[idx] = copy_double_array(init_pred[idx], 1);
    if (static_cast<int>(dim[idx].size()) != tess_dim_count[idx] ||
        static_cast<int>(pred[idx].size()) != tess_n_centres[idx]) {
      throw std::invalid_argument("Initial tessellation, dimension, and prediction shapes are incompatible.");
    }
  }

  std::mt19937_64 rng(seed);

  std::vector<AssignmentCache> caches(m);
  AssignScratch assign_scratch;
  AssignmentCache prop_cache;
  for (int j = 0; j < m; ++j) {
    reassign(x_scaled, n, p, tess[j].data(), tess_n_centres[j], tess_dim_count[j], dim[j],
             AssignmentDelta::FullRecompute, 0, euclidean, reduced.metric, reduced.member_counts,
             ncats, AssignmentCache{}, caches[j], assign_scratch);
  }

  std::vector<double> sum_all_tess(n, 0.0);
  for (int j = 0; j < m; ++j) {
    for (int obs = 0; obs < n; ++obs) {
      sum_all_tess[obs] += pred[j][caches[j].assignment[obs]];
    }
  }

  const int num_samples = total_iter > burn_in ? (total_iter - burn_in) / thinning : 0;
  std::vector<StoredDraw> stored;
  stored.reserve(num_samples);
  std::vector<double> prediction_matrix(static_cast<size_t>(n) * num_samples, 0.0);
  std::vector<int> trace_iteration(total_iter);
  std::vector<uint8_t> trace_is_burn_in(total_iter);
  std::vector<double> trace_avg_centres(total_iter);
  std::vector<double> trace_sd_centres(total_iter);
  std::vector<double> trace_avg_dims(total_iter);
  std::vector<double> trace_log_likelihood(total_iter);

  std::vector<double> residuals(n);
  std::vector<double> last_tess_pred(n, 0.0);
  std::vector<double> r_old;
  std::vector<double> r_new;
  std::vector<int> n_old;
  std::vector<int> n_new;

  double sigma_squared = 1.0;
  int storage_idx = 0;
  constexpr int progress_width = 30;
  int last_filled = -1;

  for (int iter = 1; iter <= total_iter; ++iter) {
    maybe_progress("MCMC fit", iter, total_iter, progress_width, last_filled, verbose);

    double sum_sq = 0.0;
    for (int obs = 0; obs < n; ++obs) {
      const double residual = y_scaled[obs] - sum_all_tess[obs];
      sum_sq += residual * residual;
    }
    const double shape = (nu + n) / 2.0;
    const double rate = (nu * lambda + sum_sq) / 2.0;
    std::gamma_distribution<double> gamma(shape, 1.0 / rate);
    sigma_squared = 1.0 / gamma(rng);

    for (int j = 0; j < m; ++j) {
      if (j == 0) {
        for (int obs = 0; obs < n; ++obs) {
          sum_all_tess[obs] -= pred[j][caches[j].assignment[obs]];
        }
      } else {
        for (int obs = 0; obs < n; ++obs) {
          sum_all_tess[obs] += last_tess_pred[obs] - pred[j][caches[j].assignment[obs]];
        }
      }

      for (int obs = 0; obs < n; ++obs) {
        residuals[obs] = y_scaled[obs] - sum_all_tess[obs];
      }

      ProposalResult proposal = propose_internal(tess[j],
                                                 tess_n_centres[j],
                                                 tess_dim_count[j],
                                                 dim[j],
                                                 p,
                                                 proposal_sd,
                                                 proposal_mu,
                                                 metric,
                                                 members,
                                                 ncats,
                                                 cat_index_of_col,
                                                 rng);

      const int d_star = static_cast<int>(proposal.dim.size());
      for (int local_dim = 0; local_dim < d_star; ++local_dim) {
        const int g0 = proposal.dim[local_dim];
        if (g0 >= 0 && g0 < p && is_binary[g0]) {
          for (int row = 0; row < proposal.n_centres; ++row) {
            double& value = proposal.tess[static_cast<size_t>(row) * d_star + local_dim];
            value = std::clamp(value, 0.0, cat_scaling);
          }
        }
      }

      reassign(x_scaled, n, p, proposal.tess.data(), proposal.n_centres, d_star, proposal.dim,
               proposal.delta, proposal.touched, euclidean, reduced.metric, reduced.member_counts,
               ncats, caches[j], prop_cache, assign_scratch);

      aggregate_residuals_both(residuals, caches[j].assignment, tess_n_centres[j],
                               prop_cache.assignment, proposal.n_centres, r_old, n_old, r_new, n_new);

      const bool has_empty =
          std::any_of(n_new.begin(), n_new.end(), [](int value) { return value == 0; });
      bool accepted = false;
      if (!has_empty) {
        const double log_alpha = log_acceptance_components(r_old,
                                                           n_old,
                                                           r_new,
                                                           n_new,
                                                           d_star,
                                                           proposal.n_centres,
                                                           sigma_squared,
                                                           sigma_squared_mu,
                                                           omega,
                                                           lambda_rate,
                                                           p,
                                                           proposal.modification);
        accepted = std::log(uniform01(rng)) < log_alpha;
      }

      if (accepted) {
        tess[j] = std::move(proposal.tess);
        tess_n_centres[j] = proposal.n_centres;
        tess_dim_count[j] = d_star;
        dim[j] = std::move(proposal.dim);
        caches[j] = std::move(prop_cache);
        sample_mu_into(r_new, n_new, sigma_squared_mu, sigma_squared, pred[j], rng);
        for (int obs = 0; obs < n; ++obs) {
          last_tess_pred[obs] = pred[j][caches[j].assignment[obs]];
        }
      } else {
        sample_mu_into(r_old, n_old, sigma_squared_mu, sigma_squared, pred[j], rng);
        for (int obs = 0; obs < n; ++obs) {
          last_tess_pred[obs] = pred[j][caches[j].assignment[obs]];
        }
      }

      if (j == m - 1) {
        for (int obs = 0; obs < n; ++obs) {
          sum_all_tess[obs] += last_tess_pred[obs];
        }
      }
    }

    double mean_centres = 0.0;
    double mean_dims = 0.0;
    double retained_log_lik_sum = 0.0;
    for (int j = 0; j < m; ++j) {
      mean_centres += tess_n_centres[j];
      mean_dims += tess_dim_count[j];

      r_old.assign(tess_n_centres[j], 0.0);
      n_old.assign(tess_n_centres[j], 0);
      for (int obs = 0; obs < n; ++obs) {
        const int cell = caches[j].assignment[obs];
        const double tess_contribution = pred[j][cell];
        const double partial_residual = y_scaled[obs] - (sum_all_tess[obs] - tess_contribution);
        r_old[cell] += partial_residual;
        n_old[cell] += 1;
      }
      retained_log_lik_sum +=
          tessellation_log_likelihood_component(r_old, n_old, sigma_squared, sigma_squared_mu);
    }
    mean_centres /= static_cast<double>(m);
    mean_dims /= static_cast<double>(m);

    double sd_centres = 0.0;
    if (m > 1) {
      for (int j = 0; j < m; ++j) {
        const double diff = tess_n_centres[j] - mean_centres;
        sd_centres += diff * diff;
      }
      sd_centres = std::sqrt(sd_centres / static_cast<double>(m - 1));
    }

    const int trace_idx = iter - 1;
    trace_iteration[trace_idx] = iter;
    trace_is_burn_in[trace_idx] = iter <= burn_in ? 1 : 0;
    trace_avg_centres[trace_idx] = mean_centres;
    trace_sd_centres[trace_idx] = sd_centres;
    trace_avg_dims[trace_idx] = mean_dims;
    trace_log_likelihood[trace_idx] = retained_log_lik_sum / static_cast<double>(m);

    if (iter > burn_in && (iter - burn_in) % thinning == 0) {
      for (int obs = 0; obs < n; ++obs) {
        prediction_matrix[static_cast<size_t>(obs) * num_samples + storage_idx] = sum_all_tess[obs];
      }

      StoredDraw draw;
      draw.sigma = sigma_squared;
      draw.tessellations.resize(m);
      for (int j = 0; j < m; ++j) {
        StoredTess& st = draw.tessellations[j];
        st.n_centres = tess_n_centres[j];
        st.d = tess_dim_count[j];
        st.centres = tess[j];
        st.dim = dim[j];
        st.mu = pred[j];
      }
      stored.push_back(std::move(draw));
      ++storage_idx;
    }
  }

  py::list posterior_tess;
  py::list posterior_dim;
  py::list posterior_pred;
  std::vector<double> posterior_sigma(num_samples);
  for (int s = 0; s < num_samples; ++s) {
    py::list sample_tess;
    py::list sample_dim;
    py::list sample_pred;
    for (int j = 0; j < m; ++j) {
      const StoredTess& st = stored[s].tessellations[j];
      sample_tess.append(make_matrix(st.centres, st.n_centres, st.d));
      sample_dim.append(make_int_vector(st.dim));
      sample_pred.append(make_vector(st.mu));
    }
    posterior_tess.append(sample_tess);
    posterior_dim.append(sample_dim);
    posterior_pred.append(sample_pred);
    posterior_sigma[s] = stored[s].sigma;
  }

  py::array_t<double> pred_matrix_arr({n, num_samples});
  auto pred_matrix = pred_matrix_arr.mutable_unchecked<2>();
  for (int obs = 0; obs < n; ++obs) {
    for (int sample = 0; sample < num_samples; ++sample) {
      pred_matrix(obs, sample) = prediction_matrix[static_cast<size_t>(obs) * num_samples + sample];
    }
  }

  py::dict trace_stats;
  trace_stats["iteration"] = make_int_vector(trace_iteration);
  py::array_t<uint8_t> burn_in_arr(static_cast<py::ssize_t>(trace_is_burn_in.size()));
  auto burn_in_out = burn_in_arr.mutable_unchecked<1>();
  for (int idx = 0; idx < total_iter; ++idx) {
    burn_in_out(idx) = trace_is_burn_in[idx];
  }
  trace_stats["is_burn_in"] = burn_in_arr;
  trace_stats["average_centres_per_tessellation"] = make_vector(trace_avg_centres);
  trace_stats["sd_centres_per_tessellation"] = make_vector(trace_sd_centres);
  trace_stats["average_dimensions_per_tessellation"] = make_vector(trace_avg_dims);
  trace_stats["log_likelihood"] = make_vector(trace_log_likelihood);

  py::dict result;
  result["posterior_tess"] = posterior_tess;
  result["posterior_dim"] = posterior_dim;
  result["posterior_pred"] = posterior_pred;
  result["posterior_sigma"] = make_vector(posterior_sigma);
  result["prediction_matrix"] = pred_matrix_arr;
  result["trace_stats"] = trace_stats;
  return result;
}

py::array_t<int> cell_indices(py::array_t<double, py::array::c_style | py::array::forcecast> query_arr,
                              py::array_t<double, py::array::c_style | py::array::forcecast> centres_arr,
                              py::array_t<int, py::array::c_style | py::array::forcecast> dim_arr,
                              py::array_t<int, py::array::c_style | py::array::forcecast> metric_red_arr,
                              py::array_t<int, py::array::c_style | py::array::forcecast> member_red_arr,
                              py::object ncats_obj = py::none()) {
  py::buffer_info query_info = query_arr.request();
  py::buffer_info centres_info = centres_arr.request();
  if (query_info.ndim != 2 || centres_info.ndim != 2) {
    throw std::invalid_argument("query and centres must be two-dimensional arrays.");
  }

  const int n = static_cast<int>(query_info.shape[0]);
  const int p = static_cast<int>(query_info.shape[1]);
  const int n_centres = static_cast<int>(centres_info.shape[0]);
  const int d = static_cast<int>(centres_info.shape[1]);
  const auto* query = static_cast<const double*>(query_info.ptr);
  const auto* centres_ptr = static_cast<const double*>(centres_info.ptr);
  std::vector<double> centres(centres_ptr, centres_ptr + centres_info.size);
  std::vector<int> dim = copy_int_array(dim_arr);
  std::vector<int> metric_red = copy_int_array(metric_red_arr);
  std::vector<int> member_red = copy_int_array(member_red_arr);

  if (static_cast<int>(dim.size()) != d) {
    throw std::invalid_argument("dim length must match centres column count.");
  }
  const int member_total = std::accumulate(member_red.begin(), member_red.end(), 0);
  if (member_total != p) {
    throw std::invalid_argument("member_red must sum to query column count.");
  }

  std::vector<int> ncats;
  if (ncats_obj.is_none()) {
    std::vector<int> metric_full;
    metric_full.reserve(p);
    for (int group = 0; group < static_cast<int>(metric_red.size()); ++group) {
      metric_full.insert(metric_full.end(), member_red[group], metric_red[group]);
    }
    ncats = compute_ncats_from_data(query, n, p, metric_full);
  } else {
    ncats = copy_int_array(ncats_obj);
  }

  int expected_cats = 0;
  for (int group = 0; group < static_cast<int>(metric_red.size()); ++group) {
    if (metric_red[group] == 2) {
      expected_cats += member_red[group];
    }
  }
  if (static_cast<int>(ncats.size()) != expected_cats) {
    throw std::invalid_argument("ncats length must match the number of categorical columns.");
  }

  const bool euclidean = all_euclidean_metric(metric_red);
  AssignmentCache cache;
  AssignScratch scratch;
  reassign(query, n, p, centres.data(), n_centres, d, dim, AssignmentDelta::FullRecompute, 0,
           euclidean, metric_red, member_red, ncats, AssignmentCache{}, cache, scratch);
  return make_int_vector(cache.assignment);
}

py::array_t<double> predict_ensemble(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_arr,
    py::list posterior_tess,
    py::list posterior_dim,
    py::list posterior_pred,
    py::array_t<int, py::array::c_style | py::array::forcecast> metric_red_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> member_red_arr,
    py::object ncats_obj,
    bool verbose) {
  py::buffer_info x_info = x_arr.request();
  if (x_info.ndim != 2) {
    throw std::invalid_argument("x must be a two-dimensional array.");
  }
  const int n = static_cast<int>(x_info.shape[0]);
  const int p = static_cast<int>(x_info.shape[1]);
  const auto* x_row = static_cast<const double*>(x_info.ptr);

  FlatPosterior flat = flatten_posterior(posterior_tess, posterior_dim, posterior_pred);
  const int num_samples = flat.num_samples;
  if (num_samples == 0) {
    return py::array_t<double>({n, 0});
  }
  const int m = flat.m;

  std::vector<int> metric_red = copy_int_array(metric_red_arr);
  std::vector<int> member_red = copy_int_array(member_red_arr);
  const int member_total = std::accumulate(member_red.begin(), member_red.end(), 0);
  if (member_total != p) {
    throw std::invalid_argument("member_red must sum to x column count.");
  }

  std::vector<int> ncats;
  if (ncats_obj.is_none()) {
    std::vector<int> metric_full;
    metric_full.reserve(p);
    for (int group = 0; group < static_cast<int>(metric_red.size()); ++group) {
      metric_full.insert(metric_full.end(), member_red[group], metric_red[group]);
    }
    ncats = compute_ncats_from_data(x_row, n, p, metric_full);
  } else {
    ncats = copy_int_array(ncats_obj);
  }

  int expected_cats = 0;
  for (int group = 0; group < static_cast<int>(metric_red.size()); ++group) {
    if (metric_red[group] == 2) {
      expected_cats += member_red[group];
    }
  }
  if (static_cast<int>(ncats.size()) != expected_cats) {
    throw std::invalid_argument("ncats length must match the number of categorical columns.");
  }

  const bool euclidean = all_euclidean_metric(metric_red);
  py::array_t<double> out({n, num_samples});
  auto out_view = out.mutable_unchecked<2>();

  AssignScratch assign_scratch;
  AssignmentCache cache;
  std::vector<double> draw_pred(n, 0.0);
  std::vector<int> dim0;

  constexpr int progress_width = 30;
  int last_filled = -1;

  for (int s = 0; s < num_samples; ++s) {
    maybe_progress("Predict", s + 1, num_samples, progress_width, last_filled, verbose);
    std::fill(draw_pred.begin(), draw_pred.end(), 0.0);

    for (int j = 0; j < m; ++j) {
      const int idx = s * m + j;
      const int nC = flat.n_centres[idx];
      const int d = flat.d[idx];
      const double* centres = flat.centres.data() + flat.centre_off[idx];
      const double* mu = flat.mus.data() + flat.mu_off[idx];
      dim0.assign(flat.dims.begin() + flat.dim_off[idx], flat.dims.begin() + flat.dim_off[idx] + d);

      reassign(x_row, n, p, centres, nC, d, dim0, AssignmentDelta::FullRecompute, 0, euclidean,
               metric_red, member_red, ncats, AssignmentCache{}, cache, assign_scratch);

      for (int obs = 0; obs < n; ++obs) {
        draw_pred[obs] += mu[cache.assignment[obs]];
      }
    }

    for (int obs = 0; obs < n; ++obs) {
      out_view(obs, s) = draw_pred[obs];
    }
  }

  return out;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "Standalone C++ backend for the AddiVortes Python package.";
  m.def("run_mcmc",
        &run_mcmc,
        py::arg("x_scaled"),
        py::arg("y_scaled"),
        py::arg("metric"),
        py::arg("members"),
        py::arg("n_tessellations"),
        py::arg("total_iter"),
        py::arg("burn_in"),
        py::arg("thinning"),
        py::arg("nu"),
        py::arg("lambda_value"),
        py::arg("sigma_squared_mu"),
        py::arg("omega"),
        py::arg("lambda_rate"),
        py::arg("proposal_sd"),
        py::arg("proposal_mu"),
        py::arg("init_tess"),
        py::arg("init_dim"),
        py::arg("init_pred"),
        py::arg("binary_cols"),
        py::arg("cat_scaling"),
        py::arg("seed"),
        py::arg("verbose"));
  m.def("cell_indices",
        &cell_indices,
        py::arg("query"),
        py::arg("centres"),
        py::arg("dim"),
        py::arg("metric_red"),
        py::arg("member_red"),
        py::arg("ncats") = py::none());
  m.def("predict_ensemble",
        &predict_ensemble,
        py::arg("x"),
        py::arg("posterior_tess"),
        py::arg("posterior_dim"),
        py::arg("posterior_pred"),
        py::arg("metric_red"),
        py::arg("member_red"),
        py::arg("ncats") = py::none(),
        py::arg("verbose") = false);
  m.def("log_acceptance_structure",
        &log_structure_and_selection,
        py::arg("d_new"),
        py::arg("n_centres_new"),
        py::arg("sigma_squared"),
        py::arg("omega"),
        py::arg("lambda_rate"),
        py::arg("p"),
        py::arg("modification"));
}
