#include "mc_kernel.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace fastmc {

double Xorshift64::normal() {
    double u1 = uniform();
    double u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

MCResult mc_euro_call(
    double S0, double K, double T, double r, double q, double sigma,
    std::size_t n_paths, std::size_t n_steps,
    std::uint64_t seed,
    int n_threads
) {
    #ifdef HAS_OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    #endif

    const double dt = T / static_cast<double>(n_steps);
    const double drift = (r - q - 0.5 * sigma * sigma) * dt;
    const double diff = sigma * std::sqrt(dt);
    const double disc = std::exp(-r * T);

    const std::size_t half = n_paths / 2;
    // accumulate payoff sum and sum of squares in parallel (pairs: each pair = antithetic avg)
    double sum = 0.0;
    double sum_sq = 0.0;

    #ifdef HAS_OPENMP
    #pragma omp parallel reduction(+:sum,sum_sq)
    {
        int tid = omp_get_thread_num();
        Xorshift64 rng(seed + 0x9E3779B9ULL * (tid + 1));
        #pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(half); ++i) {
            double logS1 = 0.0;
            double logS2 = 0.0;
            for (std::size_t t = 0; t < n_steps; ++t) {
                double z = rng.normal();
                logS1 += drift + diff * z;
                logS2 += drift - diff * z;
            }
            double S1 = S0 * std::exp(logS1);
            double S2 = S0 * std::exp(logS2);
            double p = 0.5 * (std::max(S1 - K, 0.0) + std::max(S2 - K, 0.0));
            p *= disc;
            sum += p;
            sum_sq += p * p;
        }
    }
    #else
    Xorshift64 rng(seed);
    for (std::size_t i = 0; i < half; ++i) {
        double logS1 = 0.0;
        double logS2 = 0.0;
        for (std::size_t t = 0; t < n_steps; ++t) {
            double z = rng.normal();
            logS1 += drift + diff * z;
            logS2 += drift - diff * z;
        }
        double S1 = S0 * std::exp(logS1);
        double S2 = S0 * std::exp(logS2);
        double p = 0.5 * (std::max(S1 - K, 0.0) + std::max(S2 - K, 0.0));
        p *= disc;
        sum += p;
        sum_sq += p * p;
    }
    #endif

    const double n = static_cast<double>(half);
    const double mean = sum / n;
    const double var = (sum_sq / n - mean * mean) * n / std::max(n - 1.0, 1.0);
    const double stderr_ = std::sqrt(std::max(var / n, 0.0));
    return {mean, stderr_};
}

}  // namespace fastmc
