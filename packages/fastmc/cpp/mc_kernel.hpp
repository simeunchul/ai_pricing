#pragma once
#include <cstdint>
#include <cstddef>

namespace fastmc {

// PCG-like fast RNG (Xorshift64*)
struct Xorshift64 {
    std::uint64_t state;
    explicit Xorshift64(std::uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}

    inline std::uint64_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    // uniform in (0,1)
    inline double uniform() {
        return (next() >> 11) * (1.0 / 9007199254740992.0);
    }

    // Box-Muller
    inline double normal();
};

// European call MC price under GBM, antithetic. Returns (price, stderr).
struct MCResult {
    double price;
    double std_err;
};

MCResult mc_euro_call(
    double S0, double K, double T, double r, double q, double sigma,
    std::size_t n_paths, std::size_t n_steps,
    std::uint64_t seed,
    int n_threads
);

}  // namespace fastmc
