// Extended QuantLib C++ Benchmark
// Adds Monte Carlo, SABR calibration, and batch pricing
//
// Compile with:
// clang++ -std=c++17 -O3 -I$HOME/dev/QuantLib -L$HOME/dev/QuantLib/build/ql \
//         -lQuantLib -o quantlib_benchmark_extended quantlib_benchmark_extended.cpp
//
// Run with:
// DYLD_LIBRARY_PATH=$HOME/dev/QuantLib/build/ql ./quantlib_benchmark_extended

#include <ql/quantlib.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>

using namespace QuantLib;
using namespace std::chrono;

// =============================================================================
// Timing Utility
// =============================================================================

template<typename Func>
std::pair<double, double> benchmark(Func f, int n_runs = 1000, int n_warmup = 100) {
    for (int i = 0; i < n_warmup; ++i) f();

    std::vector<double> times;
    times.reserve(n_runs);

    for (int i = 0; i < n_runs; ++i) {
        auto start = high_resolution_clock::now();
        f();
        auto end = high_resolution_clock::now();
        double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;
        times.push_back(us);
    }

    std::sort(times.begin(), times.end());
    double median = times[n_runs / 2];

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / n_runs;
    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double std_dev = std::sqrt(sq_sum / n_runs);

    return {median, std_dev};
}

// =============================================================================
// European Option Pricing (from original benchmark)
// =============================================================================

double european_price(double S, double K, double T, double r, double sigma, Option::Type type) {
    Date today(1, January, 2025);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(T * 365);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(S));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, r, Actual365Fixed()));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(today, 0.0, Actual365Fixed()));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, Actual365Fixed()));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, K);
    auto exercise = ext::make_shared<EuropeanExercise>(maturity);
    VanillaOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));

    return option.NPV();
}

// =============================================================================
// Monte Carlo European Option
// =============================================================================

double mc_european_price(double S, double K, double T, double r, double sigma,
                         Option::Type type, int npaths, int nsteps) {
    Date today(1, January, 2025);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(T * 365);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(S));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, r, Actual365Fixed()));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(today, 0.0, Actual365Fixed()));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, Actual365Fixed()));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, K);
    auto exercise = ext::make_shared<EuropeanExercise>(maturity);
    VanillaOption option(payoff, exercise);

    auto rng = ext::make_shared<MersenneTwisterUniformRng>(42);
    auto engine = ext::make_shared<MCEuropeanEngine<PseudoRandom>>(
        process, nsteps, Null<Size>(), false, false, npaths, 1e-6, Null<Size>(), 42);
    option.setPricingEngine(engine);

    return option.NPV();
}

// =============================================================================
// Monte Carlo Asian Option
// =============================================================================

double mc_asian_price(double S, double K, double T, double r, double sigma,
                      Option::Type type, int npaths) {
    Date today(1, January, 2025);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(T * 365);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(S));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, r, Actual365Fixed()));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(today, 0.0, Actual365Fixed()));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, Actual365Fixed()));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);

    Average::Type avgType = Average::Arithmetic;
    Real runningSum = 0.0;
    Size pastFixings = 0;
    std::vector<Date> fixingDates;
    int nFixings = 12;  // Monthly fixings
    for (int i = 1; i <= nFixings; ++i) {
        fixingDates.push_back(today + Integer(i * T * 365 / nFixings));
    }

    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, K);
    auto exercise = ext::make_shared<EuropeanExercise>(maturity);

    DiscreteAveragingAsianOption option(avgType, runningSum, pastFixings,
                                         fixingDates, payoff, exercise);

    auto engine = ext::make_shared<MCDiscreteArithmeticAPEngine<PseudoRandom>>(
        process, false, false, false, npaths, 1e-6, Null<Size>(), 42);
    option.setPricingEngine(engine);

    return option.NPV();
}

// =============================================================================
// American Option (Binomial Tree)
// =============================================================================

double american_price(double S, double K, double T, double r, double sigma,
                      Option::Type type, int nsteps = 100) {
    Date today(1, January, 2025);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(T * 365);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(S));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, r, Actual365Fixed()));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(today, 0.0, Actual365Fixed()));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, Actual365Fixed()));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);
    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, K);
    auto exercise = ext::make_shared<AmericanExercise>(today, maturity);
    VanillaOption option(payoff, exercise);

    option.setPricingEngine(
        ext::make_shared<BinomialVanillaEngine<CoxRossRubinstein>>(process, nsteps));

    return option.NPV();
}

// =============================================================================
// Batch European Pricing
// =============================================================================

std::vector<double> batch_european_price(double S, double r,
                                          const std::vector<double>& Ks,
                                          const std::vector<double>& Ts,
                                          const std::vector<double>& sigmas) {
    size_t n = Ks.size();
    std::vector<double> prices(n);
    for (size_t i = 0; i < n; ++i) {
        prices[i] = european_price(S, Ks[i], Ts[i], r, sigmas[i], Option::Call);
    }
    return prices;
}

// =============================================================================
// SABR Implied Volatility
// =============================================================================

double sabr_vol(double F, double K, double T, double alpha, double beta, double rho, double nu) {
    return sabrVolatility(K, F, T, alpha, beta, nu, rho);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "======================================================================\n";
    std::cout << "QUANTLIB C++ EXTENDED BENCHMARK\n";
    std::cout << "======================================================================\n";
    std::cout << "QuantLib version: " << QL_VERSION << "\n\n";

    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.2;

    // -------------------------------------------------------------------------
    // European Pricing (Analytic)
    // -------------------------------------------------------------------------
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "EUROPEAN OPTION PRICING (Analytic)\n";
    std::cout << "----------------------------------------------------------------------\n";

    double eu_price = european_price(S, K, T, r, sigma, Option::Call);
    std::cout << "Price: " << std::setprecision(6) << eu_price << "\n";

    auto [eu_median, eu_std] = benchmark([&]() {
        european_price(S, K, T, r, sigma, Option::Call);
    }, 1000, 100);

    std::cout << std::setprecision(2);
    std::cout << "Timing (1000 runs): " << eu_median << " μs (median)\n\n";

    // -------------------------------------------------------------------------
    // American Option (Binomial 100-step)
    // -------------------------------------------------------------------------
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "AMERICAN OPTION PRICING (100-step binomial)\n";
    std::cout << "----------------------------------------------------------------------\n";

    double am_price = american_price(S, K, T, r, sigma, Option::Put, 100);
    std::cout << std::setprecision(6) << "Price: " << am_price << "\n";

    auto [am_median, am_std] = benchmark([&]() {
        american_price(S, K, T, r, sigma, Option::Put, 100);
    }, 500, 50);

    std::cout << std::setprecision(2);
    std::cout << "Timing (500 runs): " << am_median << " μs (median)\n\n";

    // -------------------------------------------------------------------------
    // Monte Carlo European (10k paths)
    // -------------------------------------------------------------------------
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "MONTE CARLO EUROPEAN (10,000 paths, 50 steps)\n";
    std::cout << "----------------------------------------------------------------------\n";

    double mc_price = mc_european_price(S, K, T, r, sigma, Option::Call, 10000, 50);
    std::cout << std::setprecision(6) << "Price: " << mc_price << "\n";

    auto [mc_median, mc_std] = benchmark([&]() {
        mc_european_price(S, K, T, r, sigma, Option::Call, 10000, 50);
    }, 20, 2);

    std::cout << std::setprecision(2);
    std::cout << "Timing (20 runs): " << mc_median / 1000 << " ms (median)\n\n";

    // -------------------------------------------------------------------------
    // Monte Carlo Asian (10k paths)
    // -------------------------------------------------------------------------
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "MONTE CARLO ASIAN (10,000 paths, 12 fixings)\n";
    std::cout << "----------------------------------------------------------------------\n";

    double asian_price = mc_asian_price(S, K, T, r, sigma, Option::Call, 10000);
    std::cout << std::setprecision(6) << "Price: " << asian_price << "\n";

    auto [asian_median, asian_std] = benchmark([&]() {
        mc_asian_price(S, K, T, r, sigma, Option::Call, 10000);
    }, 20, 2);

    std::cout << std::setprecision(2);
    std::cout << "Timing (20 runs): " << asian_median / 1000 << " ms (median)\n\n";

    // -------------------------------------------------------------------------
    // Batch Pricing (1000 options)
    // -------------------------------------------------------------------------
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "BATCH PRICING (1,000 European options)\n";
    std::cout << "----------------------------------------------------------------------\n";

    std::mt19937 gen(42);
    std::uniform_real_distribution<> K_dist(80.0, 120.0);
    std::uniform_real_distribution<> T_dist(0.1, 2.0);
    std::uniform_real_distribution<> sigma_dist(0.1, 0.5);

    std::vector<double> Ks(1000), Ts(1000), sigmas(1000);
    for (int i = 0; i < 1000; ++i) {
        Ks[i] = K_dist(gen);
        Ts[i] = T_dist(gen);
        sigmas[i] = sigma_dist(gen);
    }

    auto [batch_median, batch_std] = benchmark([&]() {
        batch_european_price(S, r, Ks, Ts, sigmas);
    }, 10, 2);

    std::cout << "Timing (10 runs): " << batch_median / 1000 << " ms (median)\n";
    std::cout << "Per-option: " << batch_median / 1000 << " μs\n\n";

    // -------------------------------------------------------------------------
    // SABR Implied Vol
    // -------------------------------------------------------------------------
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "SABR IMPLIED VOLATILITY\n";
    std::cout << "----------------------------------------------------------------------\n";

    double F = 100.0;
    double alpha = 0.2, beta = 0.5, rho = -0.3, nu = 0.4;
    double sabr_iv = sabr_vol(F, K, T, alpha, beta, rho, nu);
    std::cout << std::setprecision(6) << "Implied Vol: " << sabr_iv << "\n";

    auto [sabr_median, sabr_std] = benchmark([&]() {
        sabr_vol(F, K, T, alpha, beta, rho, nu);
    }, 10000, 1000);

    std::cout << std::setprecision(3);
    std::cout << "Timing (10000 runs): " << sabr_median << " μs (median)\n\n";

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::cout << "======================================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "======================================================================\n";
    std::cout << "  Benchmark                       QuantLib C++\n";
    std::cout << "  ─────────────────────────────────────────────\n";
    std::cout << std::setprecision(2);
    std::cout << "  European (analytic)             " << std::setw(10) << eu_median << " μs\n";
    std::cout << "  American (100-step binomial)    " << std::setw(10) << am_median << " μs\n";
    std::cout << "  MC European (10k paths)         " << std::setw(10) << mc_median/1000 << " ms\n";
    std::cout << "  MC Asian (10k paths)            " << std::setw(10) << asian_median/1000 << " ms\n";
    std::cout << "  Batch (1000 options)            " << std::setw(10) << batch_median/1000 << " ms\n";
    std::cout << std::setprecision(3);
    std::cout << "  SABR implied vol                " << std::setw(10) << sabr_median << " μs\n";
    std::cout << "======================================================================\n";

    return 0;
}
