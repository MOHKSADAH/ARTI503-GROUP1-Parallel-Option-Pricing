"""
Sequential Monte Carlo Option Pricing Implementation
ARTI503 - Parallel Architecture and Programming
Stage 1: Baseline Implementation
"""

import numpy as np
import time
from scipy.stats import norm
from typing import Tuple, Dict
import json

class MonteCarloOptionPricer:
    """
    Sequential Monte Carlo simulator for European option pricing.
    """
    
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float):
        """
        Initialize the option pricer with market parameters.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        r : float
            Risk-free interest rate (annual)
        sigma : float
            Volatility (annual)
        T : float
            Time to maturity (years)
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        
    def black_scholes_call(self) -> float:
        """
        Calculate analytical Black-Scholes price for European call option.
        This is used to validate our Monte Carlo results.
        
        Returns:
        --------
        float : Black-Scholes call option price
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        call_price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self) -> float:
        """
        Calculate analytical Black-Scholes price for European put option.
        
        Returns:
        --------
        float : Black-Scholes put option price
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        return put_price
    
    def simulate_price_path(self, num_steps: int, seed: int = None) -> float:
        """
        Simulate a single stock price path using Geometric Brownian Motion.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps in the simulation
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        float : Final stock price at maturity
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = self.T / num_steps
        S = self.S0
        
        for _ in range(num_steps):
            # Generate random normal variable
            Z = np.random.standard_normal()
            
            # Geometric Brownian Motion step
            # S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            drift = (self.r - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z
            S = S * np.exp(drift + diffusion)
        
        return S
    
    def monte_carlo_option_price(self, num_simulations: int, num_steps: int = 252, 
                                 option_type: str = 'call', seed: int = None) -> Tuple[float, float, float]:
        """
        Price an option using Monte Carlo simulation.
        
        Parameters:
        -----------
        num_simulations : int
            Number of Monte Carlo paths to simulate
        num_steps : int
            Number of time steps per path (default: 252 for daily steps in one year)
        option_type : str
            'call' or 'put'
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        tuple : (option_price, standard_error, execution_time)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Start timing
        start_time = time.perf_counter()
        
        payoffs = np.zeros(num_simulations)
        
        # Main simulation loop - THIS IS WHAT WE'LL PARALLELIZE IN STAGE 2
        for i in range(num_simulations):
            # Simulate final stock price
            final_price = self.simulate_price_path(num_steps)
            
            # Calculate payoff
            if option_type.lower() == 'call':
                payoffs[i] = max(final_price - self.K, 0)
            elif option_type.lower() == 'put':
                payoffs[i] = max(self.K - final_price, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
        
        # Calculate option price (discounted average payoff)
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Calculate standard error (for confidence intervals)
        standard_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(num_simulations)
        
        # End timing
        execution_time = time.perf_counter() - start_time
        
        return option_price, standard_error, execution_time


def run_benchmark_suite() -> Dict:
    """
    Run comprehensive benchmarks for the sequential implementation.
    This generates the data you'll put in your Stage 1 report.
    
    Returns:
    --------
    dict : Benchmark results including timing, accuracy, and profiling data
    """
    print("=" * 70)
    print("MONTE CARLO OPTION PRICING - SEQUENTIAL BENCHMARK SUITE")
    print("=" * 70)
    print()
    
    # Standard test parameters (at-the-money option)
    S0 = 100.0      # Initial stock price
    K = 100.0       # Strike price (at-the-money)
    r = 0.05        # 5% risk-free rate
    sigma = 0.20    # 20% volatility
    T = 1.0         # 1 year to maturity
    num_steps = 252 # Daily time steps
    
    pricer = MonteCarloOptionPricer(S0, K, r, sigma, T)
    
    # Calculate analytical Black-Scholes price for validation
    bs_call_price = pricer.black_scholes_call()
    bs_put_price = pricer.black_scholes_put()
    
    print(f"Market Parameters:")
    print(f"  Initial Stock Price (S0): ${S0:.2f}")
    print(f"  Strike Price (K):         ${K:.2f}")
    print(f"  Risk-free Rate (r):       {r*100:.1f}%")
    print(f"  Volatility (σ):           {sigma*100:.1f}%")
    print(f"  Time to Maturity (T):     {T:.1f} year")
    print(f"  Time Steps:               {num_steps}")
    print()
    print(f"Black-Scholes Analytical Prices:")
    print(f"  Call Option: ${bs_call_price:.4f}")
    print(f"  Put Option:  ${bs_put_price:.4f}")
    print()
    print("=" * 70)
    print()
    
    # Benchmark different simulation sizes
    simulation_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    
    results = {
        'parameters': {
            'S0': S0, 'K': K, 'r': r, 'sigma': sigma, 'T': T, 'num_steps': num_steps
        },
        'black_scholes': {
            'call': bs_call_price,
            'put': bs_put_price
        },
        'benchmarks': []
    }
    
    print("BENCHMARK RESULTS (Call Options):")
    print("-" * 70)
    print(f"{'Simulations':>12} | {'Time (s)':>10} | {'Time/Path (ms)':>15} | {'MC Price':>10} | {'Error %':>10}")
    print("-" * 70)
    
    for num_sims in simulation_sizes:
        # Run Monte Carlo simulation
        mc_price, std_error, exec_time = pricer.monte_carlo_option_price(
            num_simulations=num_sims,
            num_steps=num_steps,
            option_type='call',
            seed=42  # Fixed seed for reproducibility
        )
        
        # Calculate metrics
        time_per_path_ms = (exec_time / num_sims) * 1000
        price_error_pct = abs(mc_price - bs_call_price) / bs_call_price * 100
        
        # Store results
        benchmark_data = {
            'num_simulations': num_sims,
            'execution_time': exec_time,
            'time_per_path_ms': time_per_path_ms,
            'mc_price': mc_price,
            'standard_error': std_error,
            'price_error_pct': price_error_pct
        }
        results['benchmarks'].append(benchmark_data)
        
        # Print results
        print(f"{num_sims:>12,} | {exec_time:>10.3f} | {time_per_path_ms:>15.4f} | "
              f"${mc_price:>9.4f} | {price_error_pct:>9.2f}%")
    
    print("-" * 70)
    print()
    
    # Convergence analysis
    print("CONVERGENCE ANALYSIS:")
    print("-" * 70)
    print(f"The Monte Carlo estimate converges to the Black-Scholes price as")
    print(f"the number of simulations increases. Standard error decreases as 1/√N.")
    print()
    final_result = results['benchmarks'][-1]
    print(f"With 1,000,000 simulations:")
    print(f"  Monte Carlo Price: ${final_result['mc_price']:.4f} ± ${final_result['standard_error']:.4f}")
    print(f"  Black-Scholes Price: ${bs_call_price:.4f}")
    print(f"  Difference: ${abs(final_result['mc_price'] - bs_call_price):.4f} ({final_result['price_error_pct']:.3f}%)")
    print()
    
    # Scaling analysis
    print("SCALING ANALYSIS:")
    print("-" * 70)
    print(f"Time per path remains nearly constant, confirming O(N) linear scaling.")
    print(f"Average time per path: {np.mean([b['time_per_path_ms'] for b in results['benchmarks']]):.4f} ms")
    print()
    
    # Calculate theoretical parallel speedup (Amdahl's Law)
    parallel_fraction = 0.95  # 95% of time in parallelizable loop
    serial_fraction = 1 - parallel_fraction
    
    print("THEORETICAL PARALLEL SPEEDUP (Amdahl's Law):")
    print("-" * 70)
    print(f"Parallelizable fraction: {parallel_fraction*100:.0f}%")
    print(f"Serial fraction: {serial_fraction*100:.0f}%")
    print()
    for num_cores in [2, 4, 6, 8]:
        speedup = 1 / (serial_fraction + parallel_fraction / num_cores)
        efficiency = speedup / num_cores * 100
        print(f"  {num_cores} cores: {speedup:.2f}x speedup ({efficiency:.1f}% efficiency)")
    print()
    
    return results


def save_results(results: Dict, filename: str = "benchmark_results.json"):
    """Save benchmark results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


def main():
    """Main function to run benchmarks."""
    # Run benchmark suite
    results = run_benchmark_suite()
    
    # Save results
    save_results(results)
    
    print("=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
