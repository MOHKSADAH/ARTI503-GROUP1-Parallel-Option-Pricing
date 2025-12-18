

import numpy as np
import time
from scipy.stats import norm
from typing import Tuple, Dict, List
import json
from multiprocessing import Pool, cpu_count
import os

class ParallelMonteCarloOptionPricer:
    """
    Parallel Monte Carlo simulator for European option pricing.
    
    This class extends the sequential implementation with parallel processing using
    the REDUCTION pattern, which was identified as the most efficient approach in
    Assignment 2 (achieving 3.63x speedup with 90.7% efficiency on 4 cores).
    
    Key Design Decisions:
    ---------------------
    1. REDUCTION PATTERN: Each worker computes payoffs independently and returns results.
       No shared variables, no locks, no synchronization overhead.
    
    2. INDEPENDENT RNG STREAMS: Each worker gets a unique seed to ensure:
       - Statistical independence between paths
       - Reproducibility of results
       - Thread-safe random number generation
    
    3. WORKLOAD DISTRIBUTION: Paths are evenly distributed across workers using
       multiprocessing.Pool.map() for automatic load balancing.
    """
    
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float):
        """
        Initialize the parallel option pricer with market parameters.
        
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
        """Calculate analytical Black-Scholes price for validation."""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        call_price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self) -> float:
        """Calculate analytical Black-Scholes price for validation."""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        return put_price
    
    def _simulate_single_path(self, args: Tuple) -> float:
        """
        Worker function: Simulate a single path and return its payoff.
        
        This is the core parallelized function. Each worker independently:
        1. Generates its own random number stream (using unique seed)
        2. Simulates the price path using Geometric Brownian Motion
        3. Calculates and returns the payoff
        
        NO SHARED VARIABLES - This is the key to the reduction pattern's efficiency.
        
        Parameters:
        -----------
        args : tuple
            (path_index, num_steps, option_type, base_seed)
            
        Returns:
        --------
        float : Option payoff for this path
        """
        path_index, num_steps, option_type, base_seed = args
        
        # Create unique seed for this path to ensure statistical independence
        # Each worker gets a different seed based on path_index
        np.random.seed(base_seed + path_index)
        
        # Time step size
        dt = self.T / num_steps
        
        # Initial stock price
        S = self.S0
        
        # Simulate price path using Geometric Brownian Motion
        # This is the computational hotspot (60% of execution time from profiling)
        for _ in range(num_steps):
            # Generate random normal variable
            Z = np.random.standard_normal()
            
            # GBM discretization: S(t+dt) = S(t) * exp((r - 0.5*σ²)*dt + σ*√dt*Z)
            drift = (self.r - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z
            S = S * np.exp(drift + diffusion)
        
        # Calculate payoff based on option type
        if option_type == 'call':
            payoff = max(S - self.K, 0.0)
        else:  # put
            payoff = max(self.K - S, 0.0)
        
        return payoff
    
    def monte_carlo_option_price_parallel(
        self, 
        num_simulations: int, 
        num_cores: int = None,
        num_steps: int = 252, 
        option_type: str = 'call', 
        seed: int = 42
    ) -> Tuple[float, float, float, int]:
        """
        Price an option using PARALLEL Monte Carlo simulation with REDUCTION pattern.
        
        Parallelization Strategy:
        -------------------------
        1. Divide num_simulations paths across num_cores workers
        2. Each worker computes payoffs for its assigned paths independently
        3. Collect all payoffs and perform final reduction (sum) in main process
        4. No synchronization during simulation - all reduction happens at the end
        
        This approach achieves near-linear speedup because:
        - Zero synchronization overhead during computation
        - Perfect load balancing (paths distributed evenly)
        - No lock contention or race conditions
        - Minimal serial fraction (< 5% as measured in Stage 1)
        
        Parameters:
        -----------
        num_simulations : int
            Total number of Monte Carlo paths to simulate
        num_cores : int, optional
            Number of parallel workers (default: all available cores)
        num_steps : int
            Number of time steps per path (default: 252 for daily steps)
        option_type : str
            'call' or 'put'
        seed : int
            Base random seed for reproducibility
            
        Returns:
        --------
        tuple : (option_price, standard_error, execution_time, cores_used)
        """
        # Determine number of cores to use
        if num_cores is None:
            num_cores = cpu_count()
        else:
            num_cores = min(num_cores, cpu_count())
        
        # Start timing
        start_time = time.perf_counter()
        
        # Prepare arguments for each path simulation
        # Each tuple contains: (path_index, num_steps, option_type, base_seed)
        args_list = [
            (i, num_steps, option_type, seed)
            for i in range(num_simulations)
        ]
        
        # PARALLEL EXECUTION USING REDUCTION PATTERN
        # Create a pool of worker processes
        with Pool(processes=num_cores) as pool:
            # Map the simulation function across all paths
            # pool.map automatically distributes work and collects results
            payoffs = pool.map(self._simulate_single_path, args_list)
        
        # REDUCTION STEP: Aggregate results (this is the only serial part)
        # Convert to numpy array for efficient computation
        payoffs = np.array(payoffs)
        
        # Calculate option price (discounted average payoff)
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Calculate standard error for confidence intervals
        standard_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(num_simulations)
        
        # End timing
        execution_time = time.perf_counter() - start_time
        
        return option_price, standard_error, execution_time, num_cores


def run_parallel_benchmark_suite() -> Dict:
    """
    Run comprehensive parallel benchmarks across different core counts and simulation sizes.
    
    This function systematically tests:
    1. Different numbers of cores (1, 2, 4, 6, 8) for scalability analysis
    2. Different simulation sizes (10K to 1M paths) for workload analysis
    3. Calculates speedup and efficiency relative to sequential baseline
    
    Returns:
    --------
    dict : Complete benchmark results including speedup and efficiency metrics
    """
    print("=" * 80)
    print("PARALLEL MONTE CARLO OPTION PRICING - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print()
    
    # Standard test parameters (same as Stage 1 for direct comparison)
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0
    num_steps = 252
    
    pricer = ParallelMonteCarloOptionPricer(S0, K, r, sigma, T)
    
    # Display system information
    max_cores = cpu_count()
    print(f"System Information:")
    print(f"  Available CPU Cores: {max_cores}")
    print(f"  Python Version: {os.sys.version.split()[0]}")
    print()
    
    # Display market parameters
    bs_call_price = pricer.black_scholes_call()
    print(f"Market Parameters:")
    print(f"  Initial Stock Price (S0): ${S0:.2f}")
    print(f"  Strike Price (K):         ${K:.2f}")
    print(f"  Risk-free Rate (r):       {r*100:.1f}%")
    print(f"  Volatility (σ):           {sigma*100:.1f}%")
    print(f"  Time to Maturity (T):     {T:.1f} year")
    print(f"  Time Steps per Path:      {num_steps}")
    print(f"  Black-Scholes Price:      ${bs_call_price:.4f}")
    print()
    print("=" * 80)
    print()
    
    # Test configurations
    simulation_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    core_counts = [1, 2, 4, 6, 8]
    core_counts = [c for c in core_counts if c <= max_cores]  # Only test available cores
    
    results = {
        'parameters': {
            'S0': S0, 'K': K, 'r': r, 'sigma': sigma, 'T': T, 'num_steps': num_steps
        },
        'system': {
            'max_cores': max_cores,
            'tested_cores': core_counts
        },
        'black_scholes': {
            'call': bs_call_price
        },
        'benchmarks': []
    }
    
    # Run benchmarks for each simulation size
    for num_sims in simulation_sizes:
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {num_sims:,} Simulation Paths")
        print(f"{'='*80}")
        print()
        print(f"{'Cores':>6} | {'Time (s)':>10} | {'Speedup':>8} | {'Efficiency':>10} | {'MC Price':>10} | {'Error %':>8}")
        print("-" * 80)
        
        benchmark_data = {
            'num_simulations': num_sims,
            'results_by_cores': []
        }
        
        # Store baseline (1 core) time for speedup calculation
        baseline_time = None
        
        # Test each core count
        for num_cores in core_counts:
            # Run parallel simulation
            mc_price, std_error, exec_time, cores_used = pricer.monte_carlo_option_price_parallel(
                num_simulations=num_sims,
                num_cores=num_cores,
                num_steps=num_steps,
                option_type='call',
                seed=42
            )
            
            # Calculate metrics
            if baseline_time is None:
                baseline_time = exec_time
                speedup = 1.0
            else:
                speedup = baseline_time / exec_time
            
            efficiency = (speedup / num_cores) * 100
            price_error_pct = abs(mc_price - bs_call_price) / bs_call_price * 100
            
            # Store results
            core_result = {
                'num_cores': num_cores,
                'execution_time': exec_time,
                'speedup': speedup,
                'efficiency': efficiency,
                'mc_price': mc_price,
                'standard_error': std_error,
                'price_error_pct': price_error_pct
            }
            benchmark_data['results_by_cores'].append(core_result)
            
            # Print results
            print(f"{num_cores:>6} | {exec_time:>10.3f} | {speedup:>8.2f}x | {efficiency:>9.1f}% | "
                  f"${mc_price:>9.4f} | {price_error_pct:>7.2f}%")
        
        results['benchmarks'].append(benchmark_data)
        print()
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print()
    print("Best Speedup Achieved:")
    for benchmark in results['benchmarks']:
        num_sims = benchmark['num_simulations']
        best_result = max(benchmark['results_by_cores'], key=lambda x: x['speedup'])
        print(f"  {num_sims:>10,} paths: {best_result['speedup']:.2f}x on "
              f"{best_result['num_cores']} cores ({best_result['efficiency']:.1f}% efficiency)")
    
    return results


def save_benchmark_results(results: Dict, filename: str = "parallel_benchmark_results.json"):
    """Save comprehensive benchmark results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filename}")


def generate_comparison_report(results: Dict):
    """
    Generate a formatted comparison report for the paper.
    
    This creates LaTeX-ready tables comparing parallel vs sequential performance.
    """
    print("\n" + "="*80)
    print("LATEX TABLE GENERATION")
    print("="*80)
    print()
    
    print("Table: Parallel Performance Summary (1M Simulations)")
    print("-" * 80)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|r|r|r|r|}")
    print("\\hline")
    print("\\textbf{Cores} & \\textbf{Time (s)} & \\textbf{Speedup} & \\textbf{Efficiency (\\%)} \\\\")
    print("\\hline")
    
    # Get 1M simulation results
    for benchmark in results['benchmarks']:
        if benchmark['num_simulations'] == 1_000_000:
            for core_result in benchmark['results_by_cores']:
                print(f"{core_result['num_cores']} & "
                      f"{core_result['execution_time']:.2f} & "
                      f"{core_result['speedup']:.2f}x & "
                      f"{core_result['efficiency']:.1f}\\% \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Parallel Performance Across Different Core Counts}")
    print("\\end{table}")
    print()


def main():
    """Main function to run comprehensive parallel benchmarks."""
    print()
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "STAGE 3: PARALLEL IMPLEMENTATION" + " "*26 + "║")
    print("║" + " "*15 + "Monte Carlo Option Pricing - Reduction Pattern" + " "*16 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    # Run comprehensive benchmark suite
    results = run_parallel_benchmark_suite()
    
    # Save results
    save_benchmark_results(results)
    
    # Generate comparison report
    generate_comparison_report(results)
if __name__ == "__main__":
    main()
