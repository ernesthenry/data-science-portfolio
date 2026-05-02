import numpy as np
import scipy.stats as stats

def perform_z_test(rate_a, n_a, rate_b, n_b):
    """Perform a z-test for proportions."""
    p_combined = (rate_a * n_a + rate_b * n_b) / (n_a + n_b)
    std_error = np.sqrt(p_combined * (1 - p_combined) * (1/n_a + 1/n_b))
    z_score = (rate_b - rate_a) / (std_error + 1e-9)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) # Two-tailed
    return z_score, p_value

def bayesian_conversion_probability(a_conversions, a_size, b_conversions, b_size, prior_alpha=1, prior_beta=1, samples=10000):
    """Calculate the probability that B is better than A."""
    a_samples = np.random.beta(a_conversions + prior_alpha, a_size - a_conversions + prior_beta, samples)
    b_samples = np.random.beta(b_conversions + prior_alpha, b_size - b_conversions + prior_beta, samples)
    return np.mean(b_samples > a_samples)

def calculate_mde(baseline_conversion, power=0.8, alpha=0.05, sample_size=1000):
    """Calculate the Minimum Detectable Effect (MDE) for a given sample size."""
    # Simplified MDE calculation
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    variance = 2 * baseline_conversion * (1 - baseline_conversion)
    mde = (z_alpha + z_beta) * np.sqrt(variance / sample_size)
    
    return mde

if __name__ == "__main__":
    # Example
    z, p = frequentist_test(100, 1000, 130, 1000)
    print(f"Frequentist Result: Z={z:.2f}, P={p:.4f}")
    
    prob, lift = bayesian_probability(100, 1000, 130, 1000)
    print(f"Bayesian Result: Prob(B>A)={prob:.2%}, Lift={lift:.2%}")
