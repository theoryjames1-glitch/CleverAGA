# CleverAGA

# Adaptive Genetic Algorithm (AGA)

---

## Taxonomy

Adaptive Genetic Algorithm (AGA) is a **stochastic, population-based, metaheuristic** for numerical and combinatorial optimization.
It belongs to the class of **Evolutionary Algorithms**, closely related to:

* Genetic Algorithms (GA)
* Evolution Strategies (ES)
* Natural Evolution Strategies (NES)
* Covariance Matrix Adaptation (CMA-ES)

---

## Inspiration

AGA is inspired by **biological evolution**, but with a twist:

* Instead of modeling discrete genetics (crossover/mutation),
* It models **adaptive search distributions** that evolve over time.
  The idea comes from viewing evolution not as inheritance of genes, but as **feedback-driven adaptation of probability laws**.

---

## Strategy

* Maintain a **probability distribution** over candidate solutions, defined by mean vector (μ) and variance (σ²).
* Each iteration:

  1. Sample a population of candidate solutions (perturbations).
  2. Evaluate them on the objective function.
  3. Assign utilities (weights) based on relative performance.
  4. Update μ (solution estimate) toward better candidates.
  5. Adapt σ to balance exploration (diversity) and exploitation (precision).

This makes AGA a form of **Evolutionary Gradient Descent**:

* Equivalent to gradient descent in distribution space,
* But without needing explicit derivatives.

---

## Procedure

```
Adaptive Genetic Algorithm (AGA)

1. Initialize mean μ and variance σ for parameters
2. Repeat until stopping condition:
    a. Sample N candidate solutions θ_i = μ + σ * ε_i
    b. Evaluate each candidate L(θ_i)
    c. Assign weights w_i (rank-based or softmax over losses)
    d. Update mean:
           μ ← Σ w_i θ_i
    e. Update variance:
           if diversity high → increase σ
           if diversity low  → decrease σ
3. Return best found solution
```

---

## Heuristics

* **Population size (N):** Larger N gives smoother updates, but slower per iteration. Typical: 50–200.
* **Initialization:** Start with σ large enough to cover the search space.
* **Weighting:** Rank-based utilities are more stable than raw losses.
* **Adaptation:** Use multiplicative updates for σ to ensure positivity.
* **Termination:** Stop when σ shrinks below threshold or max iterations reached.
* **Hybridization:** Combine with SGD or Adam for faster local convergence.

---

## Code (Python-like pseudocode)

```python
def AGA(f, mu, sigma=1.0, N=100, steps=1000):
    for t in range(steps):
        eps = np.random.randn(N, len(mu))
        thetas = mu + sigma * eps
        losses = np.array([f(theta) for theta in thetas])

        # Rank-based utilities
        ranks = np.argsort(np.argsort(losses))
        utilities = (len(losses)-1 - ranks) - (len(losses)-1)/2
        utilities = utilities / (utilities.std() + 1e-8)

        # Update mean
        grad_mu = np.mean(utilities[:,None] * eps, axis=0) / sigma
        mu = mu + 0.1 * grad_mu

        # Update sigma
        grad_sigma = np.mean(utilities * (eps**2 - 1), axis=0)
        sigma = sigma * np.exp(0.05 * grad_sigma)

        sigma = np.clip(sigma, 1e-6, 5.0)
    return mu
```

---

## References

* Hansen, N., and Ostermeier, A. (2001). “Completely derandomized self-adaptation in evolution strategies.” Evolutionary Computation.
* Wierstra, D. et al. (2008). “Natural Evolution Strategies.” JMLR.
* Brownlee, J. (2011). *Clever Algorithms: Nature-Inspired Programming Recipes*.

---

✨ In short: AGA is **Genetic Algorithm reimagined as a control system** — no crossover, no fixed mutation rate, just a **probabilistic adaptive filter** that learns how to explore and exploit.

