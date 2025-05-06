# Problem 2
# Estimating Pi Using Monte Carlo Methods
## Motivation
Monte Carlo simulations are a powerful computational technique that use randomness to solve problems or estimate values. One elegant application of Monte Carlo methods is estimating the value of $\pi$ using geometric probability. By randomly generating points and analyzing their positions relative to a geometric shape, we can approximate $\pi$ in an intuitive and visually engaging way.
This method connects fundamental concepts in probability, geometry, and numerical computation. The Monte Carlo approach for estimating $\pi$ highlights the versatility and simplicity of randomness while offering insights into convergence rates and computational efficiency.
---
## Part 1: Estimating $\pi$ Using a Circle
### 1. Theoretical Foundation
The idea behind using the circle-based Monte Carlo method to estimate $\pi$ relies on the ratio of points inside a circle to the total number of points in a square. Consider a unit circle inscribed inside a square. The unit circle is centered at the origin with the equation:
$$
x^2 + y^2 \leq 1
$$
This circle is inscribed within a square of side length 2. If we randomly generate points $(x, y)$ within the square (with $x, y \in [-1, 1]$), we can check whether each point lies inside the circle using the equation above.
The ratio of the area of the circle to the area of the square is:
$$
\frac{A_{\text{circle}}}{A_{\text{square}}} = \frac{\pi}{4}
$$
Thus, the ratio of points inside the circle to the total number of points will approximate the ratio of the areas, allowing us to estimate $\pi$. Specifically:
$$
\pi \approx 4 \times \frac{\text{Number of points inside the circle}}{\text{Total number of points}}
$$
### 2. Simulation
We'll simulate the process by generating random points in the square and counting how many fall inside the unit circle.
#### Python Code for Circle-Based Monte Carlo Method:
```python
import numpy as np
import matplotlib.pyplot as plt
def estimate_pi_advanced(num_points):
   # Generate random points in a 2D space [-1, 1] x [-1, 1]
   x = np.random.uniform(-1, 1, num_points)
   y = np.random.uniform(-1, 1, num_points)
   # Calculate the number of points inside the unit circle
   inside_circle = np.sum(x**2 + y**2 <= 1)
   # Estimate Pi
   pi_estimate = 4 * inside_circle / num_points
   # Plotting
   plt.figure(figsize=(6, 6))
   plt.scatter(x, y, color='blue', s=1, alpha=0.5, label="Points")
   plt.scatter(x[x**2 + y**2 <= 1], y[x**2 + y**2 <= 1], color='green', s=1, label="Inside Circle")
   plt.scatter(x[x**2 + y**2 > 1], y[x**2 + y**2 > 1], color='red', s=1, label="Outside Circle")
   plt.title(f"Monte Carlo Estimation of Pi: {pi_estimate}")
   plt.gca().set_aspect('equal', adjustable='box')
   plt.legend()
   plt.show()
   return pi_estimate
# Running the simulation with a larger number of points
num_points = 100000
pi_estimate = estimate_pi_advanced(num_points)
print(f"Estimated Pi: {pi_estimate}")
