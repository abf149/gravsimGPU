# Gravsim

# Purpose

Simulate Newtonian gravitation. Leverage GPU acceleration with CUDA.

# Prerequisites
* CUDA tools esp. nvcc
* NVIDIA GPU, NVIDIA RTX 2080 Ti utilized for this test

# Description

This tool models Newtonian gravitation and leverages GPU acceleration. 

*Differential equation solver:* Euler method
*Number of planets, simulated time per iteration:* Configurable in `dynamics.h`
*Number of sim iterations:* Configurable in `main.h`

## Simulation process

There are `n` planets. Each planet is a `struct` with the following fields (MKS):

(mass kg, x meters, y meters, z meters, vx m/s, vy m/s, vz m/s)

The whole system is an array `planets` in RAM of planet `struct`s

At startup, the `planets` array is initialized into some hard-coded planetary configuration.

Each iteration employs four modeling stages. Stages 2 and 3 employ two different GPU kernels for hardware acceleration.

### Stage 1: host => device copy ###

The `planets` array is copied from the CPU memory to the GPU memory.

### Stage 2: delta-v stage (acceleration)

#### High-level description:

1. Newton's law is used to estimate accelerations
2. Accelerations are combined with simulated time per iteration to get the velocity change for each planet

#### GPU acceleration approach:

Mathematical decomposition for GPU acceleration: 
* There is a gravitational interaction between each pair of planets
* Each pairwise interaction contributes an incremental acceleration on both planets in the pair
* The acceleration acting on each planet is the vector sum of the component accelerations of all its interactions
* Since the number of pairwise interactions grows as the square of the number of planets, it will be effective to parallelize at the granularity of pairwise interactions.

GPU kernel implementation:
* The pairwise interactions are enumerated and span the range `0..n*(n-1)/2-1`
* `n*(n-1)/2` rather than `n*(n+1)/2` because we exclude self-interaction
* A GPU thread is spun-up for each pairwise interaction, `n*(n-1)/2` in total
* A given GPU thread calculates an incremental acceleration for both planets in the pair, and multiples by the simulation timestep to get the incremental velocity change for each planet in the pair
* The GPU thread finally adds the incremental velocity adjustments back into the `planets` array

### Stage 3: delta-x stage (velocity)

We already updated velocity in the previous step; the interaction-heavy part is out of the way. In this stage each planet's position is now updated based on its new velocity, taking into account the simulation timestep.

Mathematical decomposition for GPU acceleration:
* Each planet's position-change is solely a function of its own velocity
* We can parallelize on a per-planet basis
* Each planet gets its own GPU thread

GPU kernel impleementation
* The planets are enumeration and span the range `0..n-1`
* A GPU thread is spun up for each planet
* The GPU thread calculates the new planet position as x_new = x + v*dt
* This update is written to the `planets` array

### Stage 4: device => host memory copy

The updated `planets` array is copied back to CPU memory. This allows you to analyze the simulation result at that iteration.

For a faster simulation, you could perform many iterations without copying the `planets` array from GPU memory back to host memory.
