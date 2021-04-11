# Gravsim

# Purpose

Simulate Newtonian gravitation. Leverage GPU acceleration with CUDA.

# Prerequisites
* CUDA tools esp. nvcc
* NVIDIA GPU, NVIDIA RTX 2080 Ti utilized for this test

# Build command

```
nvcc -rdc=true dynamics_cuda.cu main.cu -o gravsim
```

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

## Simulation output

Currently the simulation is setup with a timestep of `1s` per iteration, and many months of iteration time (`26280000` iterations). The simulation is initialized with two planets:
* Earth/planet 0: 5.9722e24 kg, no velocity, at origin
* Moon/planet 1: 7.3477e22 kg, vy = 1028 m/s, x=384.4e6 meters

These numbers are true-to-life for the Earth-Moon system, and we would expect a roughly 30-day periodicity in the simulation. **Look at the velocity vector ("`v=...`") in the output below** and you will see that the Moon's velocity is roughly periodic with a nearly 30-day period.

Why isn't the position vector of the Earth or the Moon ("`x=...`") in the output below periodic? Conservation of momentum. The Moon is created with positive `vy` and the Earth is given no corresponding negative-`y` momentum, therefore the entire system is translating upward in simulation space. So the planet positions will not return to their initial places after ~30 days.

 

```
Creating planets...Done creating planets.Planet states, day 0:
- Planet 0: m=73479997433437751345152.000000, x=(384400000.000000,0.000000,0.000000), v=(0.000000,1027.777832,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(0.000000,0.000000,0.000000), v=(0.000000,0.000000,0.000000)
Planet states, day 2:
- Planet 0: m=73479997433437751345152.000000, x=(371015232.000000,101571832.000000,0.000000), v=(-266.441376,991.975769,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(164918.359375,14733.025391,0.000000), v=(3.278475,0.440867,0.000000)
Planet states, day 3:
- Planet 0: m=73479997433437751345152.000000, x=(331744672.000000,196097552.000000,0.000000), v=(-513.785217,887.186951,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(647871.625000,116540.382812,0.000000), v=(6.321939,1.730609,0.000000)
Planet states, day 4:
- Planet 0: m=73479997433437751345152.000000, x=(269489600.000000,277017824.000000,0.000000), v=(-724.415405,721.124146,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(1414142.750000,386032.500000,0.000000), v=(8.911172,3.773671,0.000000)
Planet states, day 5:
- Planet 0: m=73479997433437751345152.000000, x=(188590704.000000,338672192.000000,0.000000), v=(-883.435120,505.985352,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(2409067.500000,891447.687500,0.000000), v=(10.868766,6.419720,0.000000)
Planet states, day 6:
- Planet 0: m=73479997433437751345152.000000, x=(94865640.000000,377044800.000000,0.000000), v=(-980.196167,257.409027,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(3562597.500000,1683776.000000,0.000000), v=(12.058659,9.474282,0.000000)
Planet states, day 7:
- Planet 0: m=73479997433437751345152.000000, x=(-5143616.000000,389632640.000000,0.000000), v=(-1008.413513,-6.939897,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(4797433.000000,2793241.500000,0.000000), v=(12.406128,12.715958,0.000000)
Planet states, day 9:
- Planet 0: m=73479997433437751345152.000000, x=(-104486960.000000,375778336.000000,0.000000), v=(-966.918579,-268.600647,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(6023184.000000,4226550.500000,0.000000), v=(11.895376,15.930181,0.000000)
Planet states, day 10:
- Planet 0: m=73479997433437751345152.000000, x=(-196324656.000000,336589312.000000,0.000000), v=(-859.119629,-509.682556,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(7153023.500000,5970349.000000,0.000000), v=(10.568937,18.898281,0.000000)
Planet states, day 11:
- Planet 0: m=73479997433437751345152.000000, x=(-274398272.000000,275054656.000000,0.000000), v=(-693.046570,-714.052002,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(8113231.000000,7989905.500000,0.000000), v=(8.525568,21.407309,0.000000)
Planet states, day 12:
- Planet 0: m=73479997433437751345152.000000, x=(-333362432.000000,195453872.000000,0.000000), v=(-480.171722,-868.160095,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(8838218.000000,10228776.000000,0.000000), v=(5.906773,23.300644,0.000000)
Planet states, day 13:
- Planet 0: m=73479997433437751345152.000000, x=(-369360352.000000,103422616.000000,0.000000), v=(-235.245178,-961.900269,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(9280835.000000,12615712.000000,0.000000), v=(2.892553,24.454981,0.000000)
Planet states, day 14:
- Planet 0: m=73479997433437751345152.000000, x=(-379895488.000000,5295887.000000,0.000000), v=(25.402424,-989.398499,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(9411856.000000,15109812.000000,0.000000), v=(-0.314327,24.797199,0.000000)
Planet states, day 16:
- Planet 0: m=73479997433437751345152.000000, x=(-364314144.000000,-92185720.000000,0.000000), v=(284.346130,-948.805969,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(9222488.000000,17576508.000000,0.000000), v=(-3.500370,24.294813,0.000000)
Planet states, day 17:
- Planet 0: m=73479997433437751345152.000000, x=(-323683136.000000,-182295536.000000,0.000000), v=(524.299805,-842.787170,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(8721534.000000,19975240.000000,0.000000), v=(-6.452932,22.989117,0.000000)
Planet states, day 18:
- Planet 0: m=73479997433437751345152.000000, x=(-260631008.000000,-258789184.000000,0.000000), v=(729.110962,-678.067383,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(7942998.500000,22172152.000000,0.000000), v=(-8.974612,20.964664,0.000000)
Planet states, day 19:
- Planet 0: m=73479997433437751345152.000000, x=(-179470032.000000,-316338592.000000,0.000000), v=(884.738098,-465.466187,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(6944441.000000,24126036.000000,0.000000), v=(-10.888534,18.345247,0.000000)
Planet states, day 20:
- Planet 0: m=73479997433437751345152.000000, x=(-85682752.000000,-350743776.000000,0.000000), v=(980.205505,-219.098221,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(5789604.000000,25817410.000000,0.000000), v=(-12.062240,15.321961,0.000000)
Planet states, day 21:
- Planet 0: m=73479997433437751345152.000000, x=(14337840.000000,-359569632.000000,0.000000), v=(1008.366577,44.479641,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(4553956.000000,27181712.000000,0.000000), v=(-12.408961,12.089544,0.000000)
Planet states, day 22:
- Planet 0: m=73479997433437751345152.000000, x=(113657744.000000,-341902912.000000,0.000000), v=(966.711243,307.187439,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(3328717.000000,28240426.000000,0.000000), v=(-11.895715,8.866432,0.000000)
Planet states, day 24:
- Planet 0: m=73479997433437751345152.000000, x=(205408992.000000,-298752448.000000,0.000000), v=(857.260498,550.636414,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(2200139.250000,28962190.000000,0.000000), v=(-10.549925,5.870755,0.000000)
Planet states, day 25:
- Planet 0: m=73479997433437751345152.000000, x=(283140768.000000,-233010656.000000,0.000000), v=(687.326355,757.326172,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(1243588.750000,29426018.000000,0.000000), v=(-8.459596,3.327133,0.000000)
Planet states, day 26:
- Planet 0: m=73479997433437751345152.000000, x=(341253664.000000,-149026000.000000,0.000000), v=(468.435333,912.456665,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(527945.062500,29655688.000000,0.000000), v=(-5.767908,1.419369,0.000000)
Planet states, day 27:
- Planet 0: m=73479997433437751345152.000000, x=(375670016.000000,-52628844.000000,0.000000), v=(216.055954,1004.427917,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(103940.929688,29714662.000000,0.000000), v=(-2.661315,0.287587,0.000000)
Planet states, day 28:
- Planet 0: m=73479997433437751345152.000000, x=(383941152.000000,49526240.000000,0.000000), v=(-51.923882,1026.457275,0.000000)
- Planet 1: m=5972000231429685244854272.000000, x=(2024.937866,29714662.000000,0.000000), v=(0.635352,0.016110,0.000000)
```
