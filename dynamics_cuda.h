/**
 * CUDA kernels for physics dynamics simulations
 */

#ifndef H_DYNAMICS_CUDA_H
#define H_DYNAMICS_CUDA_H

#include <stdio.h>
#include "dynamics.h"
#include "planet.h"

__global__ void delta_v_kernel(planet *planets, int n, int max_interaction_idx, float dt);

__global__ void delta_x_kernel(planet *planets, int n, int max_planet_idx, float dt);
	
#endif
