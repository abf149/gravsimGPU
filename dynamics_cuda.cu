#include "dynamics_cuda.h"

/** Model acceleration & gravitation */
__global__ void delta_v_kernel(planet *planets, int n, int max_interaction_idx, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > max_interaction_idx) return;

  int pL = (int)ceil(((float)i)/((float)(n-1)));
  int pR = i - (n-1)*pL;

  if (pL == pR) pR++;

  planet *L = &planets[pL];
  planet *R = &planets[pR];

  float GmRdt = -G * R->m * dt, GmLdt = G * L->m * dt;
  float dLmRx = L->x - R->x, dLmRy = L->y - R->y, dLmRz = L->z - R->z;

  if (dLmRx == 0.0 && dLmRy == 0.0 && dLmRz == 0.0) {
    dLmRx = 1;
    dLmRy = 1;
    dLmRz = 1;
  }

  float dist_denom = powf(dLmRx*dLmRx + dLmRy*dLmRy + dLmRz*dLmRz,1.5f);
  
  //printf("GmRdt: %f GmLdt: %f dLmRx: %f dLmRy: %f dLmRz: %f dist_denom: %f\n",GmRdt,GmLdt,dLmRx,dLmRy,dLmRz,dist_denom);

  L->vx += GmRdt * dLmRx / dist_denom;
  R->vx += GmLdt * dLmRx / dist_denom;

  L->vy += GmRdt * dLmRy / dist_denom;
  R->vy += GmLdt * dLmRy / dist_denom;

  L->vz += GmRdt * dLmRz / dist_denom;
  R->vz += GmLdt * dLmRz / dist_denom;
}

/** Model velocity */
__global__ void delta_x_kernel(planet *planets, int n, int max_planet_idx, float dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > max_planet_idx) return;

  planet *p = &planets[i];

  p->x += p->vx * dt;
  p->y += p->vy * dt;
  p->z += p->vz * dt;
}
