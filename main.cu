/**
 * Gravitational simulation main loop
 */

#include <stdio.h>
#include "main.h"
#include <stdlib.h>

planet *planets;

void init_planets(planet *p, int n) {
/*
  p[0].m=7.348f*powf(10.0f,22.0f);
  p[0].x=384400000.0f;
  p[0].y=0.0f;
  p[0].z=0.0f;
  p[0].vx=0.0f;
  p[0].vy=0.0f;
  p[0].vz=0.0f;
  p[1].m=5.972f*powf(10.0f,24.0f);
  p[1].x=0.0f;
  p[1].y=0.0f;
  p[1].z=0.0f;
  p[1].vx=0.0f;
  p[1].vy=0.0f;
  p[1].vz=0.0f;
*/

  printf("Creating planets...");

/*

  for (int i=0; i<n; i++) {
   p[i].m=0.0f;
   p[i].x=0.0f;
   p[i].y=0.0f;
   p[i].z=0.0f;
   p[i].vx=0.0f;
   p[i].vy=0.0f;
   p[i].vz=0.0f;
  }

*/

  p[0].m=7.348f*powf(10.0f,22.0f);
  p[0].x=384400000.0f;
  p[0].y=0.0f;
  p[0].z=0.0f;
  p[0].vx=0.0f;
  p[0].vy=1027.777778f;
  p[0].vz=0.0f;
  p[1].m=5.972f*powf(10.0f,24.0f);
  p[1].x=0.0f;
  p[1].y=0.0f;
  p[1].z=0.0f;
  p[1].vx=0.0f;
  p[1].vy=0.0f;
  p[1].vz=0.0f;

  printf("Done creating planets.");
}

int get_max_interaction_idx(int n) {
  return n*(n-1)/2-1;
}

void print_planet_states(planet *p, int n, int iter) {
  printf("Planet states, day %d:\n",(int)ceil(((float)iter)*T/86400.0f));
  for (int i=0; i<n; i++) {
    printf("- Planet %d: m=%f, x=(%f,%f,%f), v=(%f,%f,%f)\n",i,p[i].m,p[i].x,p[i].y,p[i].z,p[i].vx,p[i].vy,p[i].vz);
  }
}

int main() {

  // Setup

  planet *devPlanets;

  planets = (planet*)malloc(N * sizeof(planet));
  init_planets(planets, N);

  cudaMalloc((void**)&devPlanets, N * sizeof(planet));
  
  // Iter

  int print_rate=10000;

  for (int i=0; i<NITER; i++) {

    if (i % print_rate == 0) print_planet_states(planets, N, i);

    cudaMemcpy(devPlanets, planets, N * sizeof(planet), cudaMemcpyHostToDevice);

    delta_v_kernel<<<512,512>>>(devPlanets, N, get_max_interaction_idx(N), T);
 
    delta_x_kernel<<<512,512>>>(devPlanets, N, N-1, T);

    cudaDeviceSynchronize();

    cudaMemcpy(planets, devPlanets, N * sizeof(planet), cudaMemcpyDeviceToHost);

  }

  // Print result

  print_planet_states(planets, N, N-1);

  // Cleanup

  cudaFree(devPlanets);
  free(planets);

  //cudaProfilerStop();
}
