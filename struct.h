#ifndef STRUCT_H
#define STRUCT_H

typedef struct {
  unsigned long s;
  unsigned long t;
} edge;

typedef struct {
  unsigned long n;
  unsigned long long e;
  unsigned long long emax;
  edge *edges;
  unsigned long *map;
} edgelist;

typedef struct {
  unsigned long n;
  unsigned long long e;
  unsigned long long emax;
  edge *edges;
  unsigned long long *cd;
  unsigned long *adj;
  long double *weights;
  long double totalWeight;
  unsigned long *map;
} adjlist;

typedef struct {
  unsigned long n;
  adjlist **sg;
} clusters;

#endif
