#ifndef PARTITION_H
#define PARTITION_H

#define _GNU_SOURCE

#include "struct.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define K 5
#define MIN_IMPROVEMENT 0.005

typedef unsigned long (*partition)(adjlist*,unsigned long*);

partition choose_partition(char*);
unsigned long init(adjlist *g,unsigned long *lab);
unsigned long louvain(adjlist *g, unsigned long *lab);
unsigned long louvainComplete(adjlist *g, unsigned long *lab);
unsigned long louvainAttributed(adjlist *g, unsigned long *lab);

void set_attr_louvain_weight(long double lambda);

typedef struct {
  unsigned long size;
  unsigned long *node2Community;
  long double *in;
  long double *tot;

  long double *neighCommWeights;
  unsigned long *neighCommPos;
  unsigned long neighCommNb;

  unsigned long *commSize;
  long double *attrSums;
} louvainPartition;

void freeLouvainPartition(louvainPartition *p);
louvainPartition *createLouvainPartition(adjlist *g);
long double modularity(louvainPartition *p, adjlist *g);
void neighCommunities(louvainPartition *p, adjlist *g, unsigned long node);
adjlist* louvainPartition2Graph(louvainPartition *p, adjlist *g);
long double louvainOneLevel(louvainPartition *p, adjlist *g);
long double louvainOneLevelAttributed(louvainPartition *p, adjlist *g);

void shuff(unsigned long, unsigned long*);
unsigned long labprop(adjlist*,unsigned long*);

#endif
