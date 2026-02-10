#include "partition.h"
#include "attr.h"

#include <math.h>

#define NLINKS2 8

static long double g_attr_lambda = 0.2L;

void set_attr_louvain_weight(long double lambda) {
  if (lambda < 0.0L) {
    lambda = 0.0L;
  }
  g_attr_lambda = lambda;
}

unsigned long init(adjlist *g,unsigned long *lab){
  unsigned long i,n=g->n;
  unsigned long nlab=(K>n)?n:K;
  for (i=0;i<n;i++){
    lab[i]=rand()%nlab;
  }
  return nlab;
}

partition choose_partition(char *c){
  printf("Chosen partition algorithm: ");
  if (strcmp(c,"0")==0){
    printf("Random partition\n");
    return init;
  }
  if (strcmp(c,"1")==0){
    printf("Louvain partition\n");
    return louvainComplete;
  }
  if (strcmp(c,"2")==0){
    printf("Louvain first-level partition\n");
    return louvain;
  }
  if (strcmp(c,"3")==0){
    printf("Label propagation partition\n");
    return labprop;
  }
  if (strcmp(c,"4")==0){
    printf("Attributed Louvain partition\n");
    return louvainAttributed;
  }
  printf("unknown\n");
  exit(1);
}

int myCompare(const void *a, const void *b, void *array2) {
  unsigned long ia = *(const unsigned long *)a;
  unsigned long ib = *(const unsigned long *)b;
  long diff = ((unsigned long *)array2)[ia] - ((unsigned long *)array2)[ib];
  return (0 < diff) - (diff < 0);
}

unsigned long *mySort(unsigned long *part, unsigned long size) {
  unsigned long i;
  unsigned long *nodes = (unsigned long *)malloc(size * sizeof(unsigned long));
  for (i = 0; i < size; i++) {
    nodes[i] = i;
  }
  qsort_r(nodes, size, sizeof(unsigned long), myCompare, (void *)part);
  return nodes;
}

inline long double degreeWeighted(adjlist *g, unsigned long node) {
  unsigned long long i;
  if (g->weights == NULL) {
    return 1.0L * (g->cd[node + 1] - g->cd[node]);
  }
  long double res = 0.0L;
  for (i = g->cd[node]; i < g->cd[node + 1]; i++) {
    res += g->weights[i];
  }
  return res;
}

inline long double selfloopWeighted(adjlist *g, unsigned long node) {
  unsigned long long i;
  for (i = g->cd[node]; i < g->cd[node + 1]; i++) {
    if (g->adj[i] == node) {
      return (g->weights == NULL) ? 1.0L : g->weights[i];
    }
  }
  return 0.0L;
}

static inline void attr_remove_node(louvainPartition *p, adjlist *g, unsigned long node, unsigned long comm) {
  unsigned long d = attr_dim();
  if (d == 0 || p->attrSums == NULL) {
    return;
  }
  unsigned long oid = (g->map == NULL) ? node : g->map[node];
  const float *x = get_node_attributes(oid);
  if (x == NULL) {
    return;
  }
  unsigned long j;
  long double *dst = p->attrSums + comm * d;
  for (j = 0; j < d; j++) {
    dst[j] -= (long double)x[j];
  }
}

static inline void attr_insert_node(louvainPartition *p, adjlist *g, unsigned long node, unsigned long comm) {
  unsigned long d = attr_dim();
  if (d == 0 || p->attrSums == NULL) {
    return;
  }
  unsigned long oid = (g->map == NULL) ? node : g->map[node];
  const float *x = get_node_attributes(oid);
  if (x == NULL) {
    return;
  }
  unsigned long j;
  long double *dst = p->attrSums + comm * d;
  for (j = 0; j < d; j++) {
    dst[j] += (long double)x[j];
  }
}

static inline void removeNode(louvainPartition *p, adjlist *g, unsigned long node, unsigned long comm, long double dnodecomm) {
  p->in[comm]  -= 2.0L * dnodecomm + selfloopWeighted(g, node);
  p->tot[comm] -= degreeWeighted(g, node);
  if (p->commSize[comm] > 0) {
    p->commSize[comm]--;
  }
  attr_remove_node(p, g, node, comm);
}

static inline void insertNode(louvainPartition *p, adjlist *g, unsigned long node, unsigned long comm, long double dnodecomm) {
  p->in[comm]  += 2.0L * dnodecomm + selfloopWeighted(g, node);
  p->tot[comm] += degreeWeighted(g, node);
  p->node2Community[node] = comm;
  p->commSize[comm]++;
  attr_insert_node(p, g, node, comm);
}

inline long double gain(louvainPartition *p, adjlist *g, unsigned long comm, long double dnc, long double degc) {
  long double totc = p->tot[comm];
  long double m2 = g->totalWeight;
  return (dnc - totc * degc / m2);
}

static inline long double attr_gain(louvainPartition *p, adjlist *g, unsigned long node, unsigned long comm) {
  if (g_attr_lambda <= 0.0L || attr_dim() == 0 || p->attrSums == NULL || p->commSize[comm] == 0) {
    return 0.0L;
  }
  const long double *vec = p->attrSums + comm * attr_dim();
  long double cos = attr_cosine_node_to_comm(g, node, vec, p->commSize[comm]);
  return g_attr_lambda * cos;
}

void free_adjlist2(adjlist *g){
  free(g->cd);
  free(g->adj);
  free(g->weights);
  free(g);
}

void freeLouvainPartition(louvainPartition *p) {
  free(p->in);
  free(p->tot);
  free(p->neighCommWeights);
  free(p->neighCommPos);
  free(p->node2Community);
  free(p->commSize);
  free(p->attrSums);
  free(p);
}

louvainPartition *createLouvainPartition(adjlist *g) {
  unsigned long i;
  unsigned long d = attr_dim();

  louvainPartition *p = malloc(sizeof(louvainPartition));
  p->size = g->n;

  p->node2Community = malloc(p->size * sizeof(unsigned long));
  p->in = malloc(p->size * sizeof(long double));
  p->tot = malloc(p->size * sizeof(long double));

  p->neighCommWeights = malloc(p->size * sizeof(long double));
  p->neighCommPos = malloc(p->size * sizeof(unsigned long));
  p->neighCommNb = 0;

  p->commSize = calloc(p->size, sizeof(unsigned long));
  p->attrSums = (d == 0) ? NULL : calloc(p->size * d, sizeof(long double));

  for (i = 0; i < p->size; i++) {
    p->node2Community[i] = i;
    p->in[i] = selfloopWeighted(g, i);
    p->tot[i] = degreeWeighted(g, i);
    p->neighCommWeights[i] = -1;
    p->neighCommPos[i] = 0;
    p->commSize[i] = 1;
    attr_insert_node(p, g, i, i);
  }

  return p;
}

long double modularity(louvainPartition *p, adjlist *g) {
  long double q = 0.0L;
  long double m2 = g->totalWeight;
  unsigned long i;

  for (i = 0; i < p->size; i++) {
    if (p->tot[i] > 0.0L) {
      q += p->in[i] - (p->tot[i] * p->tot[i]) / m2;
    }
  }
  return q / m2;
}

void neighCommunitiesInit(louvainPartition *p) {
  unsigned long i;
  for (i = 0; i < p->neighCommNb; i++) {
    p->neighCommWeights[p->neighCommPos[i]] = -1;
  }
  p->neighCommNb = 0;
}

void neighCommunities(louvainPartition *p, adjlist *g, unsigned long node) {
  unsigned long long i;
  unsigned long neigh, neighComm;
  long double neighW;
  p->neighCommPos[0] = p->node2Community[node];
  p->neighCommWeights[p->neighCommPos[0]] = 0.0L;
  p->neighCommNb = 1;

  for (i = g->cd[node]; i < g->cd[node + 1]; i++) {
    neigh = g->adj[i];
    neighComm = p->node2Community[neigh];
    neighW = (g->weights == NULL) ? 1.0L : g->weights[i];
    if (neigh != node) {
      if (p->neighCommWeights[neighComm] == -1) {
        p->neighCommPos[p->neighCommNb] = neighComm;
        p->neighCommWeights[neighComm] = 0.0L;
        p->neighCommNb++;
      }
      p->neighCommWeights[neighComm] += neighW;
    }
  }
}

void neighCommunitiesAll(louvainPartition *p, adjlist *g, unsigned long node) {
  unsigned long long i;
  unsigned long neigh, neighComm;
  long double neighW;

  for (i = g->cd[node]; i < g->cd[node + 1]; i++) {
    neigh = g->adj[i];
    neighComm = p->node2Community[neigh];
    neighW = (g->weights == NULL) ? 1.0L : g->weights[i];

    if (p->neighCommWeights[neighComm] == -1) {
      p->neighCommPos[p->neighCommNb] = neighComm;
      p->neighCommWeights[neighComm] = 0.0L;
      p->neighCommNb++;
    }
    p->neighCommWeights[neighComm] += neighW;
  }
}

unsigned long updatePartition(louvainPartition *p, unsigned long *part, unsigned long size) {
  unsigned long *renumber = calloc(p->size, sizeof(unsigned long));
  unsigned long i, last = 1;
  for (i = 0; i < p->size; i++) {
    if (renumber[p->node2Community[i]] == 0) {
      renumber[p->node2Community[i]] = last++;
    }
  }

  for (i = 0; i < size; i++) {
    part[i] = renumber[p->node2Community[part[i]]] - 1;
  }

  free(renumber);
  return last - 1;
}

adjlist* louvainPartition2Graph(louvainPartition *p, adjlist *g) {
  unsigned long node, i, j;
  unsigned long *renumber = malloc(g->n * sizeof(unsigned long));
  for (node = 0; node < g->n; node++) {
    renumber[node] = 0;
  }
  unsigned long last = 1;
  for (node = 0; node < g->n; node++) {
    if (renumber[p->node2Community[node]] == 0) {
      renumber[p->node2Community[node]] = last++;
    }
  }
  for (node = 0; node < g->n; node++) {
    p->node2Community[node] = renumber[p->node2Community[node]] - 1;
  }

  unsigned long *order = mySort(p->node2Community, g->n);
  adjlist *res = malloc(sizeof(adjlist));
  unsigned long long e1 = NLINKS2;
  res->n = last - 1;
  res->e = 0;
  res->cd = calloc((1 + res->n), sizeof(unsigned long long));
  res->cd[0] = 0;
  res->adj = malloc(NLINKS2 * sizeof(unsigned long));
  res->totalWeight = 0.0L;
  res->weights = malloc(NLINKS2 * sizeof(long double));
  res->map = NULL;

  neighCommunitiesInit(p);
  unsigned long oldComm = p->node2Community[order[0]];
  unsigned long currentComm = oldComm;

  for (i = 0; i <= p->size; i++) {
    node = (i == p->size) ? 0 : order[i];
    currentComm = (i == p->size) ? (currentComm + 1) : p->node2Community[order[i]];

    if (oldComm != currentComm) {
      res->cd[oldComm + 1] = res->cd[oldComm] + p->neighCommNb;

      for (j = 0; j < p->neighCommNb; j++) {
        unsigned long neighComm = p->neighCommPos[j];
        long double neighCommWeight = p->neighCommWeights[p->neighCommPos[j]];

        res->adj[res->e] = neighComm;
        res->weights[res->e] = neighCommWeight;
        res->totalWeight += neighCommWeight;
        (res->e)++;

        if (res->e == e1) {
          e1 *= 2;
          res->adj = realloc(res->adj, e1 * sizeof(unsigned long));
          res->weights = realloc(res->weights, e1 * sizeof(long double));
          if (res->adj == NULL || res->weights == NULL) {
            printf("error during memory allocation\n");
            exit(1);
          }
        }
      }

      if (i == p->size) {
        res->adj = realloc(res->adj, res->e * sizeof(unsigned long));
        res->weights = realloc(res->weights, res->e * sizeof(long double));
        free(order);
        free(renumber);
        return res;
      }

      oldComm = currentComm;
      neighCommunitiesInit(p);
    }

    neighCommunitiesAll(p, g, node);
  }

  printf("bad exit\n");
  return res;
}

long double louvainOneLevel(louvainPartition *p, adjlist *g) {
  unsigned long nbMoves;
  long double startModularity = modularity(p, g);
  long double newModularity = startModularity;
  long double curModularity;
  unsigned long i,j,node;
  unsigned long oldComm,newComm,bestComm;
  long double degreeW, bestCommW, bestGain, newGain;

  do {
    curModularity = newModularity;
    nbMoves = 0;

    for (i = 0; i < g->n; i++) {
      node = i;
      oldComm = p->node2Community[node];
      degreeW = degreeWeighted(g, node);

      neighCommunitiesInit(p);
      neighCommunities(p, g, node);

      removeNode(p, g, node, oldComm, p->neighCommWeights[oldComm]);

      bestComm = oldComm;
      bestCommW = 0.0L;
      bestGain = 0.0L;
      for (j = 0; j < p->neighCommNb; j++) {
        newComm = p->neighCommPos[j];
        newGain = gain(p, g, newComm, p->neighCommWeights[newComm], degreeW);
        if (newGain > bestGain) {
          bestComm = newComm;
          bestCommW = p->neighCommWeights[newComm];
          bestGain = newGain;
        }
      }

      insertNode(p, g, node, bestComm, bestCommW);
      if (bestComm != oldComm) {
        nbMoves++;
      }
    }

    newModularity = modularity(p, g);
  } while (nbMoves > 0 && newModularity - curModularity > MIN_IMPROVEMENT);

  return newModularity - startModularity;
}

long double louvainOneLevelAttributed(louvainPartition *p, adjlist *g) {
  unsigned long nbMoves;
  unsigned long i, j, node;
  unsigned long oldComm, newComm, bestComm;
  long double degreeW, bestCommW, bestGain, newGain;

  do {
    nbMoves = 0;

    for (i = 0; i < g->n; i++) {
      node = i;
      oldComm = p->node2Community[node];
      degreeW = degreeWeighted(g, node);

      neighCommunitiesInit(p);
      neighCommunities(p, g, node);

      removeNode(p, g, node, oldComm, p->neighCommWeights[oldComm]);

      bestComm = oldComm;
      bestCommW = 0.0L;
      bestGain = attr_gain(p, g, node, oldComm);

      for (j = 0; j < p->neighCommNb; j++) {
        newComm = p->neighCommPos[j];
        newGain = gain(p, g, newComm, p->neighCommWeights[newComm], degreeW);
        newGain += attr_gain(p, g, node, newComm);
        if (newGain > bestGain) {
          bestComm = newComm;
          bestCommW = p->neighCommWeights[newComm];
          bestGain = newGain;
        }
      }

      insertNode(p, g, node, bestComm, bestCommW);
      if (bestComm != oldComm) {
        nbMoves++;
      }
    }
  } while (nbMoves > 0);

  return 0.0L;
}

unsigned long louvain(adjlist *g, unsigned long *lab) {
  unsigned long i,n;
  for (i = 0; i < g->n; i++) {
    lab[i] = i;
  }

  louvainPartition *gp = createLouvainPartition(g);
  louvainOneLevel(gp, g);
  n = updatePartition(gp, lab, g->n);
  freeLouvainPartition(gp);

  return n;
}

unsigned long louvainComplete(adjlist *g, unsigned long *lab) {
  adjlist *g2;
  unsigned long n, i;
  unsigned long originalSize = g->n;
  long double improvement;
  for (i = 0; i < g->n; i++) {
    lab[i] = i;
  }

  while (1) {
    louvainPartition *gp = createLouvainPartition(g);
    improvement = louvainOneLevel(gp, g);
    n = updatePartition(gp, lab, originalSize);

    if (improvement < MIN_IMPROVEMENT) {
      freeLouvainPartition(gp);
      break;
    }

    g2 = louvainPartition2Graph(gp, g);
    if (g->n < originalSize) {
      free_adjlist2(g);
    }
    freeLouvainPartition(gp);
    g = g2;
  }

  if (g->n < originalSize) {
    free_adjlist2(g);
  }

  return n;
}

unsigned long louvainAttributed(adjlist *g, unsigned long *lab) {
  unsigned long i, n;
  for (i = 0; i < g->n; i++) {
    lab[i] = i;
  }
  louvainPartition *gp = createLouvainPartition(g);
  louvainOneLevelAttributed(gp, g);
  n = updatePartition(gp, lab, g->n);
  freeLouvainPartition(gp);
  return n;
}

void shuff(unsigned long n, unsigned long *tab){
  unsigned long i,j,tmp;
  for (i=n-1;i>0;i--){
    j=rand()%i;
    tmp=tab[i];
    tab[i]=tab[j];
    tab[j]=tmp;
  }
}

unsigned long labprop(adjlist *g,unsigned long *lab) {
  unsigned long n=g->n,i,k,u,nl,l,lmax,nmax,nlab;
  unsigned long long j;
  bool b;
  static unsigned long *tab=NULL,*list=NULL,*nodes=NULL,*new=NULL;

  if (tab==NULL){
    tab=calloc(n,sizeof(unsigned long));
    list=malloc(n*sizeof(unsigned long));
    nodes=malloc(n*sizeof(unsigned long));
    new=malloc(n*sizeof(unsigned long));
  }

  for (i=0;i<n;i++) {
    lab[i]=i;
    nodes[i]=i;
    new[i]=(unsigned long)-1;
  }

  do {
    b=0;
    shuff(n,nodes);
    for (i=0;i<n;i++) {
      u=nodes[i];
      nl=0;
      for (j=g->cd[u];j<g->cd[u+1];j++) {
        l=lab[g->adj[j]];
        if (tab[l]++==0){
          list[nl++]=l;
        }
      }
      lmax=lab[u];
      nmax=tab[lmax];
      if (nl>0){
        shuff(nl,list);
      }
      for (k=0;k<nl;k++){
        l=list[k];
        if (tab[l]>nmax){
          lmax=l;
          nmax=tab[l];
          b=1;
        }
        tab[l]=0;
      }
      lab[u]=lmax;
    }
  } while(b);

  nlab=0;
  for (i=0;i<n;i++) {
    l=lab[i];
    if (new[l]==(unsigned long)-1){
      new[l]=nlab++;
    }
    lab[i]=new[l];
  }

  return nlab;
}
