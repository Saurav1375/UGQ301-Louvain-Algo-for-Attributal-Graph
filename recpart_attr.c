#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <strings.h>
#include <time.h>

#include "partition.h"
#include "struct.h"
#include "attr.h"

#define NNODES 10000000
#define HMAX 100

inline unsigned long max3(unsigned long a,unsigned long b,unsigned long c){
  a = (a > b) ? a : b;
  return (a > c) ? a : c;
}

adjlist* readadjlist(char* input){
  unsigned long n1=NNODES,n2,u,v,i;
  unsigned long *d=calloc(n1,sizeof(unsigned long));
  adjlist *g=malloc(sizeof(adjlist));
  FILE *file;

  g->n=0;
  g->e=0;
  file=fopen(input,"r");
  while (fscanf(file,"%lu %lu", &u, &v)==2) {
    g->e++;
    g->n=max3(g->n,u,v);
    if (g->n+1>=n1) {
      n2=g->n+NNODES;
      d=realloc(d,n2*sizeof(unsigned long));
      bzero(d+n1,(n2-n1)*sizeof(unsigned long));
      n1=n2;
    }
    d[u]++;
    d[v]++;
  }
  fclose(file);

  g->n++;
  d=realloc(d,g->n*sizeof(unsigned long));

  g->cd=malloc((g->n+1)*sizeof(unsigned long long));
  g->cd[0]=0;
  for (i=1;i<g->n+1;i++) {
    g->cd[i]=g->cd[i-1]+d[i-1];
    d[i-1]=0;
  }

  g->adj=malloc(2*g->e*sizeof(unsigned long));

  file=fopen(input,"r");
  while (fscanf(file,"%lu %lu", &u, &v)==2) {
    g->adj[ g->cd[u] + d[u]++ ]=v;
    g->adj[ g->cd[v] + d[v]++ ]=u;
  }
  fclose(file);

  g->weights = NULL;
  g->totalWeight = 2*g->e;
  g->map=NULL;

  free(d);

  return g;
}

void free_adjlist(adjlist *g){
  free(g->cd);
  free(g->adj);
  free(g->weights);
  free(g->map);
  free(g);
}

adjlist* mkchild(adjlist* g, unsigned long* lab, unsigned long nlab, unsigned h, unsigned long clab){
  unsigned long i,u,v,lu;
  unsigned long long j,k,tmp;

  static unsigned hmax=0;
  static unsigned long **nodes;
  static unsigned long **new;
  static unsigned long long **cd;
  static unsigned long long **e;
  static unsigned long *d;
  adjlist* sg;

  if (hmax==0){
    hmax=HMAX;
    nodes=malloc(HMAX*sizeof(unsigned long *));
    new=malloc(HMAX*sizeof(unsigned long *));
    cd=malloc(HMAX*sizeof(unsigned long long *));
    e=malloc(HMAX*sizeof(unsigned long long *));
  }
  if (h==hmax){
    hmax+=HMAX;
    nodes=realloc(nodes,hmax*sizeof(unsigned long *));
    new=realloc(new,hmax*sizeof(unsigned long *));
    cd=realloc(cd,hmax*sizeof(unsigned long long *));
    e=realloc(e,hmax*sizeof(unsigned long long *));
  }

  if (clab==0){
    d=calloc(nlab,sizeof(unsigned long));
    cd[h]=malloc((nlab+1)*sizeof(unsigned long long));
    e[h]=calloc(nlab,sizeof(unsigned long long));
    for (i=0;i<g->n;i++){
      d[lab[i]]++;
    }
    cd[h][0]=0;
    for (i=0;i<nlab;i++){
      cd[h][i+1]=cd[h][i]+d[i];
      d[i]=0;
    }
    nodes[h]=malloc(g->n*sizeof(unsigned long));
    new[h]=malloc(g->n*sizeof(unsigned long));
    for (u=0;u<g->n;u++){
      lu=lab[u];
      nodes[h][cd[h][lu]+d[lu]]=u;
      new[h][u]=d[lu]++;
      for (j=g->cd[u];j<g->cd[u+1];j++){
        v=g->adj[j];
        if (lu==lab[v]){
          e[h][lu]++;
        }
      }
    }
    free(d);
  }

  sg=malloc(sizeof(adjlist));
  sg->n=cd[h][clab+1]-cd[h][clab];
  sg->e=e[h][clab]/2;
  sg->cd=malloc((sg->n+1)*sizeof(unsigned long long));
  sg->cd[0]=0;
  sg->adj=malloc(2*sg->e*sizeof(unsigned long));
  sg->map=malloc(sg->n*sizeof(unsigned long));
  sg->weights = NULL;
  sg->totalWeight = 2*sg->e;
  tmp=0;
  for (k=cd[h][clab];k<cd[h][clab+1];k++){
    u=nodes[h][k];
    sg->map[new[h][u]]=(g->map==NULL)?u:g->map[u];
    for (j=g->cd[u];j<g->cd[u+1];j++){
      v=g->adj[j];
      if (clab==lab[v]){
        sg->adj[tmp++]=new[h][v];
      }
    }
    sg->cd[new[h][u]+1]=tmp;
  }

  if (clab==nlab-1){
    free(nodes[h]);
    free(new[h]);
    free(cd[h]);
    free(e[h]);
  }

  return sg;
}

void recurs(partition part, adjlist* g, unsigned h, FILE* file){
  unsigned long nlab;
  unsigned long i;
  adjlist* sg;
  unsigned long *lab;

  if (g->e==0){
    fprintf(file,"%u 1 %lu",h,g->n);
    for (i=0;i<g->n;i++){
      fprintf(file," %lu",g->map[i]);
    }
    fprintf(file,"\n");
    free_adjlist(g);
  }
  else{
    lab=malloc(g->n*sizeof(unsigned long));
    nlab=part(g,lab);
    if (nlab==1){
      fprintf(file,"%u 1 %lu",h,g->n);
      for (i=0;i<g->n;i++){
        fprintf(file," %lu",g->map[i]);
      }
      fprintf(file,"\n");
    }
    else{
      fprintf(file,"%u %lu\n",h,nlab);
      for (i=0;i<nlab;i++){
        sg=mkchild(g,lab,nlab,h,i);
        recurs(part,sg,h+1,file);
      }
    }
    free_adjlist(g);
    free(lab);
  }
}

int main(int argc,char** argv){
  if (argc < 4 || argc > 6) {
    printf("Usage: ./recpart_attr edgelist.txt hierarchy.txt attributes.txt [lambda=0.2] [partition=4]\n");
    return 1;
  }

  adjlist* g;
  partition part;
  time_t t0=time(NULL),t1,t2;
  long double lambda = (argc >= 5) ? strtold(argv[4], NULL) : 0.2L;

  srand(time(NULL));

  if (!load_attributes(argv[3])) {
    printf("Could not load attribute file: %s\n", argv[3]);
    return 1;
  }
  set_attr_louvain_weight(lambda);

  if (argc == 6) {
    part=choose_partition(argv[5]);
  } else {
    part=choose_partition("4");
  }

  printf("Reading edgelist from file %s and building adjacency array\n",argv[1]);
  g=readadjlist(argv[1]);
  printf("Number of nodes: %lu\n",g->n);
  printf("Number of edges: %llu\n",g->e);
  printf("Attribute dim: %lu\n", attr_dim());
  printf("Attribute-community lambda: %.4Lf\n", lambda);

  t1=time(NULL);
  printf("- Time to load graph+attrs = %ldh%ldm%lds\n",(t1-t0)/3600,((t1-t0)%3600)/60,((t1-t0)%60));

  FILE* file=fopen(argv[2],"w");
  recurs(part, g, 0, file);
  fclose(file);

  t2=time(NULL);
  printf("- Time to compute hierarchy = %ldh%ldm%lds\n",(t2-t1)/3600,((t2-t1)%3600)/60,((t2-t1)%60));
  printf("- Overall time = %ldh%ldm%lds\n",(t2-t0)/3600,((t2-t0)%3600)/60,((t2-t0)%60));

  free_attributes();
  return 0;
}
