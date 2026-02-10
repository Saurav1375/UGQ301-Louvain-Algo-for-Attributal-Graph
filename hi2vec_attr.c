#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <time.h>
#include <math.h>

#include "attr.h"

#define HMAX 100000

static unsigned g_k = 0;
static unsigned long g_attr_d = 0;
static double *g_proj = NULL;

inline double rand1(){
  return 2 * ((rand()+1.0) / (RAND_MAX+1.0)) - 1;
}

static void init_projection(unsigned k, unsigned long d) {
  unsigned long i;
  g_k = k;
  g_attr_d = d;
  g_proj = malloc((unsigned long)k * d * sizeof(double));
  for (i = 0; i < (unsigned long)k * d; i++) {
    g_proj[i] = rand1() / sqrt((double)k);
  }
}

static inline double attr_proj_coord(const float *x, unsigned j) {
  unsigned long t;
  double s = 0.0;
  for (t = 0; t < g_attr_d; t++) {
    s += (double)x[t] * g_proj[(unsigned long)j * g_attr_d + t];
  }
  return s;
}

void recvec_attr(FILE* file1, FILE* file2, unsigned k, double a, double beta, double *vec){
  unsigned h;
  unsigned long i,j,u,n,c;
  double ah;

  if (fscanf(file1,"%u %lu", &h, &c)!=2){
    printf("file reading error 1\n");
    return;
  }
  ah=pow(a,h);
  if (vec==NULL){
    vec=malloc(HMAX*k*sizeof(double));
    bzero(vec,k*sizeof(double));
  }

  if (c==1){
    if (fscanf(file1,"%lu", &n)!=1){
      printf("file reading error 2\n");
      return;
    }
    for (i=0;i<n;i++){
      if (fscanf(file1,"%lu",&u)!=1){
        printf("file reading error 3\n");
        return;
      }
      const float *x = get_node_attributes(u);
      fprintf(file2,"%lu",u);
      for (j=0;j<k;j++){
        double attr_term = 0.0;
        if (x != NULL && g_attr_d > 0) {
          attr_term = beta * attr_proj_coord(x, j) * ah;
        }
        fprintf(file2," %le",vec[h*k+j] + rand1()*ah + attr_term);
      }
      fprintf(file2,"\n");
    }
  }
  else{
    for (i=0;i<c;i++){
      for (j=0;j<k;j++){
        vec[(h+1)*k+j]=vec[h*k+j]+rand1()*ah;
      }
      recvec_attr(file1, file2, k, a, beta, vec);
    }
  }
}

int main(int argc,char** argv){
  if (argc != 7) {
    printf("Usage: ./hi2vec_attr k a beta hierarchy.txt attributes.txt vectors.txt\n");
    return 1;
  }

  FILE *file1,*file2;
  unsigned k;
  double a, beta;

  time_t t1,t2;
  t1=time(NULL);

  srand(time(NULL));
  k=atoi(argv[1]);
  a=atof(argv[2]);
  beta=atof(argv[3]);

  if (!load_attributes(argv[5])) {
    printf("Could not load attribute file: %s\n", argv[5]);
    return 1;
  }

  init_projection(k, attr_dim());

  file1=fopen(argv[4],"r");
  file2=fopen(argv[6],"w");

  recvec_attr(file1, file2, k, a, beta, NULL);

  fclose(file1);
  fclose(file2);
  free(g_proj);
  free_attributes();

  t2=time(NULL);
  printf("- Overall Time = %ldh%ldm%lds\n",(t2-t1)/3600,((t2-t1)%3600)/60,((t2-t1)%60));
  return 0;
}
