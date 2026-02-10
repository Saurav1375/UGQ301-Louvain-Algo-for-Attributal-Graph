#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <time.h>
#include <math.h>

#define HMAX 100000

inline double rand1(){
  return 2 * ((rand()+1.0) / (RAND_MAX+1.0)) - 1;
}

void recvec(FILE* file1, FILE* file2, unsigned k, double a, double *vec){
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
      fprintf(file2,"%lu",u);
      for (j=0;j<k;j++){
        fprintf(file2," %le",vec[h*k+j]+rand1()*ah);
      }
      fprintf(file2,"\n");
    }
  }
  else{
    for (i=0;i<c;i++){
      for (j=0;j<k;j++){
        vec[(h+1)*k+j]=vec[h*k+j]+rand1()*ah;
      }
      recvec(file1, file2, k, a, vec);
    }
  }
}

int main(int argc,char** argv){
  if (argc != 5) {
    printf("Usage: ./hi2vec k a hierarchy.txt vectors.txt\n");
    return 1;
  }

  FILE *file1,*file2;
  unsigned k;
  double a;

  time_t t1,t2;
  t1=time(NULL);

  srand(time(NULL));
  printf("Number of dimensions: %s\n",argv[1]);
  k=atoi(argv[1]);
  printf("Damping factor: %s\n",argv[2]);
  a=atof(argv[2]);

  printf("Reading hierarchy from file: %s\n",argv[3]);
  file1=fopen(argv[3],"r");
  printf("Writing vectors to file: %s\n",argv[4]);
  file2=fopen(argv[4],"w");

  recvec(file1, file2, k, a, NULL);

  fclose(file1);
  fclose(file2);

  t2=time(NULL);
  printf("- Overall Time = %ldh%ldm%lds\n",(t2-t1)/3600,((t2-t1)%3600)/60,((t2-t1)%60));

  return 0;
}
