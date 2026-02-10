#include "attr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *g_attr = NULL;
static unsigned char *g_present = NULL;
static unsigned long g_max_id = 0;
static unsigned long g_dim = 0;

static int parse_line_dim(const char *line, unsigned long *id, unsigned long *dim_out) {
  char *tmp = strdup(line);
  if (tmp == NULL) {
    return 0;
  }
  char *save = NULL;
  char *tok = strtok_r(tmp, " \t\n\r", &save);
  if (tok == NULL) {
    free(tmp);
    return 0;
  }
  *id = strtoul(tok, NULL, 10);

  unsigned long d = 0;
  while ((tok = strtok_r(NULL, " \t\n\r", &save)) != NULL) {
    d++;
  }
  *dim_out = d;
  free(tmp);
  return (d > 0);
}

int load_attributes(const char *path) {
  FILE *f = fopen(path, "r");
  if (f == NULL) {
    return 0;
  }

  char *line = NULL;
  size_t cap = 0;
  ssize_t nread;

  unsigned long max_id = 0;
  unsigned long dim = 0;
  int have_dim = 0;

  while ((nread = getline(&line, &cap, f)) != -1) {
    if (nread <= 1) {
      continue;
    }
    unsigned long id = 0, d = 0;
    if (!parse_line_dim(line, &id, &d)) {
      continue;
    }
    if (!have_dim) {
      dim = d;
      have_dim = 1;
    } else if (d != dim) {
      fprintf(stderr, "inconsistent attribute dimensions in %s\n", path);
      free(line);
      fclose(f);
      return 0;
    }
    if (id > max_id) {
      max_id = id;
    }
  }

  if (!have_dim) {
    free(line);
    fclose(f);
    return 0;
  }

  rewind(f);

  g_dim = dim;
  g_max_id = max_id;
  g_attr = calloc((g_max_id + 1) * g_dim, sizeof(float));
  g_present = calloc(g_max_id + 1, sizeof(unsigned char));
  if (g_attr == NULL || g_present == NULL) {
    fprintf(stderr, "attribute allocation error\n");
    free(line);
    fclose(f);
    return 0;
  }

  while ((nread = getline(&line, &cap, f)) != -1) {
    if (nread <= 1) {
      continue;
    }

    char *save = NULL;
    char *tok = strtok_r(line, " \t\n\r", &save);
    if (tok == NULL) {
      continue;
    }
    unsigned long id = strtoul(tok, NULL, 10);
    if (id > g_max_id) {
      continue;
    }

    unsigned long d = 0;
    while ((tok = strtok_r(NULL, " \t\n\r", &save)) != NULL && d < g_dim) {
      g_attr[id * g_dim + d] = strtof(tok, NULL);
      d++;
    }
    if (d == g_dim) {
      g_present[id] = 1;
    }
  }

  free(line);
  fclose(f);
  return 1;
}

void free_attributes(void) {
  free(g_attr);
  free(g_present);
  g_attr = NULL;
  g_present = NULL;
  g_max_id = 0;
  g_dim = 0;
}

unsigned long attr_dim(void) {
  return g_dim;
}

const float *get_node_attributes(unsigned long original_node_id) {
  if (g_dim == 0 || original_node_id > g_max_id || g_present[original_node_id] == 0) {
    return NULL;
  }
  return g_attr + original_node_id * g_dim;
}

long double attr_cosine_node_to_comm(adjlist *g, unsigned long node, const long double *comm_vec, unsigned long comm_size) {
  if (g_dim == 0 || comm_size == 0) {
    return 0.0L;
  }
  unsigned long oid = (g->map == NULL) ? node : g->map[node];
  const float *x = get_node_attributes(oid);
  if (x == NULL) {
    return 0.0L;
  }

  long double dot = 0.0L;
  long double nx = 0.0L;
  long double nc = 0.0L;
  unsigned long j;
  for (j = 0; j < g_dim; j++) {
    long double cj = comm_vec[j] / (long double)comm_size;
    long double xj = (long double)x[j];
    dot += xj * cj;
    nx += xj * xj;
    nc += cj * cj;
  }

  if (nx <= 0.0L || nc <= 0.0L) {
    return 0.0L;
  }
  return dot / (sqrtl(nx) * sqrtl(nc));
}

long double attr_dot_node_to_comm_sum(adjlist *g, unsigned long node, const long double *comm_vec) {
  if (g_dim == 0) {
    return 0.0L;
  }
  unsigned long oid = (g->map == NULL) ? node : g->map[node];
  const float *x = get_node_attributes(oid);
  if (x == NULL) {
    return 0.0L;
  }

  long double dot = 0.0L;
  unsigned long j;
  for (j = 0; j < g_dim; j++) {
    dot += (long double)x[j] * comm_vec[j];
  }
  return dot;
}
