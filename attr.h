#ifndef ATTR_H
#define ATTR_H

#include "struct.h"

int load_attributes(const char *path);
void free_attributes(void);
unsigned long attr_dim(void);
const float *get_node_attributes(unsigned long original_node_id);
long double attr_cosine_node_to_comm(adjlist *g, unsigned long node, const long double *comm_vec, unsigned long comm_size);
long double attr_dot_node_to_comm_sum(adjlist *g, unsigned long node, const long double *comm_vec);

#endif
