#ifndef _FLANN_QUANTILE_H_
#define _FLANN_QUANTILE_H_

#include <flann/flann.h>


#ifdef __cplusplus
extern "C"
{
    using namespace flann;
#else
typedef _Bool bool;
#endif

int flann_quantile_search(flann_index_t index_ptr, const double* weights,
                           float* testset, int tcount, int* indices, float* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, struct FLANNParameters* flann_params);
int flann_quantile_search_float(flann_index_t index_ptr, const double* weights,
                           float* testset, int tcount, int* indices, float* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, struct FLANNParameters* flann_params);
int flann_quantile_search_double(flann_index_t index_ptr, const double* weights,
                           double* testset, int tcount, int* indices, double* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, struct FLANNParameters* flann_params);
int flann_quantile_search_byte(flann_index_t index_ptr, const double* weights,
                           const unsigned char* testset, int tcount, int* indices, float* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, struct FLANNParameters* flann_params);
int flann_quantile_search_int(flann_index_t index_ptr, const double* weights,
                           int* testset, int tcount, int* indices, float* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, struct FLANNParameters* flann_params);

#ifdef __cplusplus
}
#endif

#endif
