#include <flann/flann.h>
#include <algorithm>
#include <vector>

#include "_flann_quantile.h"

using namespace flann;


void init_flann_parameters(FLANNParameters* p)
{
    if (p != NULL) {
        flann::log_verbosity(p->log_level);
        if (p->random_seed>0) {
            seed_random(p->random_seed);
        }
    }
}

flann::SearchParams create_search_params(FLANNParameters* p)
{
    flann::SearchParams params;
    params.checks = p->checks;
    params.eps = p->eps;
    params.sorted = p->sorted;
    params.max_neighbors = p->max_neighbors;
    params.cores = p->cores;

    return params;
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class ArgsortComparator {
    const T* data;
public:
    ArgsortComparator(const T* data) : data(data) { }
    bool operator()(int a, int b) { return data[a] < data[b]; }
};

template<typename Distance, typename WeightType>
int __flann_quantile_search(flann_index_t index_ptr,
                            const WeightType* weights,
                            typename Distance::ElementType* testset,
                            int tcount,
                            int* indices,
                            typename Distance::ResultType* dists,
                            const WeightType* weight_targets,
                            int weightcount,
                            const int* n_neighbors,
                            int neighborcount,
                            bool le_weight,
                            FLANNParameters* flann_params)
{
    // weights should have size() entries
    // testset should have tcount * veclen() entries, row-major
    // indices, dists should have (weightcount + n_neighbors) * tcount entries
    // weight_targets should be weightcount different alpha targets
    // n_neighbors should be neighborcount different ks (1-indexed)
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    if (neighborcount == 0 && weightcount == 0) {
        return 0;
    }

    try {
        init_flann_parameters(flann_params);
        if (index_ptr==NULL) {
            throw FLANNException("Invalid index");
        }
        Index<Distance>* index = (Index<Distance>*)index_ptr;

        // need weights in a vector
        size_t sz = index->size();
        std::vector<WeightType> weights_v(weights, weights + sz);

        // argsort the weight_targets
        std::vector<int> weight_idx(weightcount);
        for (size_t i = 0; i < weightcount; ++i) { weight_idx[i] = i; }
        std::sort(weight_idx.begin(), weight_idx.end(),
                  ArgsortComparator<WeightType>(weight_targets));

        // allocate results
        std::vector< std::vector<size_t> > indices_v;
        std::vector< std::vector<DistanceType> > dists_v;

        SearchParams search_params = create_search_params(flann_params);

        int min_neighbors = neighborcount == 0 ? 0 :
                *std::max_element(n_neighbors, n_neighbors + neighborcount);

        // do the actual search
        int count;
        if (weightcount > 0) {
            WeightType max_weight_target =
                    weight_targets[weight_idx[weightcount - 1]];
            count = index->quantileSearch(
                    Matrix<ElementType>(testset, tcount, index->veclen()),
                    weights_v, indices_v, dists_v,
                    max_weight_target, min_neighbors, false, search_params);
            // always want the one-past-weight result to simplify copying-out code
            // TODO: fix count if we swapped that?
        } else {
            index->knnSearch(
                    Matrix<ElementType>(testset, tcount, index->veclen()),
                    indices_v, dists_v, min_neighbors, search_params);
            count = min_neighbors * tcount;
        }

        // copy results out
        Matrix<int> m_indices(indices, tcount, weightcount + neighborcount);
        Matrix<DistanceType> m_dists(dists, tcount, weightcount + neighborcount);
        for (int i = 0; i < tcount; i++) {
            const std::vector<size_t> &res_indices = indices_v[i];
            const std::vector<DistanceType> &res_dists = dists_v[i];
            size_t res_sz = res_indices.size();

            // TODO: could shortcut this for max_weight_target....
            WeightType partsum = 0;
            int k = 0;
            for (int j = 0; j < res_sz && k < weightcount; ++j) {
                partsum += weights[res_indices[j]];
                while (k < weightcount && partsum >= weight_targets[weight_idx[k]]) {
                    if (le_weight && j == 0) {
                        m_indices[i][k] = -1; // TODO: what to do with this here?
                        m_dists[i][k] = std::numeric_limits<WeightType>::quiet_NaN();
                    } else {
                        m_indices[i][k] = res_indices[le_weight ? j - 1 : j];
                        m_dists[i][k] = res_dists[le_weight ? j - 1 : j];
                    }
                    ++k;
                }
            }

            for (int k = 0; k < neighborcount; ++k) {
                m_indices[i][weightcount + k] = res_indices[n_neighbors[k] - 1];
                m_dists[i][weightcount + k] = res_dists[n_neighbors[k] - 1];
            }
        }

        return count;
    }
    catch (std::runtime_error& e) {
        Logger::error("Caught exception: %s\n",e.what());
        return -1;
    }
}

template<typename T, typename R, typename WeightType>
int _flann_quantile_search(flann_index_t index_ptr,
                           const WeightType* weights,
                           T* testset,
                           int tcount,
                           int* indices,
                           R* dists,
                           const WeightType* weight_targets,
                           int weightcount,
                           const int* n_neighbors,
                           int neighborcount,
                           bool le_weight,
                           FLANNParameters* flann_params)
{
    enum flann_distance_t flann_distance_type = flann_get_distance_type();
    if (flann_distance_type==FLANN_DIST_EUCLIDEAN) {
        return __flann_quantile_search<L2<T> >(
            index_ptr, weights, testset, tcount, indices, dists,
            weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_MANHATTAN) {
        return __flann_quantile_search<L1<T> >(
            index_ptr, weights, testset, tcount, indices, dists,
            weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_MINKOWSKI) {
        return __flann_quantile_search<MinkowskiDistance<T> >(
            index_ptr, weights, testset, tcount, indices, dists,
            weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_HIST_INTERSECT) {
        return __flann_quantile_search<HistIntersectionDistance<T> >(
            index_ptr, weights, testset, tcount, indices, dists,
            weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_HELLINGER) {
        return __flann_quantile_search<HellingerDistance<T> >(
            index_ptr, weights, testset, tcount, indices, dists,
            weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_CHI_SQUARE) {
        return __flann_quantile_search<ChiSquareDistance<T> >(
            index_ptr, weights, testset, tcount, indices, dists,
            weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_KULLBACK_LEIBLER) {
        return __flann_quantile_search<KL_Divergence<T> >(
            index_ptr, weights, testset, tcount, indices, dists,
            weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
    }
    else {
        Logger::error( "Distance type unsupported in the C bindings, use the C++ bindings instead\n");
        return -1;
    }
}


int flann_quantile_search(flann_index_t index_ptr, const double* weights,
                           float* testset, int tcount, int* indices, float* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, FLANNParameters* flann_params) {
    return _flann_quantile_search(
        index_ptr, weights, testset, tcount, indices, dists,
        weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
}

int flann_quantile_search_float(flann_index_t index_ptr, const double* weights,
                           float* testset, int tcount, int* indices, float* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, FLANNParameters* flann_params) {
    return _flann_quantile_search(
        index_ptr, weights, testset, tcount, indices, dists,
        weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
}

int flann_quantile_search_double(flann_index_t index_ptr, const double* weights,
                           double* testset, int tcount, int* indices, double* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, FLANNParameters* flann_params) {
    return _flann_quantile_search(
        index_ptr, weights, testset, tcount, indices, dists,
        weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
}

int flann_quantile_search_byte(flann_index_t index_ptr, const double* weights,
                           unsigned char* testset, int tcount, int* indices, float* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, FLANNParameters* flann_params) {
    return _flann_quantile_search(
        index_ptr, weights, testset, tcount, indices, dists,
        weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
}

int flann_quantile_search_int(flann_index_t index_ptr, const double* weights,
                           int* testset, int tcount, int* indices, float* dists,
                           const double* weight_targets, int weightcount,
                           const int* n_neighbors, int neighborcount,
                           bool le_weight, FLANNParameters* flann_params) {
    return _flann_quantile_search(
        index_ptr, weights, testset, tcount, indices, dists,
        weight_targets, weightcount, n_neighbors, neighborcount, le_weight, flann_params);
}
