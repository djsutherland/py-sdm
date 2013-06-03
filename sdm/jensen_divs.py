################################################################################
### Jensen-Renyi divergences
# TODO: make them work for arbitrary sample sizes, integrate into estimate_divs?

def renyi_entropy_knns(knns, dim, alphas, Ks):
    '''
    Estimates Renyi entropy using a special case of our estimator above
    (with alpha=alpha-1, beta=0).

    knns: an (n_samps x max(Ks)) matrix of nearest-neighbor distances
    dim: the dimensionality of the underlying data
    alphas: a sequence of alpha values to use
    Ks: a sequence of K values to use

    Returns a matrix of entropy estimates: rows correspond to alphas, columns
    to Ks.

    See also: renyi_entropy, to run on samples directly.
    '''
    alphas = np.asarray(alphas).reshape(-1, 1)
    Ks = np.asarray(Ks).reshape(1, -1)

    N, num_ks = knns.shape
    if num_ks < np.max(Ks):
        raise ValueError("renyi_entropy_knns: knns should be square")

    # multiplicative constant:
    #   c_d^{1-alpha} * gamma(K) / gamma(K - alpha + 1)
    #   where c_d is volume of a d-dimensional ball, pi^{d/2} / gamma(d/2 + 1)
    # so exp(
    #         (1-alpha) * (d/2 * log(pi) - gammaln(d/2 + 1))
    #         + gammaln(K) - gammaln(K - alpha + 1)
    #    )
    Bs = np.exp(
        (1-alphas) * (dim/2 * np.log(np.pi) - gammaln(dim/2 + 1))
        + gammaln(Ks)
        - gammaln(Ks - alphas + 1)
    ) / (N - 1)**(alphas - 1)

    ms = np.empty((alphas.size, Ks.size))
    ms.fill(np.nan)
    for K_i, K in enumerate(Ks.flat):
        rho_k = knns[:, K-1]
        for alph_i, alpha in enumerate(alphas):
            ms[alph_i, K_i] = np.mean(rho_k ** (dim * (1 - alpha)))

    return np.log(Bs * ms) / (1 - alphas)


def renyi_entropy(samps, alphas, Ks, **kwargs):
    '''
    Estimates Renyi entropy using a special case of our estimator above
    (with alpha=alpha-1, beta=0).

    samps: an (n_samps x dimensionality) matrix of samples
    alphas: a sequence of alpha values to use
    Ks: a sequence of K values to use
    other kwargs: passed along to knn_search

    Returns a matrix of entropy estimates: one row per alpha, col per K
    '''
    # throw away the smallest one, which is dist to self
    knns, _ = knn_search(np.max(Ks) + 1, samps, samps, **kwargs)
    return renyi_entropy_knns(knns=knns[:, 1:], dim=samps.shape[1],
                              Ks=Ks, alphas=alphas)


def jensen_renyi(xx, xy, yy, yx, alphas, Ks, dim, clamp=True, **opts):
    r'''
    Estimate the Jensen-Renyi divergence between distributions,
        R_\alpha((p + q)/2) - 1/2 R_\alpha(p) - 1/2 R_\alpha(q)
    where R_\alpha is the Renyi entropy:
        1/(1 - \alpha) \log \int p^\alpha
    which is estimated based on kNN distances.

    Returns a matrix: each row corresponds to an alpha, each column to a K.
    '''
    if xy is None:  # identical bags
        return np.zeros((len(alphas), len(Ks)))

    max_K = xx.shape[1]
    N = xx.shape[0]
    M = yy.shape[0]
    alphas = np.asarray(alphas)
    entropy = functools.partial(renyi_entropy_knns,
                                dim=dim, alphas=alphas, Ks=Ks)

    # assuming N == M here
    #
    # when N != M, one option is to take min(N, M) from each
    # TODO: need to either get all NNs from x to y to combine,
    #       or redo the NN search within the mixture for this
    #
    # it'd also be possible to do some kind of negative binomial
    # type combination, to form all(?) samples we can from these samples
    # and maybe take an expectation over it or something
    #
    # TODO: can we do something with just these kNN distances?
    if N != M:
        raise ValueError("can't do jensen-renyi for nonequal sample sizes yet")
    mixture_toself = np.empty((N + M, 2 * max_K))
    mixture_toself[:N, :max_K] = xx
    mixture_toself[:N, max_K:] = xy
    mixture_toself[N:, :max_K] = yy
    mixture_toself[N:, max_K:] = yx
    mixture_toself.sort(axis=1)
    mix_entropies = entropy(mixture_toself)

    x_entropies = entropy(xx)
    y_entropies = entropy(yy)

    ests = mix_entropies - x_entropies / 2 - y_entropies / 2
    return np.maximum(ests, 0) if clamp else ests
jensen_renyi.is_symmetric = True
