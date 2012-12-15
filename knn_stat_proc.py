# This script computes various divergences and kernels based on knn statistics.
# Ref: Barnabas Poczos, Liang Xiong, Jeff Schneider, Nonparametric divergence estimation with applications to machine learning on distributions 
# 
# Liang Xiong, lxiong@cs.cmu.edu

import sys, os, time
assert os.name == 'posix', 'the os should support fork()'

import numpy as np
from scipy.special import gamma, gammaln
import scipy.io as sio
import h5py as h5
import itertools, multiprocessing as mp

eps = np.spacing(1)

searcher = None
try:
    from pyflann import FLANN
    searcher = FLANN()
except:
    print '!! Cannot find FLANN. KNN searches will be much slower.'

def tic(): return time.time()
def toc(start): return time.time() - start
def col(a): return a.reshape((a.size, 1))
def row(a): return a.reshape((1, a.size))
def get_col(X, c): return X[:,c].ravel()
def now(): return time.strftime('%m/%d %H:%M:%S')

def FixTerms(terms, tail = 0.01):
    '''for a robust mean of terms
    '''

    terms = terms[np.logical_not(np.isnan(terms))]
    n = terms.size
    
    if n >= 3:
        terms = np.sort(terms)
        terms = terms[max(1, round(n*tail)) : min(n-1,round(n*(1-tail)))]

    return terms[np.isfinite(terms)]

def NPL2(xx, xy, yy, yx, opt):
    '''non-parametric l2 divergence estimator.
    returns a vector. one elem for one K in opt.Ks
    '''

    Ks, dim = opt['Ks'], opt['dim']

    if xy is None: # identical bags
        return np.zeros(len(Ks))

    N, M = float(xx.shape[0]), float(yy.shape[0])
    c = np.pi**(dim*0.5) / gamma(dim*0.5 + 1)

    rs = []
    for K in Ks:
        rho_x, nu_x = get_col(xx, K-1), get_col(xy, K-1)
        rho_y, nu_y = get_col(yy, K-1), get_col(yx, K-1)

        t1x = ((K-1)/((N-1)*c))/(rho_x**dim)
        t1y = ((K-1)/((M-1)*c))/(rho_y**dim)
        t3x = ((K-1)/(M*c))/(nu_x**dim)
        t3y = ((K-1)/(N*c))/(nu_y**dim)

        total = FixTerms(t1x,0.05).mean() + FixTerms(t1y,0.05).mean() - FixTerms(t3x,0.05).mean() - FixTerms(t3y,0.05).mean()
        rs.append(np.sqrt(max(0, total)))

    return np.array(rs)

def NPAlpha(xx, xy, yy, yx, alphas, opt):
    '''nonparameric estimatr for alpha integral. used for renyi and hellinger divergence estimation.
    returns a matrix. each row for one alpha in alphas. each col for one K in opt.Ks
    '''

    Ks, dim = opt['Ks'], opt['dim']
    alphas = np.array(alphas)

    N, M = float(xx.shape[0]), float(yy.shape[0])
    rs = np.empty((len(alphas), len(Ks)))
    for knd in range(len(Ks)):
        K = Ks[knd]

        rho, nu = get_col(xx, K-1), get_col(xy, K-1)
        ratios = FixTerms(rho/nu)

        est = np.empty((len(alphas), 1))
        for ind in range(len(alphas)):
            alpha = alphas[ind]

            es = (((N-1)/M)**(1-alpha))*(ratios**(dim*(1-alpha))).mean()
            B = np.exp(gammaln(K)*2-gammaln(K+1-alpha)-gammaln(K+alpha-1))

            rs[ind,knd] = es*B
    
    return rs

def NPR(xx, xy, yy, yx, opt):
    '''nonparametric Renyi divergence estimator.
    returns a matrix. each row for one alpha in alphas. each col for one K in opt.Ks
    '''

    alphas = np.array(opt['alphas'])
    Ks = np.array(opt['Ks'])

    if xy is None: # identical bags
        return np.zeros((len(alphas), len(Ks)))

    alphas[alphas == 1] = 0.99 # approximate KL
    est = NPAlpha(xx, xy, yy, yx, alphas, opt)
    return np.maximum(0, np.log(np.maximum(est,eps))/(col(alphas) - 1))

def NPH(xx, xy, yy, yx, opt):
    '''nonparametric Hellinger distance estimator.
    returns a vector. one elem for one K in opt.Ks
    '''

    if xy is None: # identical bags
        return np.zeros(len(opt['Ks']))

    est = NPAlpha(xx, xy, yy, yx, [0.5], opt)
    return np.sqrt(np.maximum(0, 1 - est))

def L2Dist2(A, B):
    return -2*np.dot(A, B.T) + col((A**2).sum(1)) + (B**2).sum(1)

def NNSearch(args):
    x, y, K = args
    N, dim = x.shape

    if searcher is None:
        D = L2Dist2(x, y)
        idx = np.argsort(D, 1)[:,:K]
        dist = np.empty((x.shape[0], K))
        for i in range(x.shape[0]): dist[i] = D[i, idx[i]]
    else:
        idx, dist = searcher.nn(y, x, K, algorithm = 'linear' if dim > 5 else 'kdtree')
    dist, idx = dist.astype('float64'), idx.astype('uint16')

    # protect against identical points.
    min_dist = min(1e-2, 1e-100**(1.0/dim))
    return (np.maximum(min_dist, np.sqrt(dist)), idx)

def Handler(xx, xy, yy, yx, rt = None):
    r = [None]*len(funcs)
    for i in range(len(funcs)):
        f, sym, need_idx = funcs[i]
        if sym and rt is not None: # reuse symmetric results
            r[i] = rt[i]
        else:
            if not need_idx:
                r[i] = f(xx[0], xy[0], yy[0], yx[0], opt)
            else:
                r[i] = f(xx[0], xx[1], xy[0], xy[1], 
                         yy[0], yy[1], yx[0], yx[1], opt)
    return r

bags = []
xxs = []

def ProcPair(args):
    row, col = args
    
    if row == col:
        xx = xxs[row]
        r = Handler(xx, (None,None), (None,None), (None,None))
        rt = r
    else:
        xbag, xx, ybag, yy = bags[row], xxs[row], bags[col], xxs[col]

        mK = max(opt['Ks'])
        xy = NNSearch((xbag,ybag,mK))
        yx = NNSearch((ybag,xbag,mK))

        r = Handler(xx, xy, yy, yx)
        rt = Handler(yy, yx, xx, xy, r)
    
    for ind in range(len(r)):
        r[ind] = np.array(r[ind]).ravel()
        rt[ind] = np.array(rt[ind]).ravel()
    
    return (row, col, np.hstack(r).astype('float32'), np.hstack(rt).astype('float32'))

#########################################################

# function, is_symmetric, need_index
funcs = [(NPH,False,False),
         (NPL2,True,False), 
         (NPR,False,False)]
opt = {'Ks':np.array([1,3,5,10]),
       #'alphas':np.array([-1,-0.5,-0.2,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.01,1.1,1.2,1.3,1.4,1.5,2]),
	   'alphas':np.array([0.5,0.7,0.9,1])}

div_names = ['NP-H[K=%d]'%k for k in opt['Ks']]
div_names.extend(['NP-L2[K=%d]'%k for k in opt['Ks']])
div_names.extend(['NP-R[a=%g,K=%d]'%e for e
                  in itertools.product(opt['alphas'], opt['Ks'])])

def CheckArgs():
    if len(sys.argv) < 3:
        msg = '''python knn_stat_proc.py input_mat_file var_name [output_mat_file=input_mat_file.py_divs.mat] [n_proc=1] [n_point=inf]

Compute divergences and set kernels that are based on KNN statistics. 

input_mat_file: the input .mat file. it should be in hdf5 format i.e. saved using flag '-v7.3'.
var_name: the name of the data variable. it should be a cell array of N x dim matrices.
output_mat_file: the output .mat file.
n_proc: the number processes to use for parallel processing.
n_point: optional, the number of point to use per group.

The following quantities are computed:

''' + ', '.join(div_names) + '''

NP-H:   non-parametric estimation of the Hellinger distance.
NP-L2:  non-parametric estimation of the L2 distance.
NP-R:   non-parametric estimation of the Renyi distance.

This script does brutal force search for high-dim data. For low-dim data, the Matlab version should be fast enough.

The following python packages are required to run this script:
numpy: http://sourceforge.net/projects/numpy/files/NumPy/
scipy: http://sourceforge.net/projects/scipy/files/scipy/
h5py: http://code.google.com/p/h5py/
flann: http://www.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN

Liang Xiong, lxiong@cs.cmu.edu
'''
        print msg
        sys.exit(1)

if __name__ == '__main__':
    CheckArgs()

    input_file, input_var = sys.argv[1:3]
    output_file = input_file + '.py_divs.mat' if len(sys.argv) < 4 else sys.argv[3]
    n_proc = 1 if len(sys.argv) < 5 else int(sys.argv[4])
    n_point = 0 if len(sys.argv) < 6 else int(sys.argv[5])
    imap = itertools.imap

    print 'Reading data... '
    with h5.File(input_file, 'r') as f:
        for row in f[input_var]:
            for ptr in row:
                x = np.array(f[ptr])
                x = np.ascontiguousarray(x.T, dtype=np.float32)
                if n_point > 0 and n_point < x.shape[0]:
                    p = np.random.permutation(x.shape[0]-1)[:n_point]
                    x = x[p]
                bags.append(x)
                    
    # bags = bags[:10]

    M = len(bags)
    opt['dim'] = bags[0].shape[1]
    mK = max(opt['Ks'])

    print 'KNN Processing: M = %d, X = %d x %d,  K = %d, PID = %d.'%(M, bags[0].shape[0], bags[0].shape[1], mK, os.getpid())

    if n_proc > 1:
        pool = mp.Pool(n_proc)
        map, imap = pool.map, pool.imap_unordered

    print 'Preparing...'
    xxs = map(NNSearch, [(b,b,mK+1) for b in bags])
    xxs = [(xx[:,1:], xxi[:,1:]) for xx, xxi in xxs]
    # put xxs into the forked space
    if n_proc > 1:
        pool.close()
        pool = mp.Pool(n_proc)
        map, imap = pool.map, pool.imap_unordered

    jobs = []
    for i in range(M):
        for j in range(i,M):
            jobs.append((i, j))

    ts = tic()
    rs = []
    report_interval = max(100, int(len(jobs)/50))
    for r in imap(ProcPair, jobs):
        rs.append(r)
        c = len(rs)

        if (c <= 1000 and c % 100 == 0) or (c > 1000 and c % report_interval == 0):
            print '%s: Processed %0.2f%%. ET = %0.2f, ETA = %0.2f.' % (
                now(), 100.0*c/len(jobs), 
                toc(ts)/60.0, toc(ts)/c*(len(jobs) - c)/60.0)
    
    R = np.empty((M, M, rs[0][2].size), dtype = np.float32)
    R[:] = np.nan
    for i, j, r, rt in rs:
        R[i,j,:], R[j,i,:] = r, rt
    print 'Finished in %0.2f min.' % (toc(ts)/60.0)

    print 'Output results to ', output_file
    opt['Ds'] = R
    opt['div_names'] = div_names
    opt['ET'] = toc(ts)
    sio.savemat(output_file, opt, oned_as = 'column')

    assert not np.any(np.isnan(R.ravel())), 'nan found in the result'
    assert not np.any(np.isinf(R.ravel())), 'inf found in the result'
