from functools import partial
import os
import sys

if sys.version_info.major == 2:
    from StringIO import StringIO
else:
    from io import StringIO

import numpy as np
import h5py

from get_divs import fix_terms_clip, get_divs

################################################################################


class capture_output(object):
    def __init__(self, do_stdout=True, do_stderr=True, merge=False):
        self.do_stdout = do_stdout
        self.do_stderr = do_stderr
        self.merge = merge

        if do_stdout and do_stderr and merge:
            self.stdout = self.stderr = StringIO()
        else:
            if do_stdout:
                self.stdout = StringIO()
            if do_stderr:
                self.stderr = StringIO()

    def __enter__(self):
        if self.do_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = self.stdout
        if self.do_stderr:
            self.old_stderr = sys.stderr
            sys.stderr = self.stderr

    def __exit__(self, exc_type, exc_value, traceback):
        if self.do_stdout:
            sys.stdout = self.old_stdout
        if self.do_stderr:
            sys.stderr = self.old_stderr


################################################################################

def test_fix_terms_simple100():
    terms = np.array([
         0.5377,  1.8339, -2.2588,  0.8622,  0.3188, -1.3077, -0.4336,  0.3426,
         3.5784,  2.7694, -1.3499,  3.0349,  0.7254, -0.0631,  0.7147, -0.2050,
        -0.1241,  1.4897,  1.4090,  1.4172,  0.6715, -1.2075,  0.7172,  1.6302,
         0.4889,  1.0347,  0.7269, -0.3034,  0.2939, -0.7873,  0.8884, -1.1471,
        -1.0689, -0.8095, -2.9443,  1.4384,  0.3252, -0.7549,  1.3703, -1.7115,
        -0.1022, -0.2414,  0.3192,  0.3129, -0.8649, -0.0301, -0.1649,  0.6277,
         1.0933,  1.1093, -0.8637,  0.0774, -1.2141, -1.1135, -0.0068,  1.5326,
        -0.7697,  0.3714, -0.2256,  1.1174, -1.0891,  0.0326,  0.5525,  1.1006,
         1.5442,  0.0859, -1.4916, -0.7423, -1.0616,  2.3505, -0.6156,  0.7481,
        -0.1924,  0.8886, -0.7648, -1.4023, -1.4224,  0.4882, -0.1774, -0.1961,
         1.4193,  0.2916,  0.1978,  1.5877, -0.8045,  0.6966,  0.8351, -0.2437,
         0.2157, -1.1658, -1.1480,  0.1049,  0.7223,  2.5855, -0.6669,  0.1873,
        -0.0825, -1.9330, -0.4390, -1.7947
    ])

    expected = terms.copy()
    expected[8] = 3.3067
    expected.sort()

    fixed = fix_terms_clip(terms, tail=0.01)
    fixed.sort()

    assert np.allclose(fixed, expected, atol=1e-3)


def test_fix_terms_other100():
    terms = np.array([
         0.9195, -0.0617, -1.3385, -0.1218, -2.1000,  1.8291,  0.0549, -0.0789,
        -0.2861,  0.3739,  0.0027, -1.2186, -1.2901,  1.1663, -0.8901,  1.4472,
        -1.7756, -0.8204, -1.0579,  1.0077, -0.4595,  0.7860, -0.8349,  0.6164,
        -0.4736,  0.1797,  0.6522,  6.2691,  8.1633,  1.1778, -0.9921, -0.7535,
         1.4361,  0.3297, -0.5314,  1.7876,  0.0150, -0.7715, -0.8813,  1.1515,
         0.6752,  0.3413, -1.1232,  0.6571,  3.2662,  0.2452, -0.1967, -0.0537,
         1.2281, -0.1495, -0.8551,  0.2521,  0.9336,  2.1212,  0.6745,  0.1170,
         0.8917, -0.0123, -2.1032, -2.1327,  1.4584,  0.8522, -0.8366,  0.9018,
         1.3986,  0.3386, -0.2276, -0.7302, -0.9163,  0.0853,  1.2486,  0.0560,
         0.9663,  0.9855,  1.0368,  0.0317,  0.9394, -1.7035, -0.3171, -2.2082,
         0.0728,  1.2559, -0.0835,  0.3500, -0.0683, -0.6434,  0.9107, -0.8301,
         0.4882, -0.4319, -0.5635, -1.0781,  0.5531,  0.7233,  1.2353,  0.1558,
        -0.6426, -0.5250,  0.2199,  0.2584
    ])

    expected = terms.copy()
    expected[[27, 28, 44]] = 2.1212
    expected.sort()

    fixed = fix_terms_clip(terms, tail=0.035)
    fixed.sort()

    assert np.allclose(fixed, expected, atol=1e-4)


def test_fix_terms_with_inf_above():
    terms = np.array([
         1.1802, -0.5111, -1.3504, -0.3443, -0.7929, -0.7879,  0.8764, 19.6827,
         0.2975, -0.1433, 16.8614, 16.2429,  0.7989, -0.2036, -0.5767, -0.8718,
         0.1641,  0.0836, -1.2879,  0.1785,  0.6520, -1.2273,  1.3920, -1.1537,
         1.1435,  np.inf,  0.7007,  0.1004, 19.5552,  1.4390, -0.5372,  0.1011,
         0.3774,  0.0080,  0.1638, -0.0506, -0.5877,  1.1004,  0.9916,  0.6633,
         0.7530, -0.3251,  0.2590,  0.7998, -1.6068,  np.inf,  0.6035,  np.inf,
        -1.0864,  0.1909, -1.4197,  0.6826,  1.6760,  0.0179, -0.5544,  0.9308,
         2.5318, -0.2052, -0.7302,  0.5996,  0.2461,  0.3067, -0.2012,  2.0541,
        -0.7348, -0.4079, -1.0718,  1.4942,  0.6476, -0.2289, -1.2232,  np.inf,
        -0.0615,  0.7256, -1.0711, -1.9654, -0.5362,  1.7854,  0.7884,  0.3270,
        -0.2929
    ])

    expected = terms.copy()
    expected[[7, 10, 11, 25, 28, 45, 47, 71]] = 8.0163
    expected.sort()

    fixed = fix_terms_clip(terms, tail=0.1)
    fixed.sort()

    assert np.allclose(fixed, expected, atol=1e-4)


def test_fix_terms_with_inf_below():
    terms = np.array([
         1.1802, -0.5111, -1.3504, -0.3443, -0.7929, -0.7879,  0.8764, 19.6827,
         0.2975, -0.1433, 16.8614, 16.2429,  0.7989, -0.2036, -0.5767, -0.8718,
         0.1641,  0.0836, -1.2879,  0.1785,  0.6520, -1.2273,  1.3920, -1.1537,
         1.1435,  np.inf,  0.7007,  0.1004, 19.5552,  1.4390, -0.5372,  0.1011,
         0.3774,  0.0080,  0.1638, -0.0506, -0.5877,  1.1004,  0.9916,  0.6633,
         0.7530, -0.3251,  0.2590,  0.7998, -1.6068,  np.inf,  0.6035,  np.inf,
        -1.0864,  0.1909, -1.4197,  0.6826,  1.6760,  0.0179, -0.5544,  0.9308,
         2.5318, -0.2052, -0.7302,  0.5996,  0.2461,  0.3067, -0.2012,  2.0541,
        -0.7348, -0.4079, -1.0718,  1.4942,  0.6476, -0.2289, -1.2232,  np.inf,
        -0.0615,  0.7256, -1.0711, -1.9654, -0.5362,  1.7854,  0.7884,  0.3270,
        -0.2929
    ])

    expected = terms.copy()
    expected[[25, 45, 47, 71]] = 19.6827
    expected.sort()

    fixed = fix_terms_clip(terms, tail=0.02)
    fixed.sort()

    assert np.allclose(fixed, expected, atol=1e-4)


def test_fix_terms_with_inf_and_nan():
    terms = np.array([
         0.2346,  np.nan,  0.0160,  np.nan,  1.1949, -1.4867, -0.0240,  3.7520,
        -0.9096, -0.5122,  0.1069, -0.3973,  0.7500, -1.3019, -0.9338, -0.2939,
         1.2118, -1.0767, -1.3027,  0.0099,  0.4957, -0.6932, -0.5446, -0.1583,
         0.4763,  1.0468, -0.0382,  0.5777, -0.4535,  1.1198,  0.8421, -0.4130,
         1.1107,  0.0005,  1.2196,  0.1620,  1.1247, -1.9055, -0.5186, -0.8005,
         0.6188,  0.8332,  0.8700,  0.9239,  9.8975, -0.2494,  0.2930,  1.7697,
         np.inf, -0.8324,  1.4550, -0.9705, -0.9090, -0.7298, -0.3125,  0.5379,
         1.0355,  1.1462,  0.2040, -1.1386, -0.0775, -0.9242,  2.4677,  6.2213,
        -1.1986, -0.0884,  0.5466,  0.6762,  0.2894,  2.2231,  np.inf
    ])

    expected = terms.copy()
    expected = expected[np.logical_not(np.isnan(expected))]
    expected[[46, 68]] = 9.8975
    expected.sort()

    fixed = fix_terms_clip(terms, tail=0.02)
    fixed.sort()

    assert np.allclose(fixed, expected, atol=1e-4)

################################################################################

K = 3
div_funcs = ['bc', 'hellinger', 'l2', 'renyi:.999']
div_names = ['Bhattacharyya coefficient', 'Hellinger distance', 'L2 divergence',
             'Renyi-0.999 divergence']


def load_bags(filename, groupname):
    bags = []
    labels = []
    with h5py.File(filename, 'r') as f:
        for label, group in f[groupname].iteritems():
            if label == 'divs':
                continue
            for bag in group.itervalues():
                bags.append(bag[...])
                labels.append(label)
    return bags, labels


def load_divs(filename, groupname):
    divs = []
    with h5py.File(filename, 'r') as f:
        g = f[groupname]['divs']
        for name in div_names:
            divs.append(g[name][...])
    return divs


def assert_close(got, expected, msg, atol=1e-4):
    assert np.allclose(got, expected, atol=atol), msg


def check_div(bags, expected, name, **args):
    capturer = capture_output(True, True, merge=False)
    with capturer:
        divs = get_divs(bags, specs=div_funcs, Ks=[K],
                        tail=.01, min_dist=0, fix_mode='clip', **args)

    argstr = ', '.join('{}={}'.format(k, v) for k, v in args.iteritems())

    divs = divs.transpose((2, 0, 1))
    for df, calc, exp in zip(div_funcs, divs, expected):
        f = partial(assert_close, calc, exp,
                    "bad results for {}:{}".format(name, df))
        f.description = "divs: {} - {} - {}".format(name, df, argstr)
        yield f,


def test_divs():
    filename = os.path.join(os.path.dirname(__file__), 'test_dists.hdf5')
    args = [{'n_proc': n_proc, 'status_fn': status_fn}
            for n_proc in [1, None]
            for status_fn in [None, True]]
    # TODO: test a custom status_fn also

    for groupname in ['gaussian', 'gaussian-50']:
        bags, labels = load_bags(filename, groupname)
        expected = load_divs(filename, groupname)
        name = "test_dists.{}".format(groupname)
        for extra in args:
            for test in check_div(bags, expected, name, **extra):
                yield test

################################################################################

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('error', module='get_divs')

    import nose
    nose.main()
