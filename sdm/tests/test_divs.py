from __future__ import division
from functools import partial
import os
import sys

if sys.version_info.major == 2:
    from StringIO import StringIO
else:
    from io import StringIO

import numpy as np
import h5py
from scipy.special import psi

if __name__ == '__main__':
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))

from sdm.np_divs import estimate_divs, normalize_div_name
from sdm.features import Features
from sdm.utils import iteritems, itervalues, strict_map


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

def generate_gaussians(name, mean_std_nums, dim, min_pts, max_pts):
    bags = []
    categories = []
    for mean, std, num in mean_std_nums:
        cat_name = 'mean{}-std{}'.format(mean, std)
        for x in range(num):
            n_pts = np.random.randint(min_pts, max_pts+1)
            feats = np.random.normal(mean, std, size=(n_pts, dim))
            bags.append(feats)
            categories.append(cat_name)
    features = Features(bags, categories=categories)
    features.save_as_hdf5('data/{}.h5'.format(name))


################################################################################


def assert_close(got, expected, msg, atol=1e-8, rtol=1e-5):
    assert np.allclose(got, expected, atol=atol, rtol=rtol), msg


def load_divs(f, specs, Ks):
    specs = np.asarray(strict_map(normalize_div_name, specs))
    Ks = np.ravel(Ks)

    n_bags = next(itervalues(next(itervalues(f)))).shape[0]

    divs = np.empty((n_bags, n_bags, specs.size, Ks.size), dtype=np.float32)
    divs.fill(np.nan)

    for i, spec in enumerate(specs):
        try:
            group = f[spec]
        except KeyError:
            msg = "missing div func {} in {}"
            raise KeyError(msg.format(spec, f.filename))

        for j, K in enumerate(Ks):
            try:
                vals = group[str(K)]
            except KeyError:
                msg = "{} is missing K={} in {}"
                raise KeyError(msg.format(spec, K, f.filename))

            divs[:, :, i, j] = vals
    return divs


def check_div(feats, expected, specs, Ks, name, min_dist=None, **args):
    capturer = capture_output(True, True, merge=False)
    with capturer:
        ds = estimate_divs(feats, specs=specs, Ks=Ks, min_dist=min_dist, **args)

    argstr = ', '.join('{}={}'.format(k, v) for k, v in iteritems(args))

    for spec_i, spec in enumerate(specs):
        for K_i, K in enumerate(Ks):
            calc = ds[:, :, spec_i, K_i]
            exp = expected[:, :, spec_i, K_i]

            diff = np.abs(calc - exp)
            i, j = np.unravel_index(np.argmax(diff), calc.shape)
            msg = "bad results for {}:{}, K={}\n".format(name, spec, K) + \
                  "(max diff {} = |{} - {}| at {},{})".format(
                      diff[i, j], calc[i, j], exp[i, j], i, j)

            f = partial(assert_close, calc, exp, atol=5e-5, msg=msg)
            f.description = \
                "divs: {} - {}, K={} - {}".format(name, spec, K, argstr)
            yield f,


def test_divs():
    dir = os.path.join(os.path.dirname(__file__), 'data')
    argses = [{'cores': cores, 'status_fn': status_fn}
              for cores in [1, None]
              for status_fn in [None]]  # , True]]
    # TODO: test a custom status_fn also

    specs = ['hellinger', 'kl', 'l2', 'linear',
             'renyi:0.5', 'renyi:0.7', 'renyi:0.9', 'renyi:0.99']
    Ks = [1, 3, 5, 10]
    for name in ['gaussian-2d-mean0-std1,2', 'gaussian-20d-mean0-std1,2']:
        for dtype in [np.float64, np.float32]:
            feats = Features.load_from_hdf5(
                os.path.join(dir, name + '.h5'),
                features_dtype=dtype)

            with h5py.File(os.path.join(dir, name + '.divs.h5'), 'r') as f:
                expected = load_divs(f, specs, Ks)
                min_dist = f.attrs['min_dist']

            tests = []
            for args in argses:
                tests.extend(check_div(feats, expected, specs, Ks, name,
                                       min_dist=min_dist, **args))
            for test in sorted(tests, key=lambda t: t[0].description):
                yield test


def test_with_and_without_js():
    dir = os.path.join(os.path.dirname(__file__), 'data')
    name = 'gaussian-2d-mean0-std1,2'
    feats = Features.load_from_hdf5(os.path.join(dir, name + '.h5'))

    specs = ['hellinger']
    Ks = [3, 5]

    with h5py.File(os.path.join(dir, name + '.divs.h5'), 'r') as f:
        expected = load_divs(f, specs, Ks)
        min_dist = f.attrs['min_dist']

    est = partial(estimate_divs, feats, Ks=Ks, min_dist=min_dist,
                  status_fn=None)

    with capture_output(True, True, merge=False):
        ds = est(specs=specs + ['js'])
        oth_with = ds[:, :, :-1, :]
        js_with = ds[:, :, -1, :]

        js_without = est(specs=['js'])[:, :, 0, :]

    assert_close(oth_with, expected, atol=5e-5,
                 msg="others ({}) broke with JS".format(', '.join(specs)))
    assert_close(js_with, js_without, atol=5e-5,
                 msg="JS different with/without others")


def test_kl_simple():
    # verified by hand
    # Dhat(P||Q) = \log m/(n-1) + d / n  \sum_{i=1}^n \log \nu_k(i)/rho_k(i)
    x = np.reshape([0., 1, 3], (3, 1))
    y = np.reshape([.2, 1.2, 3.2, 7.2], (4, 1))

    n = x.shape[0]
    m = y.shape[0]

    x_to_y = np.log(m / (n-1)) + 1/n * (
        np.log(1.2 / 3) + np.log(.8 / 2) + np.log(1.8 / 3))
    y_to_x = np.log(n / (m-1)) + 1/m * (
        np.log(.8 / 3) + np.log(1.2 / 2) + np.log(2.2 / 3) + np.log(6.2 / 6))

    res = estimate_divs(Features([x, y]), specs=['kl'], Ks=[2]).squeeze()
    print res
    print x_to_y, y_to_x
    # assert res[0, 0] == 0
    # assert res[1, 1] == 0
    assert np.allclose(res[1, 0], y_to_x), "{} vs {}".format(res[1, 0], y_to_x)
    assert np.allclose(res[0, 1], x_to_y), "{} vs {}".format(res[0, 1], x_to_y)


def test_js_simple():
    # verified by hand
    x = np.reshape([0, 1, 3], (3, 1))
    y = np.reshape([.2, 1.2, 3.2, 6.2], (4, 1))

    mix_ent = np.log(2) + np.log(3) + psi(2) \
        + (np.log(.2) + np.log(.8) + np.log(1.8) - psi(1) - 2*psi(2)) / 6 \
        + (np.log(.2) + np.log(2) + np.log(3.2) - psi(1) - 3*psi(2)) / 8
    x_ent = np.log(2) + (np.log(3) + np.log(2) + np.log(3)) / 3
    y_ent = np.log(3) + (np.log(3) + np.log(2) + np.log(3) + np.log(5)) / 4
    right_js = mix_ent - (x_ent + y_ent) / 2
    expected = np.array([[0, right_js], [right_js, 0]])
    # TODO: what about clamping???

    est = estimate_divs(Features([x, y]), specs=['js'], Ks=[2],
                        status_fn=None).squeeze()
    assert_close(est, expected, atol=5e-5, msg="JS estimate not as expected")

################################################################################

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('error', module='sdm')

    import nose
    nose.main()
