#!/usr/bin/env python
from __future__ import division, print_function

from bisect import bisect
from collections import defaultdict
import csv
from functools import partial
from itertools import islice, takewhile
import os
import random
import warnings

import numpy as np

from vlfeat.phow import (vl_phow, DEFAULT_MAGNIF, DEFAULT_CONTRAST_THRESH,
                         DEFAULT_WINDOW_SIZE, DEFAULT_COLOR, COLOR_CHOICES)

from .utils import (positive_int, positive_float, nonnegative_float,
                    strict_map, str_types, confirm_outfile, iteritems)
from .features import Features

# NOTE: depends on skimage for resizing, and either opencv, matplotlib with PIL,
# or skimage with one of the plugins below for reading images


DEFAULT_STEP = 20
DEFAULT_SIZES = (6, 9, 12)
def get_features(img, color=DEFAULT_COLOR,
                 step=DEFAULT_STEP, sizes=DEFAULT_SIZES,
                 magnif=DEFAULT_MAGNIF, window_size=DEFAULT_WINDOW_SIZE,
                 phow_blank_thresh=DEFAULT_CONTRAST_THRESH,
                 fast=True, verbose=False):
    '''
    Extract dense SIFT features from img.
    Returns:
        - a row-instance array of feature descriptors (128 or 384 dimensional)
        - a row-instance array of locations: [x, y, scale]
    '''

    frames, descrs = vl_phow(img, fast=fast, verbose=verbose,
        color=color, step=step, sizes=sizes, magnif=magnif,
        window_size=window_size, contrast_thresh=phow_blank_thresh)

    frames = frames[:, [0, 1, 3]]  # throw out norm data
    frames[:, 2] /= magnif

    return frames, descrs


IMREAD_MODES = ['skimage-pil', 'skimage-qt', 'skimage-gdal', 'cv2',
                'matplotlib', 'skimage-freeimage']
def _find_working_imread(modes=IMREAD_MODES):
    "Finds an image-reading mode that works; returns the name and a function."
    if isinstance(modes, str_types):
        modes = [modes]

    for mode in modes:
        try:
            if mode.startswith('skimage-'):
                from skimage.io import use_plugin, imread
                use_plugin(mode[len('skimage-'):])
            elif mode == 'cv2':
                import cv2
                def imread(f):
                    img = cv2.imread(f)
                    if img.ndim == 3:
                        b, g, r = np.rollaxis(img, axis=-1)
                        return np.dstack([r, g, b])
                    return img
            elif mode == 'matplotlib':
                import matplotlib.pyplot as mpl
                imread = lambda f: mpl.imread(f)[::-1]

            return mode, imread

        except ImportError:
            pass
    else:
        raise ImportError("couldn't import any of {}".format(', '.join(modes)))


def _load_features(filename, imread_mode=IMREAD_MODES, size=None, **kwargs):
    """
    Loads filename, optionally resizes it, and calls get_features.

    size should be either None or a (width, height) tuple, where having one
    element be None means that the relevant entry is chosen so as to maintain
    the aspect ratio.
    """
    _, imread = _find_working_imread(imread_mode)
    img = imread(filename)

    if size is not None and size != (None, None):
        import skimage.transform
        curr_x, curr_y = img.shape[:2]
        new_y, new_x = size
        if new_x is None:
            newsize = (int(np.round(curr_x * new_y / curr_y)), new_y)
        elif new_y is None:
            newsize = (new_x, int(np.round(curr_y * new_x / curr_y)))
        else:
            newsize = size
        img = skimage.transform.resize(img, newsize)

    return get_features(img, **kwargs)


def _load_extras(paths):
    "Returns a list of dicts mapping extra_name to a value for each file."
    dirnames = set(os.path.dirname(path) for path in paths)

    lists = {}
    csv_contents = defaultdict(dict)
    for dirname in dirnames:
        lists[dirname] = ls = sorted(os.listdir(dirname))
        for csv_filename in (f for f in ls if f.endswith('.csv')):
            name = csv_filename[:-len('.csv')]
            vals = {}
            with open(os.path.join(dirname, csv_filename)) as f:
                for line in csv.reader(f):
                    try:
                        fname, val = line
                        vals[fname] = float(val)
                    except ValueError:
                        warnings.warn('Bad line in {} (length {})'.format(
                            os.path.join(dirname, csv_filename), len(line)))
                        break
                else:
                    csv_contents[dirname][name] = vals

    extras = []
    for path in paths:
        dirname, basename = os.path.split(path)
        files_in_dir = lists[dirname]
        pos = bisect(files_in_dir, basename)
        extension_files = takewhile(lambda x: x.startswith(basename),
                                    islice(files_in_dir, pos, None))
        extra = {}
        for fname in extension_files:
            extra_name = fname[len(basename):]
            if extra_name[0] in '._-':
                extra_name = extra_name[1:]
            extra[extra_name] = np.loadtxt(os.path.join(dirname, fname))
        for extra_name, vals in iteritems(csv_contents[dirname]):
            if basename in vals:
                extra[extra_name] = vals[basename]
        extras.append(extra)
    return extras


def _sample_uniform(lst, n):
    if len(lst) <= n:
        return lst
    indices = np.round(np.linspace(0, len(lst) - 1, n))
    assert np.all(np.diff(indices) >= 1)
    return [lst[int(i)] for i in indices]

SAMPLERS = {
    'first': lambda lst, n: lst[:n],
    'random': lambda lst, n: random.sample(lst, min(n, len(lst))),
    'uniform': _sample_uniform,
}

DEFAULT_EXTENSIONS = frozenset(['jpg', 'png', 'bmp'])
def find_paths(dirs, img_per_cla=None, sampler='first',
               extensions=DEFAULT_EXTENSIONS):
    '''
    Choose paths from directories.

    Returns corresponding lists of selected categories and paths.
    '''
    if not hasattr(dirs, 'items'):
        dirs = dict((dirname, dirname) for dirname in dirs)

    # make a dict of cat => list of (dirname, fname) pairs
    ims_by_cat = defaultdict(list)
    seen_names = defaultdict(set)
    for dirname, cat in iteritems(dirs):
        for fname in os.listdir(dirname):
            if '.' in fname and fname.rsplit('.', 1)[1].lower() in extensions:
                if fname in seen_names[cat]:
                    raise ValueError("more than one {!r} with category {!r}"
                                     .format(fname, cat))
                seen_names[cat].add(fname)
                ims_by_cat[cat].append((dirname, fname))

    # do sampling and split it up
    sample = (lambda x, n: x) if img_per_cla is None else SAMPLERS[sampler]
    cats, paths = zip(*[
        (cat, os.path.join(dirname, fname))
        for cat, images in iteritems(ims_by_cat)
        for dirname, fname in sample(sorted(images), img_per_cla)
    ])
    return cats, paths


def extract_image_features(paths, cats, imread_mode=IMREAD_MODES,
                           parallel=False, **kwargs):
    '''
    Extracts features from images in a list of data directories.

    dirs: either an iterable of directory names
          or a dict with directory names as keys and categories as values
    img_per_cla: how many images to read from each directory; None means all
    sampler: 'first' for the first img_per_cla lexicographically
             'uniform': evenly spaced from the images
             'random': a random sample of the images
    extensions: (case-insensitive) filename extensions to treat as images
    parallel: - if False (default), run serially
              - if an object with a `map` method (e.g. multiprocessing.Pool),
                run extractions through that
              - if True, run in a pool with one process per CPU core
              - if an integer, run in a pool with that many processes

    Other arguments are passed on to get_features().

    Returns a Features tuple.
    '''
    extras = _load_extras(paths)
    image_names = [os.path.basename(path) for path in paths]

    # sort out parallelism options
    pool = None
    if hasattr(parallel, 'map'):
        do_map = parallel.map
    elif parallel is False:
        do_map = strict_map
    else:
        import multiprocessing as mp
        pool = mp.Pool(None if parallel is True else parallel)
        do_map = pool.map

    # find an imread mode now, so we don't have to try bad imports every time
    imread_mode, _ = _find_working_imread(imread_mode)

    # do the actual extraction, skipping any images we get no features from
    n_skipped = [0]  # python closure silliness
    def predicate(f_d):
        if f_d[0].size == 0:
            n_skipped[0] += 1
            return False
        return True
    load_features = partial(_load_features, imread_mode=imread_mode, **kwargs)
    frames, descrs = zip(*filter(predicate, do_map(load_features, paths)))

    if n_skipped[0]:
        msg = "Skipped {} images that got no features out.".format(n_skipped[0])
        warnings.warn(msg)

    if pool is not None:
        pool.close()
        pool.join()
    return Features(descrs, categories=cats, names=image_names,
                    frames=frames, **extras)


################################################################################
### Command line

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract dense SIFT features from a collection of images.")

    # helper for boolean flags
    # based on http://stackoverflow.com/a/9236426/344821
    class ActionNoYes(argparse.Action):
        def __init__(self, opt_name, off_name=None, dest=None,
                     default=True, required=False, help=None):

            if off_name is None:
                off_name = 'no-' + opt_name
            self.off_name = '--' + off_name

            if dest is None:
                dest = opt_name.replace('-', '_')

            super(ActionNoYes, self).__init__(
                    ['--' + opt_name, '--' + off_name],
                    dest, nargs=0, const=None,
                    default=default, required=required, help=help)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, option_string != self.off_name)

    _def = " default %(default)s."

    parser.add_argument('--n-proc', default=None, dest='parallel',
        type=lambda x: False if x.strip() == '1' else positive_int(x),
        help="Number of processes to use; default is as many as CPU cores.")

    # options for finding and loading images
    files = parser.add_argument_group('File options')

    files.add_argument('--paths-csv', metavar='FILE',
        help="A CSV file with columns 'path' and optionally 'cat' specifying "
             "the files to load (overrides the directory-related args).")

    parser.set_defaults(dirs={})

    class AddDirs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            join = partial(os.path.join, values)
            spec = dict((join(d), d) for d in os.listdir(values)
                        if os.path.isdir(join(d)))
            getattr(namespace, self.dest).update(spec)
    files.add_argument('--root-dir', action=AddDirs, dest='dirs', metavar='DIR',
        help="Adds all the directories under this path as class-level dirs.")

    class AddDir(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            getattr(namespace, self.dest).update(dict(
                (d, os.path.basename(d.rstrip('/'))) for d in values))
    files.add_argument('--dirs', nargs='+', action=AddDir, dest='dirs',
        metavar='DIR', help="Adds the path as a directory.")

    class AddDirWithCat(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            path, cat = values
            getattr(namespace, self.dest)[path] = cat
    files.add_argument('--dir-with-cat', '--labeled-dir',
        nargs=2, action=AddDirWithCat, dest='dirs', metavar=('DIR', 'CATEGORY'),
        help="Adds a directory with a specified category.")

    files.add_argument('--num-per-category', default=None, type=int,
        dest='img_per_cla', metavar='NUM',
        help="Limit the number of images loaded from each class; "
             "default is unlimited.")
    files.add_argument('--sampler', choices=SAMPLERS, default='first',
        help="How to choose which images if there are more than the limit.")

    files.add_argument('--extensions', default=DEFAULT_EXTENSIONS,
        type=lambda s: set(s.lower().split(',')),
        help="Comma-separated list of (case-insensitive) filename extensions "
             "to load; default {}.".format(','.join(DEFAULT_EXTENSIONS)))

    files.add_argument('--imread-mode', choices=IMREAD_MODES,
        default=IMREAD_MODES, help="Choose a library for reading the images.")

    files.add_argument('--resize', default=None, dest='size',
        type=lambda s: [None if x == '*' else int(x) for x in s.split('x', 1)],
        help="Resize images to this size (e.g. 250x250). Use * to scale, "
             "so that 500x* makes images 500px wide while maintaining "
             "aspect ratio. Requires scikit-image. Default: keep at "
             "original size.")

    # options for output files
    out = parser.add_argument_group('Ouptut options')
    out.add_argument('--output-format',
        choices=['single-hdf5', 'perimage-npz'], default='single-hdf5',
        help="Output format: single-hdf5 for a single hdf5 file (the default), "
             "or perimage-npz for npz files for each image.")

    out.add_argument('save_path',
        help="The output file path if single-hdf5, "
              "or the base directory to put the per-image files.")

    # options for feature extraction
    sift = parser.add_argument_group('SIFT options')

    color = sift.add_mutually_exclusive_group()
    color.add_argument('--color', choices=COLOR_CHOICES,
                       nargs='?', const='hsv', default='gray')
    color.add_argument('--grayscale', action='store_const',
                       dest='color', const='gray')

    sift.add_argument('--step', type=positive_int, default=DEFAULT_STEP,
        help="The step between frame centers;" + _def)
    sift.add_argument('--sizes', default=DEFAULT_SIZES,
        type=lambda s: tuple(map(positive_int, s.split(','))),
        help="The scales to extract features at; default {}.".format(
            ', '.join(map(str, DEFAULT_SIZES))))
    sift.add_argument('--magnif', type=positive_float, default=DEFAULT_MAGNIF,
        help="The image is smoothed by a Gaussian kernel with "
             "std dev size/magnif;" + _def)
    sift.add_argument('--window-size', type=positive_float,
        default=DEFAULT_WINDOW_SIZE,
        help="Size of the Gaussian window, in spatial bin units;" + _def)
    sift.add_argument('--phow-blank-threshold', '--contrast-threshold',
        dest='phow_blank_thresh', metavar='NUM',
        type=nonnegative_float, default=DEFAULT_CONTRAST_THRESH,
        help="Contrast threshold under which features are zeroed;" + _def)
    sift._add_action(ActionNoYes('fast', 'slow', default=True,
        help="Whether to use fast SIFT computation in dsift; does by default."))

    args = parser.parse_args()
    if not args.dirs and not args.paths_csv:
        parser.error("Must specify some images to load.")
    return args


def main():
    args = parse_args()
    confirm_outfile(args.save_path, dir=args.output_format != 'single-hdf5')

    if args.paths_csv:
        cats = []
        paths = []
        with open(args.paths_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                paths.append(os.path.expanduser(row['path']))
                cats.append(row.get('cat', 'default'))
    else:
        cats, paths = find_paths(args.dirs,
                                 extensions=args.extensions,
                                 img_per_cla=args.img_per_cla,
                                 sampler=args.sampler)

    skip = set('paths_csv dirs extensions img_per_cla '
               'sampler output_format save_path'.split())
    kwargs = dict((k, v) for k, v in vars(args).iteritems() if k not in skip)
    features = extract_image_features(paths, cats, **kwargs)
    # TODO: show a progressbar here

    print("Saving results to '{}'".format(args.save_path))
    if args.output_format == 'single-hdf5':
        features.save_as_hdf5(args.save_path, args=repr(vars(args)))
    else:
        features.save_as_perbag(args.save_path, args=repr(vars(args)))

if __name__ == '__main__':
    main()
