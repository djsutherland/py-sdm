#!/usr/bin/env python
from __future__ import division, print_function

from collections import defaultdict
from functools import partial
from operator import methodcaller
import os
import random

from get_divs import (positive_int, positive_float, nonnegative_float,
                      strict_map, str_types, izip, raw_input)
from vlfeat.phow import (vl_phow, DEFAULT_MAGNIF, DEFAULT_CONTRAST_THRESH,
                         DEFAULT_WINDOW_SIZE, DEFAULT_COLOR, COLOR_CHOICES)

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
        raise ImportError("couldn't import any of {}".format(
            ', '.join(imread_mode)))


def _load_features(filename, imread_mode=IMREAD_MODES, size=None, **kwargs):
    _, imread = _find_working_imread(imread_mode)
    img = imread(filename)

    if size is not None and size != (None, None):
        import skimage.transform
        curr_x, curr_y = img.shape[:2]
        new_x, new_y = size
        if new_x is None or new_y is None:
            scale = new_x / curr_x if new_y is None else new_y / curr_y
            img = skimage.transform.rescale(img, scale)
        else:
            img = skimage.transform.resize(img, size)

    return get_features(img, **kwargs)


def _sample_uniform(lst, n):
    if len(lst) <= n:
        return lst
    indices = np.round(np.linspace(0, len(lst) - 1, n))
    assert np.diff(indices) >= 1
    return [lst[int(i)] for i in indices]

SAMPLERS = {
    'first': lambda lst, n: lst[:n],
    'random': lambda lst, n: random.sample(lst, min(n, len(lst))),
    'uniform': _sample_uniform,
}

DEFAULT_EXTENSIONS = {'jpg', 'png', 'bmp'}
def extract_features(dirs, img_per_cla=None, sampler='first',
                     extensions=DEFAULT_EXTENSIONS,
                     imread_mode=IMREAD_MODES,
                     parallel=False, **kwargs):
    '''
    Extracts features from images in a list of data directories.

    dirs: either an iterable of directory names
          or a dict with directory names as keys and class labels as values
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

    Returns tuples of labels, image names, descriptor locations, descriptors.
    '''
    if not hasattr(dirs, 'items'):
        dirs = {dirname: dirname for dirname in dirs}

    # make a dict of label => list of (dirname, fname) pairs
    ims_by_label = defaultdict(list)
    seen_names = defaultdict(set)
    for dirname, label in dirs.iteritems():
        for fname in os.listdir(dirname):
            if fname.rsplit('.', 1)[1].lower() in extensions:
                if fname in seen_names[label]:
                    raise ValueError("more than one {!r} with label {!r}"
                            .format(fname, label))
                seen_names[label].add(fname)
                ims_by_label[label].append((dirname, fname))

    # do sampling and split it up
    sample = (lambda x, n: x) if img_per_cla is None else SAMPLERS[sampler]
    labels, image_names, paths = zip(*[
        (label, fname, os.path.join(dirname, fname))
        for label, images in ims_by_label.items()
        for dirname, fname in sample(sorted(images), img_per_cla)
    ])

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

    # do the actual extraction
    load_features = partial(_load_features, imread_mode=imread_mode, **kwargs)
    frames, descrs = zip(*do_map(load_features, paths))
    return labels, image_names, frames, descrs


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

    parser.add_argument('save_file',
        help="Save into this HDF5 file. Each image's features go in "
             "/label/filename/{'features','frames'}.")

    # options for finding and loading images
    files = parser.add_argument_group('File options')

    parser.set_defaults(dirs={})

    class AddDirs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            join = partial(os.path.join, values)
            spec = {join(d): d for d in os.listdir(values)
                    if os.path.isdir(join(d))}
            getattr(namespace, self.dest).update(spec)
    files.add_argument('--root-dir', action=AddDirs, dest='dirs', metavar='DIR',
        help="Adds all the directories under this path as class-level dirs.")

    class AddDir(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            getattr(namespace, self.dest).update({d: d for d in values})
    files.add_argument('--dirs', nargs='+', action=AddDir, dest='dirs',
        metavar='DIR', help="Adds the path as a directory.")

    class AddLabeledDir(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            path, label = values
            getattr(namespace, self.dest)[path] = label
    files.add_argument('--labeled-dir', nargs=2, action=AddLabeledDir,
        dest='dirs', metavar=('DIR', 'LABEL'),
        help="Adds a directory with a specified class label.")

    files.add_argument('--num-per-class', default=None, type=int,
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
    if not args.dirs:
        parser.error("Must specify some images to load.")

    save_file = args.save_file
    del args.save_file
    return args, save_file


def save_features(filename, ls, ns, fs, ds):
    '''
    Saves the results of extract_features() into an HDF5 file.
    Each bag is saved as "features" and "frames" in /label/filename.
    '''
    with h5py.File(save_file) as f:
        for label, name, frames, descrs in izip(ls, ns, fs, ds):
            g = f.require_group(label).create_group(name)
            g['frames'] = frames
            g['features'] = descrs


def read_features(filename):
    '''Reads the file format saved by save_features().'''
    import h5py
    labels = []
    names = []
    frames = []
    descrs = []
    with h5py.File(filename, 'r') as f:
        for label, label_g in f.items():
            for fname, g in label_g.items():
                labels.append(label)
                names.append(fname)
                frames.append(g["frames"][...])
                descrs.append(g["features"][...])
    return labels, names, frames, descrs


def main():
    import h5py
    import sys

    args, save_file = parse_args()

    if os.path.exists(save_file):
        resp = raw_input("Output file '{}' already exists; will be deleted. "
                         "Continue? [yN] ".format(save_file))
        if not resp.lower().startswith('y'):
            sys.exit("Aborting.")
    try:
        with open(save_file, 'w') as f:
            pass
        os.remove(save_file)
    except Exception as e:
        sys.exit("{}: can't write to '{}'".format(e, save_file))

    results = extract_features(**vars(args))
    # TODO: progressbar

    print("Saving results to '{}'".format(save_file))
    save_features(save_file, *results)


if __name__ == '__main__':
    main()
