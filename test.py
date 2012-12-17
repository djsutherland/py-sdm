from __future__ import division, print_function

import multiprocessing as mp
import warnings


try:
    import progressbar as pb
except ImportError:
    warnings.warn("Can't import the progressbar module; output will suck.")

    def progress(**kwargs):
        def iterator(xs):
            if 'maxval' in kwargs:
                maxcount = maxval
            elif hasattr(xs, '__len__'):
                maxcount = len(xs)
            else:
                maxcount = None

            if maxcount:
                ticks = max(5, min(5000, maxcount / 100))
                suffix = ' / {}'.format(maxcount)
            else:
                ticks = 100
                suffix = ''

            for i, x in enumerate(xs):
                yield x
                if i % ticks == 0:
                    print('done', i, suffix, file=sys.stderr)
        return iterator
else:
    def progress(counter=True, **kwargs):
        try:
            widgets = kwargs.pop('widgets')
        except KeyError:
            if counter:
                widgets = [pb.SimpleProgress(), ' (', pb.Percentage(), ') ']
            else:
                widgets = [pb.Percentage(), ' ']
            widgets.extend((pb.Bar(), ' ', pb.ETA()))
        return pb.ProgressBar(widgets=widgets, **kwargs)

import time
def do_job(i, j):
    time.sleep(.01)
    return i + j

jobs = []
for i in range(100):
    for j in range(i, 100):
        jobs.append((i, j))

pbar = progress(maxval=len(jobs)).start()
pbar_lock = mp.Lock()
def update_pbar(*args, **kwargs):
    #print('bam')
    with pbar_lock:
        pbar.update(pbar.currval + 1)

pool = mp.Pool(2)
results = []
for job in jobs:
    results.append(pool.apply_async(do_job, job, callback=update_pbar))
#res = pool.starmap_async(do_job, jobs, callback=update_pbar)
#res.wait()

for res in results:
    res.wait()
