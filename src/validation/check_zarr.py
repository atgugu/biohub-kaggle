import sys, zarr, numpy as np
import numcodecs.blosc; numcodecs.blosc.use_threads = False   # same decode path as our scripts
from concurrent.futures import ProcessPoolExecutor
def chk(v):
    bad=[]
    a=zarr.open_group(f'/workspace/data/train/{v}.zarr',mode='r')['0']
    for t in range(a.shape[0]):
        try: a[t,0,0,0]; a[t]
        except Exception as e: bad.append(t)
    return v,bad
if __name__=='__main__':
    vids=[l.strip() for l in open(sys.argv[1])]
    with ProcessPoolExecutor(4) as ex:
        for v,bad in ex.map(chk,vids):
            if bad: print(v,'bad frames:',bad,flush=True)
    print('checked',len(vids))
