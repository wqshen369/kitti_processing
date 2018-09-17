#!/usr/bin/env python
import h5py


if __name__ == "__main__":
    f = h5py.File('hdf5_2048/train_10.h5' , 'r')
    print f.keys()
    a = f['data'][:]
    b = f['label'][:]
    print a.shape
    print b.shape
    # print a
    # print b
