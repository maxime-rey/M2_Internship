"""
File that returns the header from .fits files in a json.
"""

from __future__ import print_function

import argparse
import os
import json

from collections import OrderedDict
import astropy.io.fits as AF

__author__ = "Yannick Copin & Maxime Rey"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="returns the header from .fits file(s) in a json")
    parser.add_argument('-o', '--output', help='output json file name', required=True)
    parser.add_argument('-k', '--keys', help='only save specified key (input CSV)')
    parser.add_argument('files', nargs='+', help='input files')
    args = parser.parse_args() 

    adict = {}

    if args.keys is not None:
        keys = args.keys.split(',')

    for arg in args.files:
        header = AF.getheader(arg, ignore_missing_end=True)
        if args.keys is None:
            keys = header.keys()
        vals = OrderedDict([ (key, header[key]) for key in keys if (key != 'COMMENT')])
        print("%s: %d keywords" % (arg, len(vals)))
        adict[os.path.basename(arg)] = vals
        
    with open(args.output, 'w') as outfile:
        json.dump(adict, outfile)