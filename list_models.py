# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:54:48 2018

@author: Stavros
"""

from os import listdir
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-CR', type=bool, default=False, help='critical data')
parser.add_argument('-iT', type=int, default=0, help='temperature index')
args = parser.parse_args()

if args.CR:
    models_dir = 'Weights/Critical'
else:
    from data.directories import T_list
    models_dir = 'Weights/T%.4f'%T_list[args.iT]

print(args.CR)
print(listdir(models_dir))