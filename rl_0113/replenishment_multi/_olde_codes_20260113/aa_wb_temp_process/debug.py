import sys
import runpy
import os

os.chdir('/Users/weibin/Desktop/rp/replenishment')

args = 'python /Users/weibin/Desktop/rp/replenishment/replenish_evaluate.py\
 --task_name cache_rp\
 --data_ver wb_v1\
 --para_ver 0620_v2\
 --test_version _test_100k\
 --test_data_path /Users/weibin/Desktop/rp/replenishment/data/simu_data_v100_case.csv'

args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')