"""Selected current-profile checks, in separate processes; not historical hang probes.

Run through tools/with-r600-clover.sh. Dependencies: Torch and NumPy.
Use --list to inspect, or pass selected check names. Each check has a timeout;
a timeout is a failure and may require checking the driver before continuing.
"""
import argparse
from pathlib import Path
import subprocess
import sys

CHECKS = [
    'check_flat_strided_copy',
    'check_pointwise_output_offsets',
    'check_unique_legacy_regression',
    'check_legacy_topk_masks',
    'check_index_sort_reduce',
    'check_fused_silu_backward',
    'check_filter_split',
    'check_gather_spatial4',
    'check_deferred_release',
    'check_memory_reporting',
]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--timeout',type=int,default=300)
    parser.add_argument('checks',nargs='*')
    args=parser.parse_args()
    if args.list:
        print('\n'.join(CHECKS));return
    selected=args.checks or CHECKS
    if any(name not in CHECKS for name in selected):parser.error('unknown check; use --list')
    for name in selected:
        print('RUN',name,flush=True)
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name(name+'.py'))],check=True,timeout=args.timeout)
    print('PASS',len(selected),'selected r600 checks',flush=True)

if __name__=='__main__':main()
