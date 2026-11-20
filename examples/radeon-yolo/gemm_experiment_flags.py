"""Named, opt-in, math-preserving GEMM experiments."""
import os
FLAGS=dict(loader='DLPRIM_GEMM_COALESCED_B',barrier='DLPRIM_GEMM_SKIP_FINAL_BARRIER',static='DLPRIM_GEMM_STATIC_SHAPE',stores='DLPRIM_GEMM_VECTOR_STORES',selective='DLPRIM_GEMM_SELECTIVE_TILE',tiny='DLPRIM_TINY_POINTWISE',stores2='DLPRIM_GEMM_VECTOR_STORES2',shapes='DLPRIM_GEMM_SHAPE_TUNING',exact80='DLPRIM_GEMM_EXACT80')
VALUES=dict(narrow32=('DLPRIM_GEMM_NARROW_N','32'),narrow64=('DLPRIM_GEMM_NARROW_N','64'))
def apply(spec):
 names=[] if not spec or spec=='baseline' else spec.split(',')
 if any(name not in FLAGS and name not in VALUES for name in names):raise ValueError('Unknown GEMM experiment: '+spec)
 if 'narrow32' in names and 'narrow64' in names:raise ValueError('Select one narrow GEMM tile')
 for flag in FLAGS.values():os.environ.pop(flag,None)
 for flag,value in VALUES.values():os.environ.pop(flag,None)
 for name in names:
  flag,value=VALUES[name] if name in VALUES else (FLAGS[name],'1')
  os.environ[flag]=value
 return names
