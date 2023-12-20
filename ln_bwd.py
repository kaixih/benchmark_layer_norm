import os
workspace_dir = os.environ.get('WORKSPACE_DIR', '/home/tmp')
use_cudnn_ln = os.environ.get('ENABLE_CUDNN_LAYER_NORM', 'false')
os.environ['XLA_FLAGS'] = f' \
    --xla_gpu_graph_level=0 \
    --xla_gpu_enable_cudnn_layer_norm={use_cudnn_ln} \
    --xla_dump_hlo_as_html \
    --xla_dump_to={workspace_dir}/benchmark_layer_norm/ln_bwd_dump \
'

import sys
import time
from jax import jit
from jax import numpy as jnp
from jax import random
from jax import value_and_grad
from praxis import base_layer
from praxis import pax_fiddle as fdl
from praxis.layers.normalizations import LayerNorm

T, S, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

prng_key = random.PRNGKey(seed=123)
prng_key, init_key, random_key = random.split(prng_key, 3)
inputs = random.uniform(random_key, (B, S, T)).astype(jnp.bfloat16)
dy = random.uniform(random_key, (B, S, T)).astype(jnp.bfloat16)

layerN_kwargs = {
    'reductions_in_fp32': True,
    'fprop_dtype': jnp.bfloat16,
    'dim': T,
}
layerN = base_layer.instantiate(
    fdl.Config(LayerNorm, name='layer_norm', **layerN_kwargs),
)

init_vars = layerN.init(init_key, inputs)

def _train(variables, x):
  y = layerN.apply(variables, x)
  loss = y * dy.astype(y.dtype)
  return jnp.mean(loss)

train_fn = jit(value_and_grad(_train, argnums=[0, 1]))

repeats = 100

start = time.time()
for i in range(repeats):
  outputs = train_fn(init_vars, inputs)
outputs[0].block_until_ready()
end = time.time()
elapsed_time = (end - start) / repeats * 1000
# This elapsed_time is inaccurate. Don't use it.
print(f"Mean time: {elapsed_time} ms")


