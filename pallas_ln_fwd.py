import os

workspace_dir = os.environ.get('WORKSPACE_DIR', '/home/repo')
os.environ['XLA_FLAGS'] = f' \
    --xla_gpu_graph_level=0 \
    --xla_gpu_enable_cudnn_layer_norm=false \
    --xla_dump_hlo_pass_re=.* \
    --xla_dump_hlo_as_html \
    --xla_dump_to={workspace_dir}/benchmark_layer_norm/pallas_ln_fwd_dump \
'

import sys
from absl import logging
from praxis import pax_fiddle
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pytypes
from praxis import test_utils
from praxis.layers import normalizations

try:
  from jax.experimental.pallas.ops import layer_norm
except ImportError:
  logging.warning('jax_triton not found, please `pip install jax-triton`')
# pylint: enable=g-import-not-at-top

instantiate = base_layer.instantiate
to_np = test_utils.to_np
JTensor = pytypes.JTensor

PARAMS = base_layer.PARAMS

class GpuTritonFusedLayerNorm(normalizations.LayerNorm):

  def __call__(
      self, inputs: JTensor, paddings: JTensor | None = None
  ) -> JTensor:
    """Applies layer norm to inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: unused.

    Returns:
      Output after applying layer normalization, with the same shape as
      'inputs'.
    """
    del paddings  # Unused.
    bwd_pass_impl = os.getenv(
        'pax_fused_layernorm_backward_pass_impl', default='xla'
    )

    def layernorm(x, w, b):
      return layer_norm.layer_norm(x, w, b, backward_pass_impl=bwd_pass_impl)

    return layernorm(inputs, 1 + self.theta.scale, self.theta.bias)


T, S, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

p = pax_fiddle.Config(
    GpuTritonFusedLayerNorm,
    #normalizations.LayerNorm,
    name='pallas_layer_norm',
    reductions_in_fp32=True,
    fprop_dtype=jnp.bfloat16,
    dim=T,
)
ln = instantiate(p)

npy_input = np.random.normal(1.0, 0.5, [B, S, T]).astype('bfloat16')
inputs = jnp.asarray(npy_input)
prng_key = jax.random.PRNGKey(seed=123456)
prng_key, init_key = jax.random.split(prng_key)

initial_vars = ln.init(init_key, inputs)

def _infer(variables, x):
  y = ln.apply(variables, x)
  return y

infer_fn = jax.jit(_infer)

# warmup
outputs = infer_fn(initial_vars, inputs)

repeats = 100
for i in range(repeats):
  outputs = infer_fn(initial_vars, inputs)

np_outputs = to_np(outputs)

print("Mean out:", np_outputs.mean())
print("Var  out:", np.var(np_outputs))


