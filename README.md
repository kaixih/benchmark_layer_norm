# Usage

```bash
bash benchmark.sh # benchmark the XLA layer norm
bash benchmark.sh true  # benchmark the XLA + cudnn layer norm backend
```

# Note

Benchmarking a single layer in XLA presents challenges, as the fusion and
execution methods of operations within the layer are generally unknown to users.
This benchmark script utilizes the nsys profiling tools. The approach involves
calling the target function 100 times and utilizing the profiling tools to
obtain a summary of CUDA calls. Subsequently, we collect the median time for
fusions/kernels whose instances are greater than or equal to 100. These times
are accumulated to determine the layer execution time.
