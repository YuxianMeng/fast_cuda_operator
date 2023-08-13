/*
Kernel implementation for blocking repeated n-grams.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>
#include <vector>
#include <THC/THC.h>
#include <cstdio>
#include <c10/cuda/CUDAGuard.h>

// Ban repeated ngrams of length = 'no_repeat_ngram_size'
template <typename scalar_t>
__global__ void add_kernel(torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> x,
                           torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y,
                           torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                           seq_len
                           ) {
  auto r = blockIdx.x;
  auto c = threadIdx.x;
  output[r][c] = x[r][c] + y[r][c];

}

// Allocate blocks and threads based on
// batch size and sequence length and launch kernel
torch::Tensor add_cuda_forward(
    const torch::Tensor x,
    const torch::Tensor y
  ){
  const auto seq_len = y.size(1);
  const auto bsz = y.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto output = torch::zeros_like(x);

  AT_DISPATCH_ALL_TYPES(
      x.type(),
      "add_cuda",
      ([&] {
              add_kernel<scalar_t><<<bsz, 2>>>(
                  x.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                  y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                  output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
                  );
          }
      )
  );
  return output;
}
