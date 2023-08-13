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


template <typename scalar_t>
__global__ void kernel_forward(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> x,
                           torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> d,
                           torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output,
                           ) {
  // batch index
  const auto bsz_idx = blockIdx.y;
  // channel index
  const auto c = blockIdx.x * blockDim.x + threadIdx.x;
  // redundant threads
  if (c >= d.size(2) {
    return;
    }

  scalar_t cum = 0;
  scalar_t alpha = d[c]
  for (int pos = 0; pos < seq_len; pos++) {
    cum = alpha * cum + x[bsz_idx][pos][c];
    y[bsz_idx][pos][c] = cum;
  }
}


torch::Tensor cuda_forward(
    const torch::Tensor x,
    const torch::Tensor d
  ){
  const auto bsz = x.size(0);
//   const auto seq_len = x.size(1);
  const auto c = x.size(2);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto output = torch::zeros_like(x);

//   dim3 numBlocks( min(bsz, 256) );
//   dim3 threadsPerBlock( (bsz * c + numBlocks.x - 1 ) / numBlocks.x );

  const int threads = 1024;
  const dim3 blocks((c + threads - 1) / threads, batch_size);


  AT_DISPATCH_ALL_TYPES(
      x.type(),
      "add_cuda",
      ([&] {
              kernel_forward<scalar_t><<<numBlocks, threadsPerBlock>>>(
                  x.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                  output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
                  );
          }
      )
  );
  return output;
}



template <typename scalar_t>
__global__ void kernel_backward(
                           torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_d,
                           torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_x,
                           torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_y,
                           const torch::Tensor y,
                           const torch::Tensor x,
                           const torch::Tensor d,
                           ) {
  // batch index
  const auto bsz_idx = blockIdx.y;
  // channel index
  const auto c = blockIdx.x * blockDim.x + threadIdx.x;
  // redundant threads
  if (c >= d.size(2) {
    return;
    }

  scalar_t cum = 0;
  scalar_t alpha = d[c];
  for (int pos = output.size(1)-1; pos > 0; pos--) {
    auto gd = grad_y[bsz_idx][pos][c]
    cum = gd + cum * alpha;
    grad_x[bsz_idx][pos][c] = cum;
    if (pos > 0) {
      // tood 似乎可以用torch.dot? 需要验证不同bsz_idx都传导回了grad_d而没有冲突. 有冲突的话需要复制bszxseq_len份grad然后在外面sum，但这和要求传输[bsz,seq_len, c]的d等价
      grad_d[c] += gd * y[bsz_idx][pos-1][c];
    }
  }
}


std::vector<torch::Tensor> cuda_backward(
    const torch::Tensor grad_y,
    const torch::Tensor y,
    const torch::Tensor x,
    const torch::Tensor d,

  ){
  const auto bsz = x.size(0);
//   const auto seq_len = x.size(1);
  const auto c = x.size(2);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto grad_d = torch::zeros_like(d);
  auto grad_x = torch::zeros_like(x);

//   dim3 numBlocks( min(bsz, 256) );
//   dim3 threadsPerBlock( (bsz * c + numBlocks.x - 1 ) / numBlocks.x );

  const int threads = 1024;
  const dim3 blocks((c + threads - 1) / threads, batch_size);


  AT_DISPATCH_ALL_TYPES(
      x.type(),
      "add_cuda",
      ([&] {
              kernel_backward<scalar_t><<<numBlocks, threadsPerBlock>>>(
                  grad_d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                  grad_x.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  grad_y.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  y.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  x.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                  d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                  );
          }
      )
  );
  return grad_d, grad_x;
}
