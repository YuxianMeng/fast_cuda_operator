/*
Kernel implementation for blocking repeated n-grams.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

// Ban repeated ngrams of length = 'no_repeat_ngram_size'
template <typename scalar_t>
__global__ void banRepeatedTokens_kernel(const scalar_t* __restrict__ orig_tokens,
                                         const scalar_t* __restrict__ prev_tokens,
                                         const bool* __restrict__ mask,
                                         scalar_t* __restrict__ output,
                                         const size_t src_len,
                                         const size_t step,
                                         const int vocab_size,
                                         const int ngram
                                         ) {
  auto row = blockIdx.x;
  auto col = threadIdx.x;
  auto tgt_start = (row + 1) * step - ngram + 1;

//   我们存储每句话的tokens到thread-shared memory中，因为这个数据是每个thread共用的，不用每次从global memory取
//   shared[: src_len] 存储当前batch_idx的orig_tokens
//   shared[src_len: src_len + (n-1)]存储当前batch_idx的prev_tokens[-(n-1):]
  auto shared_orig_start = row * src_len;
  extern __shared__ long tokens_shm[];
  if (col == blockDim.x - 1) {
    for (int i=0; i<src_len; i++){
	  tokens_shm[i] = orig_tokens[shared_orig_start + i];
    }
    for (int i=0; i<ngram-1; i++){
      tokens_shm[src_len+i] = prev_tokens[tgt_start + i];
    }
  }
  __syncthreads();

  for (int k = 0; k < ngram - 1; k++) {
    if (tokens_shm[col + k] != tokens_shm[src_len + k]
//         or mask[row * src_len + col + k]
        ) {
      return;
    }
  }
  // reach here means ban
  auto origin_token_mask = mask[row * src_len + col + ngram - 1];
  // 如果待origin_token对应的mask为True，也即已经被保护了，可以返回
  if (origin_token_mask){
    return;
  }

  auto token_to_be_banned = tokens_shm[col + ngram - 1];
  output[row * src_len + col] = token_to_be_banned;

}

// Allocate blocks and threads based on
// batch size and sequence length and launch
// kernel
torch::Tensor src_ngram_repeat_cuda_forward(
    const torch::Tensor orig_tokens,
    const torch::Tensor prev_tokens,
    const torch::Tensor mask,
    const int vocab_size,
    const int ngram,
    const int pad
  ){
  const at::cuda::OptionalCUDAGuard device_guard(device_of(orig_tokens));
  const size_t step = prev_tokens.size(1);
  const size_t bsz = prev_tokens.size(0);
  const size_t src_len = orig_tokens.size(1);
  torch::Tensor output = torch::full_like(orig_tokens, pad); // 因为ngram repeat的数量不会比原文更长
  const size_t threads = src_len - ngram + 1;
  if (threads <= 0) return output;
  if (step+1 < ngram) return output;
  const size_t blocks = bsz;
  const int shared_mem_size = (src_len+ngram) * sizeof(long);  // TODO 应该取决于orig_tokens.scalar_type()
  // Launching N blocks where N is number of samples in a batch
  // Launching T threads where T is number of previous ngrams in a sample
  // Allocating shared mem per block for fastser access of input tokens since
  // each token will be accessed N times to compare with current Ngram where
  // N is Ngram size.
  AT_DISPATCH_ALL_TYPES(orig_tokens.type(), "src_ngram_block_cuda", ([&] {
  banRepeatedTokens_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
      orig_tokens.data<scalar_t>(),
      prev_tokens.data<scalar_t>(),
      mask.data<bool>(),
      output.data<scalar_t>(),
      src_len,
      step,
      vocab_size,
      ngram
      );
  }));
  return output;
}
