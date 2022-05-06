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
__global__ void banRepeatedTokens_kernel(const long* __restrict__ tokens,
                                         scalar_t* __restrict__ lprobs,
                                         const size_t max_predict_len,
                                         const size_t vocab_size,
                                         const int no_repeat_ngram_size) {
  auto row = blockIdx.x;
  auto col = threadIdx.x;
  auto start = row * (max_predict_len) + col;
  // Each thread compares ngram starting from
  // thread index with final ngram starting from
  // step - no_repeat_ngram_size +2
  auto check_start_pos = blockDim.x;
  auto lprob_start = row * vocab_size;
  // 我们存储每句话的tokens到thread-shared memory中，因为这个数据是每个thread共用的，不用每次从global memory取
  extern __shared__ long tokens_shm[];
  tokens_shm[col] = tokens[start];
  if (col == blockDim.x - 1) {
     for (int i=1; i<no_repeat_ngram_size; i++){
	if (col+i < max_predict_len){
          tokens_shm[col + i] = tokens[start + i];
	}
    }
  }
  __syncthreads();

  for (int k = 0; k < no_repeat_ngram_size - 1; k++) {
    if (tokens_shm[col + k] != tokens_shm[check_start_pos + k]) {
      return;
    }
  }
  // reach here means ban
  auto token_to_be_banned = tokens_shm[col + no_repeat_ngram_size - 1];
  lprobs[lprob_start + token_to_be_banned] = -INFINITY;
}

// Allocate blocks and threads based on
// batch size and sequence length and launch
// kernel
torch::Tensor ngram_repeat_block_cuda_forward(const torch::Tensor tokens,
                                              torch::Tensor lprobs,
                                              const int bsz,
                                              const int step,
                                              const int beam_size,
                                              const int no_repeat_ngram_size) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(tokens));
  int threads = step - no_repeat_ngram_size + 2;
  if (threads <= 0) return lprobs;
  const size_t max_predict_len = tokens.size(1);
  const size_t vocab_size = lprobs.size(1);
  const size_t blocks = bsz * beam_size;
  const int shared_mem_size = (step + 1) * sizeof(long);

  // Launching N blocks where N is number of samples in a batch (beams*bsz)
  // Launching T threads where T is number of previous ngrams in a sample
  // Allocating shared mem per block for fastser access of input tokens since
  // each token will be accessed N times to compare with current Ngram where
  // N is Ngram size.
  AT_DISPATCH_ALL_TYPES(lprobs.scalar_type(), "ngram_block_cuda", ([&] {
  banRepeatedTokens_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
      tokens.data<long>(),
      lprobs.data<scalar_t>(),
      max_predict_len,
      vocab_size,
      no_repeat_ngram_size);
  }));
  return lprobs;
}
