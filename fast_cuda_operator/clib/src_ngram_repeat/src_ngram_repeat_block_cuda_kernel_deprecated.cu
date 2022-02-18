/*
Kernel implementation for blocking repeated n-grams.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>
#include <vector>

// Ban repeated ngrams of length = 'no_repeat_ngram_size'
template <typename scalar_t>
__global__ void banRepeatedTokens_kernel(const long* __restrict__ orig_tokens,
                                         const long* __restrict__ prev_tokens,
                                         const bool* __restrict__ mask,
                                         int8_t* __restrict__ output,
                                         const size_t src_len,
                                         const size_t step,
                                         const int vocab_size,
                                         const int ngram) {
  auto row = blockIdx.x;
  auto col = threadIdx.x;
//   auto orig_start = row * src_len + col;
  auto tgt_start = (row + 1) * step - ngram + 1;

  // 我们存储每句话的tokens到thread-shared memory中，因为这个数据是每个thread共用的，不用每次从global memory取
  // shared[: src_len] 存储当前batch_idx的orig_tokens
  // shared[src_len: src_len + (n-1)]存储当前batch_idx的prev_tokens[-(n-1):]
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
//     if (orig_tokens[orig_start + k] != prev_tokens[tgt_start + k]) {
    if (tokens_shm[col + k] != tokens_shm[src_len + k]) {
      return;
    }
  }
  // reach here means ban
//   auto token_to_be_banned = orig_tokens[orig_start + ngram - 1];
  auto token_to_be_banned = tokens_shm[col + ngram - 1];
  output[row * vocab_size + token_to_be_banned] = 1;
}

// Allocate blocks and threads based on
// batch size and sequence length and launch
// kernel
torch::Tensor src_ngram_repeat_cuda_forward(
    const torch::Tensor orig_tokens,
    const torch::Tensor prev_tokens,
    const torch::Tensor mask,
    const int vocab_size,
    const int ngram
  ){
  const size_t step = prev_tokens.size(1);
  const size_t bsz = prev_tokens.size(0);
  const size_t src_len = orig_tokens.size(1);
  auto options = torch::TensorOptions().dtype(torch::kInt8)
                                       .layout(torch::kStrided)
                                       .device(torch::kCUDA)
                                       .requires_grad(false);
  torch::Tensor output = torch::zeros({int(bsz), vocab_size}, options);
  const size_t threads = src_len - ngram + 1;
  if (threads <= 0) return output;
  const size_t blocks = bsz;
  const int shared_mem_size = (src_len+step) * sizeof(long);
  // Launching N blocks where N is number of samples in a batch
  // Launching T threads where T is number of previous ngrams in a sample
  // Allocating shared mem per block for fastser access of input tokens since
  // each token will be accessed N times to compare with current Ngram where
  // N is Ngram size.
  AT_DISPATCH_ALL_TYPES(orig_tokens.scalar_type(), "ngram_block_cuda", ([&] {
  banRepeatedTokens_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
      orig_tokens.data<long>(),
      prev_tokens.data<long>(),
      mask.data<bool>(),
      output.data<int8_t>(),
      src_len,
      step,
      vocab_size,
      ngram
      );
  }));
  return output;
}
