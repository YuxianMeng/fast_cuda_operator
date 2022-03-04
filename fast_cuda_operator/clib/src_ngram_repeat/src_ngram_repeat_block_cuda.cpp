/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/

#include <torch/extension.h>
#include <vector>

/*
CPP Binding for CUDA OP
*/

// CUDA forward declarations
torch::Tensor src_ngram_repeat_cuda_forward(
    torch::Tensor orig_tokens,
    torch::Tensor prev_tokens,
    torch::Tensor mask,
    int vocab_size,
    int ngram,
    int pad
);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Input check and call to CUDA OP
// Backward method not required
torch::Tensor src_ngram_repeat_forward(
    torch::Tensor orig_tokens,
    torch::Tensor prev_tokens,
    torch::Tensor mask,
    int vocab_size,
    int ngram,
    int pad
) {
  CHECK_INPUT(orig_tokens);
  CHECK_INPUT(prev_tokens);
  assert(ngram > 0);
  assert(vocab_size > 0);

  return src_ngram_repeat_cuda_forward(orig_tokens, prev_tokens, mask, vocab_size, ngram, pad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &src_ngram_repeat_forward,
        "No Repeat Ngram Block forward (CUDA)");
}
