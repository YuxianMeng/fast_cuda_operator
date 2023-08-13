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
torch::Tensor add_cuda_forward(
    torch::Tensor x,
    torch::Tensor y
);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Input check and call to CUDA OP
// Backward method not required
torch::Tensor add_forward(
    torch::Tensor x,
    torch::Tensor y
) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  return add_cuda_forward(x, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_forward,
        "example custom add forward (CUDA)");
}
