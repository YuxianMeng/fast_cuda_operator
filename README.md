# fast-cuda-operator

Fast C++/CUDA implementations of NLP operations/

## Install
`python setup.py install`

### TODO
1. 用香侬平台发布
    * 目测需要每个cuda版本发布一个？

## Features
* ngram_block: Seq2Seq时不希望decode重复的ngram
    * 目前实现的是cuda版本，待实现cpu版本

## References
* [PyTorch extension tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
* [CUDA easy introdution](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
* [CUDA shared memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
* Other cuda github repositories:
    * [fairseq](https://github.com/pytorch/fairseq)
    * [fastseq](https://github.com/microsoft/fastseq)
