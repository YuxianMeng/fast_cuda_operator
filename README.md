# fast-cuda-operator

Fast C++/CUDA implementations of NLP operations/

## Install
`python setup.py install`

### TODO
1. 用香侬平台发布，需要做的事情包括
    * 让gitlab的runner支持cuda编译
    * 每个cuda版本build一个wheel

## Features
* add_example: 一个naive地利用cuda做矩阵加法的示例代码
* ngram_block: Seq2Seq时不希望decode重复的ngram
    * 目前实现的是cuda版本，待实现cpu版本
* src_ngram_repeat: Seq2Seq时不希望decode出的tgt和src有重复的ngram，有则对logits给以一定的惩罚。
## References
* [PyTorch extension tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
* [CUDA easy introdution](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
* [CUDA shared memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
* Other cuda github repositories:
    * [fairseq](https://github.com/pytorch/fairseq)
    * [fastseq](https://github.com/microsoft/fastseq)
