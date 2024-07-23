# M5 & Burn: Rust implementation of M5 model

* __M5:__ The specific architecture is modeled after the M5 network architecture described in [this paper](https://arxiv.org/pdf/1610.00087). 
* __Burn:__ is a [Deep Learning Framework](https://burn.dev/) written in Rust. Among other features, it is able to [import PyTorch models](https://burn.dev/book/import/pytorch-model.html).

The archetype of this repository is sudomonikers [whisper-burn](https://github.com/sudomonikers/whisper-burn).

## M5

* The rationale for selecting this model is that it is discussed in detail on [Speech Command Clasification with torchaudio](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html).
* `pytorch-import/pytorch/m5_model_weights.pt` contains the weights of a pre-trained M5 model. Pre-training was done at in [this Colab notebook](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/c64f4bad00653411821adcb75aea9015/speech_command_classification_with_torchaudio_tutorial.ipynb) during 2 epochs.

