# ALMTokenizer2: Towards Low bit-rate and Semantic-rich Audio Tokenizer with Flow-based Scalar Diffusion Transformer Decoder

This repository will release the source code and weight for our latest work, ALMTokenizer2. ALMTokenizer2 use the query-based quantization strategy to enhance the semantic information and reconstruction performance.  Furthermore, it introudces a flow-based scalar diffusion transformer decoder to improve 
the reconstruction performance. Compared to ALMTokenizer, the experimental results show that ALMTokenizer2 significantly improves the reconstruction performance, especially for sound and music data.  

## The difference between ALMTokenizer

- Instead of introducing multiple stage training strategy to improve the semantic information, we propose to use multiple semantic experts to extract the semantic features, then using query-based quantization for it.

- We use diffusion loss to optimize the codec (discard the GAN-based training strategy).

- In the released version, we do not apply the GPT loss for the codec, but the training code support to use it.

## Training data

The total training data includes about 10k hours of speech, sound, and music data.

## How to train the model

```bash
bash run.sh
```

## How to infer the model

```bash
huggingface-cli download Dongchao/almtokenizer2 \
  --local-dir ./Dongchao/almtokenizer2 \
  --repo-type model
```

```bash
bash infer.sh
```

## Performance 
### 1. VCTK reconstruction

|         Model        | PESQ |  STOI | MS-STFT loss |
|:--------------------:|:----:|:-----:|:------------:|
| ALMTokenizer (3 RVQ) |  2.0 | 0.81  |     1.78     |
| ALMTokenizer (8 RVQ) | 2.63 |  0.86 |     1.57     |
|   MimiCodec (8RVQ)   |  2.1 |  0.82 |     1.60     |
|     ALMTokeizer2     | 2.99 |  0.86 |     1.44     |

## Plan

Note that we will donot update this repo in recently. Because we have got more advanced codec (ReasoningCodec). If you take care of about the universal semantic codec, please refer to https://github.com/yangdongchao/UniAudio2

## Citations
```bibtex
@inproceedings{yangalmtokenizer,
  title={ALMTokenizer: A Low-bitrate and Semantic-rich Audio Codec Tokenizer for Audio Language Modeling},
  author={Yang, Dongchao and Liu, Songxiang and Guo, Haohan and Zhao, Jiankun and Wang, Yuanyuan and Wang, Helin and Ju, Zeqian and Liu, Xubo and Chen, Xueyuan and Tan, Xu and others},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

## Acknowledgement

Part of the code refers to MuCodec (https://github.com/tencent-ailab/MuCodec). 


