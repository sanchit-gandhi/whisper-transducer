# Whisper Transducer

A repository for benchmarking the [Whisper Model](https://arxiv.org/abs/2212.04356) with an [RNN-T decoder](https://arxiv.org/pdf/2002.02562.pdf).

Code is based on the [Hugging Face Whisper implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) augmented with the Emformer RNN-T joint/prediction networks.