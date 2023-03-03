from modeling_whisper_transducer import WhisperForRNNT
from configuration_whisper import WhisperConfig
import torch


# initialise the model with the 'tiny' config
config = WhisperConfig.from_pretrained("openai/whisper-tiny.en")

# set the correct RNN-T config params
config.vocab_size = 4096
config.blank_token_id = 4095
config.bos_token_id = 1
config.max_output_length = 25
config.min_output_length = 25

model = WhisperForRNNT(config)

# test with a random log-mel input
inputs = torch.ones(1, 80, 3000)

with torch.no_grad():
    out = model.forward(inputs)
