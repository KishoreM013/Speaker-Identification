from torch import nn
from transformers import Wav2Vec2Model
import torch


class Wav2Vec2SpeechModel(nn.Module):

  def __init__(self):

    super().__init__()
    self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    for param in self.wav2vec2.parameters():
      param.requires_grad = False

    self.fc = nn.Linear(self.wav2vec2.config.hidden_size, 5)

  def forward(self, input_values):

    with torch.no_grad():
      features = self.wav2vec2(input_values).last_hidden_state


    output = features.mean(dim= 1)
    output = self.fc(output)

    return output