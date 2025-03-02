import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
import torch

class SpeechDataset(Dataset):

  def __init__(self, file_path, label, processor):

    self.file_path = file_path
    self.label = label
    self.processor = processor
    self.sample_rate = 16000



  def __len__(self):
    return len(self.label)

  def __getitem__(self, idx):

    file_path = self.file_path[idx]
    label = self.label[idx]

    waveform, sr = torchaudio.load(file_path)

    if sr != self.sample_rate:
      resampler = transforms.Resample(sr, self.sample_rate)
      waveform = resampler(waveform)

    if waveform.shape[0] > 1:
      waveform = waveform.mean(dim = 0)

    input_values = self.processor(waveform.numpy(), return_tensors='pt', sample_rate=self.sample_rate, padding=True).input_values

    return input_values.squeeze(0), torch.tensor(label)