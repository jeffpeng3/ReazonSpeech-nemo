import time
from nemo.collections.asr.models import EncDecRNNTBPEModel
import torch
import librosa
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from nemo.utils import logging
logging.setLevel(logging.ERROR)
SAMPLERATE = 16000

@dataclass
class AudioData:
    """Container for audio waveform"""
    waveform: npt.NDArray[np.float32]
    samplerate: int | float


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No GPU available, please use CPU version of ReazonSpeech")

model = EncDecRNNTBPEModel.restore_from(restore_path='models/reazonspeech-nemo-v2.nemo',
                                            map_location=device)

t0 = time.time()
AUDIO = 'audio.m4a'
array, samplerate = librosa.load(AUDIO, sr=None)
audio = AudioData(array, samplerate)

#norm
waveform = audio.waveform
if audio.samplerate != SAMPLERATE:
    waveform = librosa.resample(waveform, orig_sr=audio.samplerate, target_sr=SAMPLERATE)
if len(waveform.shape) > 1:
    waveform = librosa.to_mono(waveform)
audio = AudioData(waveform, SAMPLERATE)

#pad
waveform = np.pad(audio.waveform,
                  pad_width=int(0.5 * audio.samplerate),
                  mode='constant')
audio = AudioData(waveform, audio.samplerate)

waveform_tensor = torch.tensor(audio.waveform, dtype=torch.float32)

res = model.transcribe(
    [waveform_tensor]
)


print(res)
t1 = time.time()
print(f"Transcription time: {t1 - t0} seconds")