import time
from nemo.collections.asr.models import EncDecRNNTBPEModel
import torch
import librosa
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from nemo.utils import logging
from tempfile import NamedTemporaryFile
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

model: EncDecRNNTBPEModel = EncDecRNNTBPEModel.restore_from(restore_path='models/reazonspeech-nemo-v2.nemo',
                                            map_location=device) # type: ignore

AUDIO = 'audio.m4a'
def transcribe(audio_file) -> str:
    if not isinstance(audio_file, str):
        with NamedTemporaryFile(suffix=".m4a") as tmp:
            tmp.write(audio_file)
            tmp.flush()
            array, samplerate = librosa.load(tmp.name, sr=None)
    else:
        array, samplerate = librosa.load(audio_file, sr=None)
    audio = AudioData(array, samplerate)

    #norm
    waveform = audio.waveform
    if audio.samplerate != SAMPLERATE:
        waveform = librosa.resample(waveform, orig_sr=audio.samplerate, target_sr=SAMPLERATE)
    audio = AudioData(waveform, SAMPLERATE)

    #pad
    waveform = np.pad(audio.waveform,
                    pad_width=int(0.5 * audio.samplerate),
                    mode='constant')
    audio = AudioData(waveform, audio.samplerate)

    #mono
    if len(audio.waveform.shape) > 1:
        waveform = librosa.to_mono(audio.waveform)
        audio = AudioData(waveform, audio.samplerate)

    waveform_tensor = torch.tensor(audio.waveform, dtype=torch.float32)

    res = model.transcribe(
        [waveform_tensor] # type: ignore
    )
    if isinstance(res, list):
        res = res[0]
    return res.text
with open(AUDIO, 'rb') as f:
    d = f.read()

t0 = time.time()
print(transcribe(d))
t1 = time.time()
print(transcribe(d))
t2 = time.time()
print(transcribe(d))
t3 = time.time()
print(f"Transcription time: {t1 - t0} seconds")
print(f"Transcription time: {t2 - t1} seconds")
print(f"Transcription time: {t3 - t2} seconds")