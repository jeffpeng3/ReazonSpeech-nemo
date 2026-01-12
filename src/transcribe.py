import torch
from .interface import TranscribeConfig
from .decode import decode_hypothesis, PAD_SECONDS
from .audio import pad_audio, norm_audio

from nemo.collections.asr.models import EncDecRNNTBPEModel

def load_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("No GPU available, please use CPU version of ReazonSpeech")

    logging.setLevel(logging.ERROR)
    return EncDecRNNTBPEModel.restore_from(restore_path='models/reazonspeech-nemo-v2.nemo',
                                            map_location=device)

def transcribe(model, audio, config=None):
    """Inference audio data using NeMo model

    Args:
        model (nemo.collections.asr.models.EncDecRNNTBPEModel): ReazonSpeech model
        audio (AudioData): Audio data to transcribe
        config (TranscribeConfig): Additional settings

    Returns:
        TranscribeResult
    """
    if config is None:
        config = TranscribeConfig()

    audio = pad_audio(norm_audio(audio), PAD_SECONDS)

    waveform_tensor = torch.tensor(audio.waveform, dtype=torch.float32)

    # NeMo's transcribe method may return a tuple or a list depending on version/config
    res = model.transcribe(
        [waveform_tensor],
        batch_size=1,
        return_hypotheses=True,
        verbose=config.verbose
    )

    # Handle potential return signature differences
    if isinstance(res, tuple):
        hyp = res[0]
    else:
        hyp = res

    # Get the first hypothesis (since we processed batch_size=1)
    hyp = hyp[0]

    ret = decode_hypothesis(model, hyp)

    if config.raw_hypothesis:
        ret.hypothesis = hyp

    return ret
