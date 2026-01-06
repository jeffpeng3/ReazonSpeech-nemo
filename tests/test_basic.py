import pytest
import torch
import numpy as np

def test_transcribe_pipeline():
    """
    Basic test to verify the pipeline runs.
    Requires:
    1. reazonspeech-nemo-asr installed (pip install -e .)
    2. GPU available (or code modified to support CPU)
    3. Model file present in models/reazonspeech-nemo-v2.nemo
    """
    try:
        from reazonspeech.nemo.asr.transcribe import load_model, transcribe
        from reazonspeech.nemo.asr.audio import audio_from_numpy
        from reazonspeech.nemo.asr.interface import TranscribeConfig
    except ImportError:
        pytest.skip("reazonspeech package not installed or dependencies missing")

    if not torch.cuda.is_available():
        # The library explicitly requires GPU.
        # We catch the RuntimeError to ensure it behaves as expected, 
        # OR we skip if we can't test functionality.
        # For now, let's skip.
        pytest.skip("Skipping test because CUDA is not available")

    try:
        model = load_model()
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

    # Generate dummy audio
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), False)
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio = audio_from_numpy(audio_data, sr)

    config = TranscribeConfig(verbose=True)
    
    try:
        result = transcribe(model, audio, config)
    except Exception as e:
        pytest.fail(f"Transcription failed: {e}")

    # We expect some result, even if empty string for sine wave
    assert result is not None
    # Just check it doesn't crash
