from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path

# Load ReazonSpeech model from Hugging Face
model = load_model()

# Read a local audio file
audio = audio_from_path("audio.m4a")

# Recognize speech
ret = transcribe(model, audio)

print(ret)