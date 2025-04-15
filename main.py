import torch
from TTS.api import TTS

# def download_model():
#     # Print available TTS models
#     tts_manager = TTS().list_models()
#     all_models = tts_manager.list_models()
#     print("TTS models:\n", all_models, "\n", sep="")

#     # Prompt model selection
#     model = input("Enter model:\n")
#     # for example, tts_models/multilingual/multi-dataset/xtts_v2


def test_model(model_path, config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize TTS with your model files
    tts = TTS(model_path=model_path, config_path=config_path, progress_bar=True).to(
        device
    )

    tts.tts_to_file(
        "This is a voice cloning test",
        speaker_wav="train-audio.wav",
        language="en",
        file_path="output.wav",
    )

    tts.tts_to_file(
        text="Hello world!",
        speaker="train-audio.wav",
        language="english",
        file_path="output.wav",
    )


model_path = r"C:\Users\User\Documents\Coqui\TTS\res\model_file.pth"
config_path = r"C:\Users\User\Documents\Coqui\TTS\res\config.json"
# model_name = "tts_models/en/blizzard2013/capacitron-t2-c50"

test_model(model_path, config_path)
