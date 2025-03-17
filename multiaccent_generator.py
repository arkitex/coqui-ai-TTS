import os

import ffmpeg
import kagglehub
import pandas as pd
import torch
import torchaudio

from TTS import XttsConfig
from TTS.tts.models.xtts import Xtts


def convert_mp3_to_wav(mp3_file_path, wav_output_path):
    stream = ffmpeg.input(mp3_file_path)
    stream = ffmpeg.output(stream, wav_output_path, acodec='pcm_s16le', ac=1, ar='22050')
    ffmpeg.run(stream, overwrite_output=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

# From Mozilla Common Voice, sample a few acccents and convert to wav.
def sample_accents():
    # Download latest version
    common_voice_path = kagglehub.dataset_download("mozillaorg/common-voice")
    common_voice_file = "cv-valid-dev"

    print("Path to Common Voice dataset files:", common_voice_path)

    df = pd.read_csv(common_voice_path + f"/{common_voice_file}.csv", sep=',')
    good_samples = df[(df['up_votes'] >= 2) & (df['down_votes'] == 0) & (df['accent'].notna())]

    # Check how many accents are available and their distribution:
    print(good_samples['accent'].value_counts())

    selected_accents = ['indian', 'australia', 'african']
    accented_samples = good_samples[good_samples['accent'].isin(selected_accents)]
    accented_samples = accented_samples.groupby('accent').sample(n=5, random_state=42)

    print(accented_samples)

    # Set your output directory
    output_dir = common_voice_path + f"/{common_voice_file}/wav_out"
    os.makedirs(output_dir, exist_ok=True)  # ensure the folder exists

    output_paths = {}
    for idx, row in accented_samples.iterrows():
        accent = row['accent']
        filename = row['filename']

        os.makedirs(output_dir + f"/{accent}/{common_voice_file}", exist_ok=True)  # ensure the folder exists
        input_path = common_voice_path + f"/{common_voice_file}/{filename}"
        output_path = output_dir + f"/{accent}/{filename[:-4]}.wav"

        convert_mp3_to_wav(input_path, output_path)
        accent_list = output_paths.setdefault(accent, [])
        accent_list.append(output_path)
    return output_paths


def synthesize(output_paths, sentences):
    config = XttsConfig()
    config.load_json("config/xtts/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="config/xtts/", use_deepspeed=False)
    model.cpu()

    for accent in output_paths:
        paths = output_paths[accent]
        for idx, path in enumerate(paths):
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[path])
            # gpt_cond_latent, speaker_embedding = model.speaker_manager.speakers[speaker].values()

            print(f"Inference...")

            for sentence_idx, sentence in enumerate(sentences):
                out = model.inference(
                    sentence,
                    "en",
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=0.2,  # Add custom parameters here
                )
                os.makedirs(f"out/{accent}", exist_ok=True)  # ensure the folder exis
                torchaudio.save(f"out/{accent}/{idx}_{sentence_idx}_xttsgen.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

output_paths = sample_accents()
print(output_paths)

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2023, Maria traveled from Berlin to New York City.",
    "The chef quickly judged the exquisite flavor of the sizzling dishes.",
    "Christopher regularly visits Melbourne every Thursday.",
    "The sheep grazing quietly judged the exquisite flavor of the sizzling dishes."
]

# Now, for each speaker embedding, generate 5 different sentences, which we'll use for training.
synthesize(output_paths, sentences)

