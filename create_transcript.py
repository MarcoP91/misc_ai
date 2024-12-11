import subprocess
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys
import os
torch_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
import warnings
warnings.filterwarnings("ignore")
device = "mps"

VIDEO_URL = "https://www.youtube.com/watch?v=x7bWlT2q8vs&ab_channel=JREClips"
output_template = "./downloaded_audio/%(title)s.%(ext)s"
DLP_COMMAND = F'yt-dlp "{VIDEO_URL}" --extract-audio --audio-format mp3 --output "{output_template}"'
TRANSCRIPT_SUBFOLDER = "misc"

# Script that downloads youtube audio, uses llm whisper to translate it and store it. In order to be uploaded to NotebookLM


def save_transcript(video_url, transcript_subfolder):
    print("Video URL:", video_url)
    dlp_command = F'yt-dlp "{video_url}" --extract-audio --audio-format mp3 --output "{output_template}"'
    # Execute the command and capture the output
    print("Downloading audio...")
    result = subprocess.run(dlp_command, shell=True, capture_output=True, text=True)

    # Extract the downloaded file path
    downloaded_file = result.stdout.strip()

    output = downloaded_file.split('\n')[-2].split('[ExtractAudio] Destination: ')[-1]
    file_name = output.split('/')[-1].split('.mp3')[0].replace(' ', '_')

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Transcribing audio...")
    result = pipe(output, return_timestamps=True)

    # save file
    # The string to save
    content = result["text"]

    # delete audio file
    if os.path.exists(output):
        os.remove(output)

    # Specify the file path
    file_path = "/Users/mp/Desktop/transcripts/" + transcript_subfolder + '/' + file_name + ".txt"

    # Create the subfolder if it doesn't exist
    os.makedirs("/Users/mp/Desktop/transcripts/" + transcript_subfolder, exist_ok=True)

    # Open the file in write mode and save the string
    with open(file_path, "w") as file:
        file.write(content)

    print(f"Content saved to {file_path}")


if __name__ == "__main__":
    print(f"len args: {len(sys.argv)}")
    if len(sys.argv) == 2:
        video_url = sys.argv[1]
        transcript_subfolder = TRANSCRIPT_SUBFOLDER
    elif len(sys.argv) == 3:
        video_url = sys.argv[1]
        transcript_subfolder = sys.argv[2]
    
    else:
        print()
        video_url = VIDEO_URL
        transcript_subfolder = TRANSCRIPT_SUBFOLDER

    save_transcript(video_url, transcript_subfolder)
