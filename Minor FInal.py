# run deoendencies first using : !pip install --upgrade diffusers transformers scipy
import pyaudio
import speech_recognition as sr
import torch
from diffusers import StableDiffusionPipeline
import logging
import time
import datetime
import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(filename="output_analysis.log", level=logging.INFO)


# Set up speech recognizer
recognizer = sr.Recognizer()


# Function to get audio input
def get_audio_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError as e:
            logging.error(f"Error during audio processing: {e}")
            print("Could not understand audio")
    return text


# Function for result analysis
def analyze_results(prompt, image):
    # Log the time taken for image generation
    logging.info(f"Image generation time: {end_time - start_time} seconds")

    # Display the generated image
    plt.imshow(image)
    plt.title("Generated Image")
    plt.show()

    # Save the generated image
    image.save("GeneratedImage.png")


# Set up diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)


# Record start time
start_time = time.time()

# Get voice prompt
prompt = get_audio_input()

# Record end time
end_time = time.time()

# Perform image generation
image = pipe(prompt).images[0]

# Analyze results
analyze_results(prompt, image)
