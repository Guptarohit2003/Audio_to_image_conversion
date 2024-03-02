'''import tkinter as tk
import customtkinter as ctk 

from PIL import ImageTk
from auth_token import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(master=app,height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app,height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

def generate(): 
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()'''
'''from PIL import Image
import transformers
import torch
from torch import autocast

import speech_recognition as sr

from diffusers import StableDiffusionPipeline 
auth_token = auth_token = "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"

modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

def get_voice_input():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for voice input...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Voice input recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        return ""
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition")
        return ""

def generate(prompt_text): 
    with autocast(device): 
        image = pipe(prompt_text).images[0]
    
    return image

prompt_text = get_voice_input()  # Get voice input

if prompt_text:
    generated_image = generate(prompt_text)
    generated_image.save('generatedimage.png')

    # Display the generated image
    generated_image.show()
else:
    print("No input received from voice command.")'''

import tkinter as tk
import pyaudio
import speech_recognition as sr
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image,ImageTk

# Initialize the speech recognizer
recognizer = sr.Recognizer()

#Define the model and device
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Function to handle audio input
def get_audio_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            entry.insert(tk.END, text)
        except sr.UnknownValueError:
            print("Could not understand audio")

# Function to generate image
def generate_image():
    prompt = entry.get()
    image = pipe(prompt).images[0]
    image.save("generated_image.png")
    
    # Display the generated image
    generated_image = Image.open("generated_image.png")
    generated_image.thumbnail((300, 300))
    generated_image = ImageTk.PhotoImage(generated_image)

    image_label = tk.Label(root, image=generated_image)
    image_label.image = generated_image
    image_label.pack()


# Create the GUI window
root = tk.Tk()
root.title("Audio to Image Interface")

# Create and place widgets
label = tk.Label(root, text="Enter text or speak:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

audio_button = tk.Button(root, text="Get Audio Input", command=get_audio_input)
audio_button.pack()

generate_button = tk.Button(root, text="Generate Image", command=generate_image)
generate_button.pack()

root.mainloop()

