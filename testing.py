import pyaudio
import speech_recognition as sr


recognizer = sr.Recognizer()


def get_audio_input():
    t = 1
    with sr.Microphone() as source:
        while t != 0:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                t = 0
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                print("Could not understand audio")
            s = "Recorded Prompt is : "
    return s + text


print(get_audio_input())
