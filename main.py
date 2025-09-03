import pyttsx3
import speech_recognition as sr
import webbrowser
import datetime
import wikipedia
import pyaudio
from transformers import pipeline
import torch

# Initialize text-to-speech engine
def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(audio) 
    engine.runAndWait()

# Initialize speech-to-text recognizer
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening')
        r.pause_threshold = 0.7
        audio = r.listen(source)
        try:
            print("Recognizing")
            Query = r.recognize_google(audio, language='en-in')
            print("The command is printed =", Query)
        except Exception as e:
            print(e)
            speak("Say that again, sir")
            print("Say that again, sir")
            return "None"
    return Query

# Specific utility functions
def tellDay():
    day = datetime.datetime.today().weekday() + 1
    Day_dict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    if day in Day_dict.keys():
        day_of_the_week = Day_dict[day]
        print(day_of_the_week)
        speak("The day is " + day_of_the_week)

def tellTime():
    time = str(datetime.datetime.now())
    print(time)
    hour = time[11:13]
    min = time[14:16]
    speak("The time is " + hour + " hours and " + min + " minutes") 

def FLASH():
    speak("Hello owner, I am flash. Tell me how may I help you")

# Initialize Hugging Face model
# This model will be downloaded the first time you run the script.
# It is a relatively small conversational model (1.1 billion parameters).
# Note: It may take some time to download and load the model initially.
# It is recommended to use a GPU if available for faster responses.
print("Loading Hugging Face model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
generator = pipeline("text-generation", model=model_name, device=device)
print("Model loaded successfully!")

# Conversation history list for the model's context
conversations = []

def Take_query():
    FLASH()
    while True:
        query = takeCommand().lower()

        # Handle specific commands first
        if "open google" in query:
            speak("Opening google")
            webbrowser.open("https://www.google.com/")
            continue
        
        elif "open youtube" in query:
            speak("Opening youtube ")
            webbrowser.open("www.youtube.com")
            continue
        
        elif "which day it is" in query:
            tellDay()
            continue

        elif "tell me the time" in query:
            tellTime()
            continue
        
        elif "from wikipedia" in query:
            speak("Checking the wikipedia ")
            query_for_wiki = query.replace("from wikipedia", "").strip()
            try:
                result = wikipedia.summary(query_for_wiki, sentences=4)
                speak("According to wikipedia")
                speak(result)
            except wikipedia.exceptions.PageError:
                speak("Sorry, I could not find that on Wikipedia.")
            except wikipedia.exceptions.DisambiguationError as e:
                speak("There are multiple results. Please be more specific.")
                print(e.options)
            continue
        
        elif "tell me your name" in query:
            speak("I am FLASH. Your desktop Assistant")
            continue
        
        elif "bye" in query:
            speak("Bye. Flash is switching off")
            exit()
        
        # If no specific command matches, send the query to the Hugging Face model
        elif query != "none":
            try:
                # Add the user's query to the conversation history
                conversations.append({"role": "user", "content": query})
                
                # Generate a response from the model
                # The `max_new_tokens` argument limits the length of the response
                response = generator(conversations, max_new_tokens=100)
                
                # The response is a list of dictionaries, we need to extract the last one
                reply = response[0]['generated_text'][-1]['content']
                
                speak(reply)
                
                # Add the model's reply to the conversation history
                conversations.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                print(f"Error communicating with Hugging Face model: {e}")
                speak("I'm sorry, I couldn't process that request right now.")

if __name__ == '__main__':
    Take_query()
