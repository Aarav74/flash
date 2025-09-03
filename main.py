import os
import requests
import webbrowser
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class AIAssistant:
    def __init__(self):
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Female voice
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize Hugging Face pipeline for question answering
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased"
        )
        
        # Initialize Google Generative AI (for more advanced responses)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        else:
            self.gemini_model = None
            
        # YouTube Data API key (optional)
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        
    def speak(self, text):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        
    def listen(self):
        """Listen for user input through microphone"""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            
        try:
            print("Recognizing...")
            query = self.recognizer.recognize_google(audio)
            print(f"User: {query}")
            return query.lower()
        except sr.UnknownValueError:
            self.speak("Sorry, I didn't catch that. Could you please repeat?")
            return None
        except sr.RequestError:
            self.speak("Sorry, there seems to be an issue with the speech service.")
            return None
            
    def answer_question(self, question, context=None):
        """Answer questions using Hugging Face model"""
        if context:
            # Use the QA pipeline if context is provided
            result = self.qa_pipeline(question=question, context=context)
            return result['answer']
        else:
            # Use generative AI for general questions if available
            if self.gemini_model:
                try:
                    response = self.gemini_model.generate_content(question)
                    return response.text
                except Exception as e:
                    return f"I encountered an error: {str(e)}. Please try again later."
            else:
                return "I can answer questions based on provided context, but for general knowledge questions, please set up the Google API key."
                
    def search_youtube(self, query):
        """Search YouTube for the given query"""
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(search_url)
        return f"Here are YouTube search results for {query}"
        
    def search_google(self, query):
        """Search Google for the given query"""
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(search_url)
        return f"Here are Google search results for {query}"
        
    def process_command(self, command):
        """Process user command and execute appropriate action"""
        if command is None:
            return
            
        # Question answering
        if "what is" in command or "who is" in command or "how to" in command:
            # For demonstration, using a simple context
            context = """
            Artificial intelligence is intelligence demonstrated by machines, unlike the natural intelligence 
            displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents: 
            any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
            """
            answer = self.answer_question(command, context)
            self.speak(answer)
            
        # YouTube search
        elif "youtube" in command and "search" in command:
            query = command.replace("youtube", "").replace("search", "").strip()
            if query:
                response = self.search_youtube(query)
                self.speak(response)
            else:
                self.speak("What would you like me to search on YouTube?")
                
        # Google search
        elif "google" in command and "search" in command:
            query = command.replace("google", "").replace("search", "").strip()
            if query:
                response = self.search_google(query)
                self.speak(response)
            else:
                self.speak("What would you like me to search on Google?")
                
        # General conversation
        elif "hello" in command or "hi" in command:
            self.speak("Hello! How can I assist you today?")
        elif "how are you" in command:
            self.speak("I'm functioning well, thank you for asking! How can I help you?")
        elif "bye" in command or "goodbye" in command:
            self.speak("Goodbye! Have a great day!")
            return False  # Signal to exit
            
        # Fallback - try to answer using generative AI
        else:
            answer = self.answer_question(command)
            self.speak(answer)
            
        return True
        
    def run(self):
        """Main loop to run the assistant"""
        self.speak("AI Assistant initialized. How can I help you today?")
        
        running = True
        while running:
            command = self.listen()
            if command:
                running = self.process_command(command)

if __name__ == "__main__":
    assistant = AIAssistant()
    
    # For text-based testing (comment out the run() method and use this instead)
    # while True:
    #     query = input("You: ")
    #     if query.lower() in ['exit', 'quit', 'bye']:
    #         break
    #     assistant.process_command(query)
    
    assistant.run()