#!/usr/bin/env python
# coding: utf-8

# In[4]:


import subprocess
import sys

# Install NLTK if not already installed
def install_nltk():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])

try:
    import nltk
except ImportError:
    print("NLTK not found. Installing...")
    install_nltk()

# Download NLTK data files
nltk.download('vader_lexicon')


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        return "that's cute"
    elif scores['compound'] <= -0.05:
        return "that's awful"
    else:
        return "okay..!"

while True:
    user_input = input("Enter your text (type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        break
    
    sentiment_response = sentiment_analysis(user_input)
    print(sentiment_response)


# In[ ]:


import tkinter as tk
from tkinter import ttk, messagebox
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        return "that's cute"
    elif scores['compound'] <= -0.05:
        return "that's awful"
    else:
        return "okay..!"

def analyze_sentiment():
    user_input = text_entry.get("1.0", "end-1c")
    
    if user_input.strip():  # Check if input is not empty
        sentiment_response = sentiment_analysis(user_input)
        messagebox.showinfo("Sentiment Analysis Result", sentiment_response)
    else:
        messagebox.showwarning("Warning", "Please enter some text.")

# Create the main application window
root = tk.Tk()
root.title("NatBot")

# Create GUI components
label = ttk.Label(root, text="Enter your text:")
label.pack(pady=10)

text_entry = tk.Text(root, height=5, width=50)
text_entry.pack(pady=10)

analyze_button = ttk.Button(root, text="Chat with Nat", command=analyze_sentiment)
analyze_button.pack(pady=10)

# Run the main event loop
root.mainloop()


# In[ ]:




