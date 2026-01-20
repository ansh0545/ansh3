!pip install gradio groq

import os
import gradio as gr
from groq import Groq
from google.colab import userdata

# 1. Setup the Groq Client
# Ensure you have added GROQ_API_KEY to your Colab Secrets
client = Groq(api_key=userdata.get('GROQ_API_KEY'))
client
def chat_with_groq(message, history):
    """
    Handles the conversation logic.
    'history' is passed automatically by Gradio and contains previous turns.
    """
    # Convert Gradio history format to Groq's message format
    messages = [{"role": "system", "content": "You are a helpful and concise AI assistant."}]

    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    # 2. Request Streaming Completion
    # Using Llama 3.3 70B for high intelligence, or 8B for raw speed
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

# 3. Create the Gradio Interface
demo = gr.ChatInterface(
    fn=chat_with_groq,
    title="Groq Lightning Chat",
    description="Experience sub-second responses using Groq LPUs.",
    examples=["Explain quantum computing in one sentence.", "Write a Python script to scrape a website."],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
