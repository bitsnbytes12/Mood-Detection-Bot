import requests
import random

# Local fallbacks
jokes = [
    "Why don't scientists trust atoms? Because they make up everything! 😂",
    "What do you call fake spaghetti? An impasta! 🍝",
    "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾"
]

quotes = [
    "The only way to do great work is to love what you do. — Steve Jobs",
    "Believe you can and you're halfway there. — Theodore Roosevelt",
    "The future belongs to those who believe in the beauty of their dreams. — Eleanor Roosevelt"
]

def get_joke():
    try:
        url = "https://v2.jokeapi.dev/joke/Any?type=single"
        response = requests.get(url, timeout=5).json()
        return response.get("joke", random.choice(jokes))
    except:
        return random.choice(jokes)

def get_quote():
    try:
        url = "https://api.quotable.io/random"
        response = requests.get(url, timeout=5).json()
        return f'{response["content"]} — {response["author"]}'
    except:
        return random.choice(quotes)
