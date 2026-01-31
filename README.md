📘 Mood Detection Bot

Mood Detection Bot is an AI-powered Python application that identifies a user’s emotional state using both image-based facial expression analysis and natural language processing (NLP). Based on detected mood, the bot can generate responses or take intelligent actions to respond appropriately to the user’s emotions.

🧠 Features:-
😊 Multi-Modal Mood Detection

This bot supports two main mood detection modes:

Facial Emotion Recognition: Detect emotions from facial expressions using a trained deep learning model (emotion_model.h5).

Text-Based Sentiment Analysis: Understand mood from user text inputs using an NLP engine.

🤖 Intelligent Bot Responses

Engages in conversation based on detected emotion.

Responds differently when user input reflects different moods (e.g., happy, sad, neutral).

Can be extended to trigger mood-based actions like playing music or suggesting content.

🔧 Modular Codebase

Separate modules for facial emotion detection and NLP mood classification.

Includes scripts to train new models (train.py).

Sample application logic in main1.py and main2.py.

🔍 How It Works:-
👁️ Facial Emotion Detection

Uses a deep learning model (emotion_model.h5) to classify facial expressions.

Typically based on common emotion categories: happy, sad, angry, neutral, etc.

🧠 NLP Sentiment Classification

Processes text input through NLP algorithms (e.g., vectorization, model inference) to estimate text sentiment.

Helps enhance how the bot understands emotional context from user language.

Dataset: FER2013 (Kaggle)
Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

