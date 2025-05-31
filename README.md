# REMUAE
# REM Waste AI Accent Analyzer

This project is a solution to the REM Waste technical challenge: building a tool to analyze a speaker's English accent from a video URL.

## Features

-   Accepts public video URLs (e.g., Loom, direct MP4, YouTube, Vimeo).
-   Extracts audio from the video.
-   Transcribes the audio to confirm English speech.
-   Analyzes the speaker's accent, classifying it among several English variants.
-   Outputs the detected accent, a confidence score for the accent classification, and an overall confidence score for English speaking.
-   Provides a simple web-based user interface (Streamlit).

## Technology Stack

-   *Python:* Core programming language.
-   *Streamlit:* For building the interactive web UI.
-   *yt-dlp:* For robust video download and audio extraction from various sources.
-   *faster-whisper:* An optimized implementation of Whisper for fast and accurate speech-to-text transcription.
-   *transformers:* Leveraging Hugging Face's platform for an advanced, pre-trained audio classification model.
-   *lhotse-speech/wav2vec2-base-ft-cv13-accent-classifier:* The specific pre-trained model used for English accent classification. It supports en-US, en-AU, en-CA, en-IN, en-GB, en-NZ accents.

## Setup and Local Run

1.  *Clone the repository:*
    bash
    git clone <your-repo-link>
    cd <your-repo-name>
    

2.  *Create a virtual environment (recommended):*
    bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    

3.  *Install dependencies:*
    bash
    pip install -r requirements.txt
    
    (Generate requirements.txt by running pip freeze > requirements.txt after successful installation of all libraries).

4.  *Run the Streamlit application:*
    bash
    streamlit run app.py
    
    This will open the application in your web browser, typically at http://localhost:8501.

## Deployment

This application is deployed on Streamlit Cloud (or Hugging Face Spaces). You can access the live demo here:
[Provide your deployment link here]

## How it Works

1.  *URL Input:* The user provides a public video URL.
2.  *Audio Extraction:* yt-dlp downloads the video and extracts its audio content, saving it as a temporary WAV file.
3.  *Transcription (English Confirmation):* faster-whisper transcribes the extracted audio. A successful transcription indicates that the speaker is likely an English speaker.
4.  *Accent Classification:* A pre-trained Wav2Vec2 model (specifically fine-tuned for accent detection) analyzes the audio's acoustic features to classify the English accent (e.g., American, British, Australian).
5.  *Confidence Scoring:* The model outputs probabilities for each accent class. The highest probability is used as the confidence score for the detected accent.
6.  *Output:* The classified accent, confidence scores, and a brief explanation are displayed in the Streamlit UI. Temporary audio files are cleaned up.

## Limitations

-   The accent classification model is trained on specific English accents (US, AU, CA, IN, GB, NZ). Other English accents (e.g., South African, Irish) may be mapped to the closest available category or result in lower confidence.
-   The performance depends on the quality of the audio in the video.
-   Very long videos might hit resource limits or take a significant amount of time to process, depending on the deployment environment. For this POC, a short video with clear speech is recommended.

Citation Sources
