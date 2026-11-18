
# ISL to Audio – Indian Sign Language Translator

An AI-powered application that recognizes Indian Sign Language (ISL) alphabet hand gestures through a webcam and converts them into spoken audio. The project aims to make communication easier between ISL users and people who do not understand sign language.

## Problem Statement

Many deaf and hard-of-hearing individuals use Indian Sign Language (ISL) as their primary means of communication. However, the lack of ISL awareness among the general public creates communication barriers in everyday situations.

This project aims to bridge that gap by developing a computer-vision-based system that recognizes ISL hand gestures and converts them into audible speech.

## Current Status (TRL-2)

- Successfully collected a custom dataset for all 26 English alphabets using ISL hand signs.
- Developed and trained a Machine Learning model for A–Z alphabet recognition.
- Achieved 99.75% accuracy on the current held-out recording test.
- Integrated real-time webcam-based hand gesture recognition.
- Implemented text-to-speech conversion using pyttsx3.
- Successfully implemented continuous letter recognition and automatic word formation.
- Integrated pause-based word completion and automatic speech output.

Note: The reported accuracy is based on our current dataset and testing conditions. Further testing with different users and environments is required.

## How It Works

1. **Image Acquisition:** A webcam captures the user's hand gestures in real time.
2. **Hand Detection:** MediaPipe detects the hand and extracts 21 hand landmarks.
3. **Feature Extraction:** Landmark coordinates are normalized and converted into numerical features.
4. **Alphabet Classification:** A Random Forest Machine Learning model predicts the corresponding alphabet.
5. **Word Formation:** Recognized letters are automatically combined into words during continuous signing.
6. **Speech Generation:** The completed word is converted into spoken audio using pyttsx3.

## Tech Stack

- **Programming Language:** Python 3.12
- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Computer Vision:** OpenCV
- **Hand Tracking:** MediaPipe
- **Numerical Processing:** NumPy
- **Dataset Management:** Pandas
- **Model Storage:** Joblib
- **Text-to-Speech:** pyttsx3
- **Hardware:** Laptop/Desktop webcam

## Project Structure

ISL_Translator/
│
├── dataset/
│   └── alphabet/
│       └── alphabet_landmarks.csv
│
├── models/
│   └── alphabet_model.joblib
│
├── collect_alphabet.py
├── check_dataset.py
├── train_alphabet.py
├── live_alphabet.py
├── live_alphabet_v2.py
│
└── README.md

## How to Run

### 1. Clone the Repository

git clone <your-github-repository-link>

cd ISL_Translator

### 2. Create a Virtual Environment

python -m venv venv

### 3. Activate the Environment (Windows)

venv\Scripts\activate

### 4. Install Dependencies

pip install opencv-python mediapipe<0.10.14 scikit-learn pyttsx3 pandas numpy joblib

### 5. Run the Application

python live_alphabet_v2.py

Note: The trained model file must be available inside the models directory before running the application. A compatible webcam is also required.

## Roadmap

- [x] Collect a custom dataset for ISL alphabets (A–Z).
- [x] Train the alphabet recognition model.
- [x] Implement real-time hand gesture recognition.
- [x] Integrate text-to-speech conversion.
- [x] Implement continuous letter recognition.
- [x] Develop automatic word formation.
- [x] Implement pause-based word completion and speech output.
- [ ] Improve recognition reliability across different users.
- [ ] Expand recognition to common ISL words and phrases.
- [ ] Develop sentence-level translation.
- [ ] Prepare project demonstration video and technical report.
- [ ] Conduct testing with ISL users.

## Future Scope

- Recognition of dynamic ISL gestures and commonly used words.
- Sentence-level translation with ISL grammar considerations.
- Improved recognition across different lighting conditions and backgrounds.
- Development of a portable sign-language translation system.
- Exploration of wearable technology for real-time communication.

## Author

**HARIHARAN B**  
Mechanical Engineering  
KPR Institute of Engineering and Technology  
Coimbatore, Tamil Nadu, India
