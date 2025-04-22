# Intercom-Device
We propose a multimodal intercom system enabling two-way communication between deaf and hearing individuals using voice and gesture recognition, audio-to-text, and a voice-to-sign avatar. A smartwatch displays text with haptic alerts, while an ESP32 ensures real-time processing and Wi-Fi communication.
# Multimodal Intercom System for Deaf and Hearing Communication

## Overview

This project presents an innovative **Multimodal Intercom System** designed to bridge the communication gap between deaf and hearing individuals. It integrates speech recognition, gesture recognition, real-time audio-to-text conversion, and a virtual sign language avatar. The system offers seamless two-way communication using a wearable device (smartwatch) and is powered by the ESP32 microcontroller for efficient wireless connectivity.

## Features

- 🎤 **Speech-to-Text Conversion**: Converts spoken words to text using real-time speech recognition.
- 📱 **Smartwatch Display**: Displays translated text on a wearable device with vibration alerts.
- 🖐️ **Gesture Recognition**: Detects and interprets hand gestures using computer vision (MediaPipe + Deep Learning).
- 🗣️ **Voice-to-Gesture Avatar**: Animates spoken input into sign language via a 3D avatar.
- 📹 **Video Call Support**: Real-time transcription of audio during video communication.
- 📡 **Wi-Fi Enabled Communication**: ESP32 handles wireless data exchange between devices.
- ⚙️ **Python-Powered Backend**: Uses Python libraries for audio processing, gesture recognition, and socket communication.

## Hardware Requirements

- ESP32 Development Board  
- Vibration Motor  
- Microphone Module  
- Camera Module (USB or PiCam)  
- OLED or TFT Display (optional)  
- Power Source (Battery or USB)

## Software Requirements

- Python 3.8+
- [MediaPipe](https://google.github.io/mediapipe/)
- OpenCV
- SpeechRecognition
- PyAudio
- Tkinter (for GUI)
- Socket Programming
- Arduino IDE (for ESP32 firmware)

Use Case
Designed for real-time use in homes, hospitals, schools, and public places to ensure inclusive and accessible communication for the deaf and speech-impaired community.
