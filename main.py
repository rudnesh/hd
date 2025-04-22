
import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
import pandas as pd
import mediapipe as mp
import threading
from tkinter import messagebox, simpledialog, ttk
from tkinter import filedialog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pyttsx3
import socket
import threading
import time
from collections import deque
import speech_recognition as sr
ESP32_IP = "192.168.113.102"
ESP32_PORT = 1234
ESP32_URL = f"http://{ESP32_IP}:{ESP32_PORT}"
r = sr.Recognizer()
# Initialize the pyttsx3 engine for English
engine = pyttsx3.init()
#speech to text
def speak_text_pyttsx3(text):
    engine.say(text)
    engine.runAndWait()
# Constants
GESTURES_PATH = 'gestures.csv'
TOTAL_DATAPOINTS = 1000
ESP32_IP = '192.168.113.102'  # Replace with your ESP32 IP
ESP32_PORT = 1234

# MediaPipe Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
HandLandmark = mpHands.HandLandmark

# Handpoint Columns
handpoints = (
    [f'{lm.name}_lmx1' for lm in HandLandmark] +
    [f'{lm.name}_lmy1' for lm in HandLandmark] +
    [f'{lm.name}_lmx2' for lm in HandLandmark] +
    [f'{lm.name}_lmy2' for lm in HandLandmark] +
    ['gesture_name']
)

# Flags
stop_data_collection = False
stop_recognition = False
stop_speech = False

# ISL GIF phrases
isl_gif = ['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
                'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
                'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
                'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
                'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
                 'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
                'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
                'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
                'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
                'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
                'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
                'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
                'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy'
]
# Function to send output to ESP32 and update GUI

def handle_output(text):
     root.after(0, lambda: output_label.config(text=f"Output: {output_text}"))
     print("Output:", text)
     output_label.config(text="Output: " + text)
     try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ESP32_IP, ESP32_PORT))
            s.sendall(text.encode('utf-8'))
     except Exception as e:
        print("Failed to send to ESP32:", e)


# Gesture Recognition variables
X, y, knn = None, None, None
q = deque(maxlen=20)

# ------------------- GUI Setup --------------------
root = tk.Tk()
root.title("Intercom Device")
root.geometry("600x700")
root.configure(bg="white")
output_label = tk.Label(root, text="Output: ", font=("Helvetica", 16), wraplength=800, justify="center")
output_label.pack(pady=2)
frame_main = tk.Frame(root, bg="white")
frame_main.pack(fill='both', expand=True)

header = tk.Label(frame_main, text="Intercom Device", font=("Arial", 20, "bold"), fg="#5D3FD3", bg="white")
header.pack(pady=(20, 0))
sub_header = tk.Label(frame_main, text="Speak and listen through mind.", font=("Arial", 12), bg="white")
sub_header.pack(pady=3)

image_label = tk.Label(frame_main)
image_label.pack(pady=0)

output_text = tk.Text(frame_main, height=2, width=40, font=("Arial", 16), bg="#f0f0f0")
output_text.pack(pady=2)

class ImageLabel(tk.Label):
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.loc = 0
        self.frames = []

        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image="")
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            root.after(self.delay, self.next_frame)

lbl = ImageLabel(frame_main)
lbl.pack()
# ------------------- Speech to Text --------------------
predefined_names = {
    "shashi kumar": "Activate vibrator",
    "john doe": "Activate vibrator",  # You can add more predefined names here
}

def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Adjusting for ambient noise... please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("Speech recognition active. Say 'stop' to exit.\n")

    while True:
        try:
            with mic as source:
                print("Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            print("Recognizing...")
            text = recognizer.recognize_google(audio, language='en')
            handle_output(f"Recognized Speech:\n{text}")
            text = text.lower().strip()

            print(f"You said: {text}")

            if "stop" in text:
                print("Stopping Speech Mode...")
                speak_text_pyttsx3("Stopping speech mode.")
                break  # Exit the loop


        except sr.WaitTimeoutError:
            print("Listening timed out, no speech detected.")
            handle_output("Listening timed out.")

        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            handle_output("Could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            handle_output("Speech Recognition service error.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


        except sr.WaitTimeoutError:
            print("Listening timed out, no speech detected.")
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


        except sr.WaitTimeoutError:
            print("Listening timed out, no speech detected.")
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# ------------------- Gesture Data Collection --------------------
def start_data_collection(gesture_name):
    global stop_data_collection
    stop_data_collection = False

    cap = cv2.VideoCapture(0)
    img_no = 0
    landmarks = []
    start_capture_flag = False

    while not stop_data_collection:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, c = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            start_capture_flag = not start_capture_flag

        lmks_total = []

        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(
                    frame, handslms, mpHands.HAND_CONNECTIONS,
                    mpDraw.DrawingSpec(color=(3, 252, 244), thickness=2, circle_radius=2),
                    mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                lmks = []
                for lm in handslms.landmark:
                    lmx = int(lm.x * w)
                    lmy = int(lm.y * h)
                    lmks += [lmx, lmy]
                lmks_total.append(lmks)

            if len(lmks_total) == 1:
                lmks_total.append([0] * 42)
            elif len(lmks_total) == 0:
                lmks_total = [[0] * 42, [0] * 42]

            row = lmks_total[0] + lmks_total[1] + [gesture_name]
            if start_capture_flag and img_no < TOTAL_DATAPOINTS:
                landmarks.append(row)
                img_no += 1
                cv2.putText(frame, f"Capturing {img_no}/{TOTAL_DATAPOINTS}", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (127, 255, 255), 2)

        else:
            cv2.putText(frame, "Show your hand", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, "Press 'c' to toggle capture", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if img_no >= TOTAL_DATAPOINTS:
            stop_data_collection = True
            break

        cv2.imshow("Data Collection", frame)

    cap.release()
    if cv2.getWindowProperty('Data Collection', cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyAllWindows()

    if landmarks:
        df = pd.DataFrame(landmarks, columns=handpoints)
        if os.path.exists(GESTURES_PATH):
            df_existing = pd.read_csv(GESTURES_PATH)
            df_existing = df_existing[handpoints]
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(GESTURES_PATH, index=False)
        messagebox.showinfo("Success", f"Saved {len(landmarks)} samples for gesture: {gesture_name}")

def launch_data_collection():
    gesture_name = simpledialog.askstring("Input", "Enter Gesture Name before starting:")
    if not gesture_name:
        messagebox.showwarning("Cancelled", "No gesture name provided.")
        return
    threading.Thread(target=start_data_collection, args=(gesture_name,), daemon=True).start()

def stop_data():
    global stop_data_collection
    stop_data_collection = True

# ------------------- Gesture Recognition --------------------
def start_recognition():
    if not os.path.exists(GESTURES_PATH):
        messagebox.showerror("Error", "No gesture data found. Please collect data first.")
        return

    def recognize():
        global stop_recognition
        stop_recognition = False

        df = pd.read_csv(GESTURES_PATH)
        for col in df.columns[:-1]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            messagebox.showerror("Error", "No valid gesture data available. Please collect gesture data again.")
            return

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        cap = cv2.VideoCapture(0)

        while not stop_recognition:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, c = frame.shape
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            lmks_total = []

            if result.multi_hand_landmarks:
                for handslms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                    lmks = []
                    for lm in handslms.landmark:
                        lmx = int(lm.x * w)
                        lmy = int(lm.y * h)
                        lmks += [lmx, lmy]
                    lmks_total.append(lmks)

                if len(lmks_total) == 1:
                    lmks_total.append([0] * 42)
                elif len(lmks_total) == 0:
                    lmks_total = [[0] * 42, [0] * 42]

                lmks_combined = lmks_total[0] + lmks_total[1]
                prediction = model.predict([lmks_combined])[0]
                cv2.putText(frame, f'Gesture: {prediction}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 50), 2)
                handle_output(f"Recognized Gesture:\n{prediction}")
                text = prediction.lower().strip()
                print(f"You said: {text}")
                

            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if cv2.getWindowProperty('Gesture Recognition', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyAllWindows()

    threading.Thread(target=recognize, daemon=True).start()

def stop_recog():
    global stop_recognition
    stop_recognition = True

# ------------------- Speech to ISL --------------------
def send_to_esp32(text):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ESP32_IP, ESP32_PORT))
        sock.sendall(text.encode())
        sock.close()
    except Exception as e:
        print("Error sending to ESP32:", e)

def func():
        r = sr.Recognizer()
        isl_gif=['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
                'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
                'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
                'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
                'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
                 'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
                'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
                'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
                'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
                'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
                'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
                'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
                'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy']
        
        
        arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r', 's','t','u','v','w','x','y','z']
        with sr.Microphone() as source:
                # image   = "signlang.png"
                # msg="HEARING IMPAIRMENT ASSISTANT"
                # choices = ["Live Voice","All Done!"] 
                # reply   = buttonbox(msg,image=image,choices=choices)
                r.adjust_for_ambient_noise(source) 
                i=0
                while True:
                        print("I am Listening")
                        audio = r.listen(source)
                        # recognize speech using Sphinx
                        try:
                                a=r.recognize_google(audio)
                                a = a.lower()
                                print('You Said: ' + a.lower())
                                
                                for c in string.punctuation:
                                    a= a.replace(c,"")
                                    
                                if(a.lower()=='goodbye' or a.lower()=='good bye' or a.lower()=='bye'):
                                        print("oops!Time To say good bye")
                                        break
                                
                                elif(a.lower() in isl_gif):
                                    
                                    class ImageLabel(tk.Label):
                                            """a label that displays images, and plays them if they are gifs"""
                                            def load(self, im):
                                                if isinstance(im, str):
                                                    im = Image.open(im)
                                                self.loc = 0
                                                self.frames = []

                                                try:
                                                    for i in count(1):
                                                        self.frames.append(ImageTk.PhotoImage(im.copy()))
                                                        im.seek(i)
                                                except EOFError:
                                                    pass

                                                try:
                                                    self.delay = im.info['duration']
                                                except:
                                                    self.delay = 100

                                                if len(self.frames) == 1:
                                                    self.config(image=self.frames[0])
                                                else:
                                                    self.next_frame()

                                            def unload(self):
                                                self.config(image=None)
                                                self.frames = None

                                            def next_frame(self):
                                                if self.frames:
                                                    self.loc += 1
                                                    self.loc %= len(self.frames)
                                                    self.config(image=self.frames[self.loc])
                                                    self.after(self.delay, self.next_frame)
                                    root = tk.Tk()
                                    lbl = ImageLabel(root)
                                    lbl.pack()
                                    lbl.load(r'ISL_Gifs/{0}.gif'.format(a.lower()))
                                    root.mainloop()
                                else:
                                    for i in range(len(a)):
                                                    if(a[i] in arr):
                                            
                                                            ImageAddress = 'letters/'+a[i]+'.jpg'
                                                            ImageItself = Image.open(ImageAddress)
                                                            ImageNumpyFormat = np.asarray(ImageItself)
                                                            plt.imshow(ImageNumpyFormat)
                                                            plt.draw()
                                                            plt.pause(0.8)
                                                    else:
                                                            continue

                        except:
                               print(" ")
                        plt.close()

def stop_speech_thread():
    global stop_speech
    stop_speech = True

def start_speech_thread():
    threading.Thread(target=recognize_speech, daemon=True).start()

def exit_app():
    root.destroy()

# ------------------- GUI Buttons --------------------
section_train = tk.Label(frame_main, text="TRAIN", font=("Arial", 12, "bold"), bg="white", fg="#5D3FD3")
section_train.pack()
btn_train_start = tk.Button(frame_main, text="Start to Train gesture", width=30, command=launch_data_collection)
btn_train_start.pack(pady=5)
btn_train_stop = tk.Button(frame_main, text="Stop Train gesture", width=30, command=stop_data)
btn_train_stop.pack(pady=5)

section_recognize = tk.Label(frame_main, text="RECOGNIZE", font=("Arial", 12, "bold"), bg="white", fg="#5D3FD3")
section_recognize.pack(pady=(5))
btn_rec_start = tk.Button(frame_main, text="Start Gesture Recognition", width=30, command=start_recognition)
btn_rec_start.pack(pady=5)
btn_rec_stop = tk.Button(frame_main, text="Stop Gesture Recognition", width=30, command=stop_recog)
btn_rec_stop.pack(pady=5)

section_speech = tk.Label(frame_main, text="SPEECH TO TEXT", font=("Arial", 12, "bold"), bg="white", fg="#5D3FD3")
section_speech.pack(pady=(20, 0))
btn_speech_start = tk.Button(frame_main, text="Start to listen", width=30, command=start_speech_thread)
btn_speech_start.pack(pady=5)
btn_speech_stop = tk.Button(frame_main, text="Stop to listen", width=30, command=stop_speech_thread)
btn_speech_stop.pack(pady=5)

section_translate = tk.Label(frame_main, text="SPEECH TO ISL TRANSLATE", font=("Arial", 12, "bold"), bg="white", fg="#5D3FD3")
section_translate.pack(pady=(20, 0))
btn_translate_start = tk.Button(frame_main, text="Start to translate from voice input", width=30, command=func)
btn_translate_start.pack(pady=5)

btn_exit = tk.Button(frame_main, text="EXIT", bg="#5D3FD3", fg="white", font=("Arial", 12, "bold"), width=20, command=exit_app)
btn_exit.pack(pady=(30, 10))

root.mainloop()
