import tkinter as tk
import cv2
from threading import Thread
from tkinter import ttk
from PIL import Image,ImageTk
from FaceDetector import FaceDetector
from EmotionClassifier import EmotionClassifier
from Gender_classifier import GenderClassifier
from consoleMain import edit_frame

class MyWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.wm_title("Face detection, gender and emotion classification")
        self.window.config(background="#FFFFFF")
        self.cap = None
        self.mode = "camera"
        self.file_path = ""
        self.video_thread = Thread(target=self.display_video)

        self.imageFrame = tk.Frame(self.window, width=640, height=480)
        self.imageFrame.grid(row=0, column=0, padx=10, pady=2)

        self.combobox_types = ttk.Combobox(self.imageFrame,values=["Video","Picture","Camera"],state="readonly")
        self.combobox_types.set("Camera")
        self.combobox_types.grid(row = 0, column = 0,sticky= tk.W,pady=2,padx=4)

        self.path_label = tk.Label(self.imageFrame, text = "File path:")
        self.path_label.grid(row = 0, column = 1,sticky= tk.W,pady=2,padx=4)

        self.path_entry = tk.Entry(self.imageFrame,width = 50)
        self.path_entry.grid(row = 0, column = 2,sticky= tk.W,pady=2,padx=4)

        self.ldBtn = tk.Button(self.imageFrame, text = "Load", command = self.display_file,width=5)
        self.ldBtn.grid(row = 0, column = 3,sticky= tk.E,pady=2,padx=4)

        self.main_frame = tk.Label(self.imageFrame)
        self.main_frame.grid(row=1,column=0,columnspan = 4, sticky = tk.S)

        self.faceDetector = FaceDetector()
        self.ec = EmotionClassifier()
        self.go = GenderClassifier()
        self.video_thread.start()
        self.window.mainloop()

    def display_file(self):
        self.file_path = self.path_entry.get()
        self.mode = self.combobox_types.get()

    def display_video(self):
        while True:
            if self.mode == "Camera":
                self.cap = cv2.VideoCapture(0)
                while True:
                    if self.mode != "Camera":
                        break
                    _,frame = self.cap.read()
                    frame = cv2.flip(frame, 1)
                    self.faceDetector.drawFacesImg(frame)

                    emotions, emotion_confs = self.ec.classify_emotion(frame,fd.boxes)
                    genders, gender_confs = self.go.classify_gender(frame,fd.boxes)
                    frame = edit_frame(frame, emotions, genders, emotion_confs, gender_confs, fd.boxes)

                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.main_frame.imgtk = imgtk
                    self.main_frame.configure(image=imgtk)
            elif self.mode == "Video":
                self.cap = cv2.VideoCapture(self.file_path)
                while True:
                    if self.mode != "Video":
                        break
                    read_bool, frame = self.cap.read()
                    if not read_bool:
                        break
                    frame = cv2.flip(frame, 1)

                    self.faceDetector.drawFacesImg(frame)

                    emotions, emotion_confs = self.ec.classify_emotion(frame,fd.boxes)
                    genders, gender_confs = self.go.classify_gender(frame,fd.boxes)
                    frame = edit_frame(frame, emotions, genders, emotion_confs, gender_confs, fd.boxes)

                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.main_frame.imgtk = imgtk
                    self.main_frame.configure(image=imgtk)
            elif self.mode == "Picture":
                frame = cv2.imread(self.file_path)
                if frame is None:
                    break
                self.faceDetector.drawFacesImg(frame)

                emotions, emotion_confs = self.ec.classify_emotion(frame,fd.boxes)
                genders, gender_confs = self.go.classify_gender(frame,fd.boxes)
                frame = edit_frame(frame, emotions, genders, emotion_confs, gender_confs, fd.boxes)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.main_frame.imgtk = imgtk
                self.main_frame.configure(image=imgtk)


if __name__ == "__main__":

    winddow = MyWindow()