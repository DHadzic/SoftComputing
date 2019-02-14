import tkinter as tk
import cv2
from threading import Thread
from PIL import Image,ImageTk
from face_recognition.FaceDetector import FaceDetector

class MyWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.wm_title("Face detection, gender and emotion classification")
        self.window.config(background="#FFFFFF")
        self.cap = cv2.VideoCapture(0)
        self.video_thread = Thread(target=self.display_video)
        self.imageFrame = tk.Frame(self.window, width=640, height=480)
        self.imageFrame.grid(row=0, column=0, padx=10, pady=2)
        self.label = tk.Label(self.imageFrame)
        self.label.grid(row=0,column=0)
        self.label.pack()

        self.faceDetector = FaceDetector()
        self.video_thread.start()
        self.window.mainloop()


    def display_video(self):
        while True:
            _,frame = self.cap.read()
            frame = cv2.flip(frame, 1)


            frame = self.faceDetector.drawFacesImg(frame)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)


if __name__ == "__main__":

    winddow = MyWindow()