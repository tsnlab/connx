import argparse
import signal
import sys
import threading
import tkinter

import __paths
import cv2
import numpy as np
import run
from PIL import Image, ImageTk


__paths.dummy()  # To prevent E402

np.set_printoptions(suppress=True, linewidth=160, threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='CONNX video demo')
parser.add_argument('connx', nargs=1, help='connx executable path')
parser.add_argument('model', nargs=1, help='connx model path')
parser.add_argument('--id', nargs='?', type=int, help='camera ID')
parser.add_argument('--resize', type=int, metavar='size', nargs=2,
                    help='Resize image to (width, height)')
parser.add_argument('--focus', type=float, metavar='position', nargs=4,
                    help='Focusing area, x, y, width, height')
args = parser.parse_args()

connx_path = args.connx[0]
model_path = args.model[0]
webcam_id = args.id
resize = args.resize
focus = args.focus


class App(threading.Thread):
    def __init__(self, window, title, connx_path, model_path, webcam_id, resize=(0, 0), focus=(0, 0, 1, 1)):
        threading.Thread.__init__(self)

        self.window = window
        self.title = title
        self.resize = tuple(resize)
        self.focus = focus
        self.connx_path = connx_path
        self.model_path = model_path

        self.window = window
        self.window.title(title)

        self.video = cv2.VideoCapture(webcam_id)
        if not self.video.isOpened():
            raise ValueError(f'WebCam #{webcam_id} is already in use')

        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.canvas = tkinter.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack(side='left')

        self.tensor = tkinter.Canvas(self.window, width=self.width / 2, height=self.height / 2)
        self.tensor.pack(side='top')

        self.text = tkinter.Text(self.window, width=40, height=10)
        self.text.pack(side='bottom', fill=tkinter.BOTH, expand=tkinter.YES)

        self.daemon = run.Daemon(self.connx_path, self.model_path)
        self.daemon.start()

        self.frame = None

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def stop(self):
        self.daemon.inference(None)
        self.window.quit()
        self.window.update()
        self.daemon = None

    def run(self):
        while self.daemon is not None:
            frame = self.frame
            if frame is not None:
                frame = np.expand_dims(frame, axis=0)
                frame = np.expand_dims(frame, axis=0)
                frame[frame < 127] = 0
                outputs = self.daemon.inference([frame])
                output = outputs[0][0]
                idx = np.argmax(output)
                tmp = output - np.min(output)
                sum = np.sum(tmp)
                value = tmp[idx]
                self.text.insert(1.0, f'{idx}, {value / sum * 100:0.2f}%\n')

    def loop(self):
        self.update()
        self.window.mainloop()

    def update(self):
        if self.video.isOpened():
            ret, frame = self.video.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                self.canvas.create_rectangle(self.focus[0] * self.width, self.focus[1] * self.height,
                                             self.focus[2] * self.width, self.focus[3] * self.height,
                                             outline='red')

                if self.focus != (0, 0, 1, 1):
                    frame = frame[int(self.focus[1] * self.height):int(self.focus[3] * self.height),
                                  int(self.focus[0] * self.width):int(self.focus[2] * self.width)]

                if self.resize != (0, 0):
                    frame = cv2.resize(frame, dsize=self.resize, interpolation=cv2.INTER_AREA)

                frame = 255 - frame
                frame = frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2]
                min = np.min(frame)
                max = np.max(frame)
                if min != max:
                    frame = (frame - min) / (max - min) * 255
                frame = frame.astype(np.float32)
                self.frame = frame

                frame = cv2.resize(frame, dsize=(int(self.width / 2), int(self.height / 2)),
                                   interpolation=cv2.INTER_AREA)

                self.input = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.tensor.create_image(0, 0, image=self.input, anchor=tkinter.NW)

        self.window.after(10, self.update)


app = App(tkinter.Tk(), 'CONNX Video Demo', connx_path, model_path, webcam_id, resize=resize, focus=focus)


def signal_handler(signal, frame):
    app.stop()


signal.signal(signal.SIGINT, signal_handler)

app.start()
app.loop()
