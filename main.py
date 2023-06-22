import art
import logging
import time
import tkinter as tk

import config
from modules.conducter import Conducter
from modules.data_writer import DataWriter
from nebula.hivemind import DataBorg
from nebula.nebula import Nebula


class Visualiser:

    def __init__(self, hivemind):
        self.hivemind = hivemind
        # Build UI
        self.root = tk.Tk()
        self.root.title("JessNIRS")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.canvas = tk.Canvas(self.root, height=600, width=500)
        self.canvas.pack()

        # Assign location points and colors for visuals
        self.centers = {
            'HbO/L': [120, 150],
            'HbO/R': [380, 150],
            'HbR/L': [150, 280],
            'HbR/R': [350, 280]
        }
        self.label_centre = [250, 500]
        self.colors = {
            'HbO/L': '#ff3333',
            'HbO/R': '#ff3333',
            'HbR/L': '#6666ff',
            'HbR/R': '#6666ff'
        }

        # Assign initial sizes of the circles (between 0 and 1)
        self.sizes = {
            'HbO/L': 0.5,
            'HbO/R': 0.5,
            'HbR/L': 0.5,
            'HbR/R': 0.5
        }

        # Build graphics
        self.items = {}
        for ch in self.centers:
            items_xys = [self.centers[ch][0]-40,
                         self.centers[ch][1]-40,
                         self.centers[ch][0]+40,
                         self.centers[ch][1]+40]
            self.items[ch] = self.canvas.create_oval(
                *items_xys, fill=self.colors[ch], outline=self.colors[ch])

        # Add labels
        for ch in self.centers:
            self.canvas.create_text(
                self.centers[ch], text=ch, font=('Helvetica', '15', 'bold'))

        # Stream label
        label_xys = [self.label_centre[0]-150,
                     self.label_centre[1]-50,
                     self.label_centre[0]+150,
                     self.label_centre[1]+50]
        self.frame = self.canvas.create_rectangle(*label_xys, width=3)
        self.label = self.canvas.create_text(
            self.label_centre, text='...', font=('Helvetica', '15', 'bold'))
        self.stream_mapping = {
               'mic_in': 'Microphone',
               'rnd_poetry': 'Random poetry',
               'eeg2flow': 'Brain flow',
               'flow2core': 'Brain groove',
               'core2flow': 'Self awareness',
               'audio2core': 'Audio groove',
               'audio2flow': 'Audio flow',
               'flow2audio': 'Brain audio',
               'fnirs': 'Brain oxygenation'
        }

        self.window_closed = False

    def on_close(self):
        """
        Callback function for when the window is closed.
        """
        self.window_closed = True

    def callback(self):
        """
        Callback function for updating the circle sizes based on the hivemind
        data.
        """
        self.sizes['HbO/L'] = self.hivemind.fnirs_buffer[0][-5:].mean()
        self.sizes['HbO/R'] = self.hivemind.fnirs_buffer[1][-5:].mean()
        self.sizes['HbR/L'] = self.hivemind.fnirs_buffer[2][-5:].mean()
        self.sizes['HbR/R'] = self.hivemind.fnirs_buffer[3][-5:].mean()
        for ch in self.centers:
            items_xys = [self.centers[ch][0]-70*(self.sizes[ch]+0.25),
                         self.centers[ch][1]-70*(self.sizes[ch]+0.25),
                         self.centers[ch][0]+70*(self.sizes[ch]+0.25),
                         self.centers[ch][1]+70*(self.sizes[ch]+0.25)]
            self.canvas.coords(self.items[ch], *items_xys)
        new_label = '...'
        if self.hivemind.thought_train_stream in self.stream_mapping:
            new_label = self.stream_mapping[self.hivemind.thought_train_stream]
        self.canvas.itemconfigure(self.label, text=new_label)

    def make_viz(self):
        """
        Loop to update the window content at 10 Hz.
        """
        while self.hivemind.running:
            self.root.update_idletasks()
            self.root.update()
            if self.window_closed is True:
                self.root.destroy()
                break
            self.callback()
            time.sleep(0.1)


class Main:
    """
    Main script to start the robot arm drawing digital score work.
    Conducter calls the local interpreter for project specific functions. This
    communicates directly to the robot libraries.
    Nebula kick-starts the AI Factory for generating NNet data and affect
    flows.
    This script also controls the live mic audio analyser.
    Paramaters are to be modified in config.py.
    """
    def __init__(self):
        art.tprint("JessNIRS")
        # Build initial dataclass filled with random numbers
        self.hivemind = DataBorg()

        # Logging for all modules
        logging.basicConfig(level=logging.WARNING)

        # Init the AI factory (inherits AIFactory, Listener)
        nebula = Nebula(speed=config.speed)

        # Init Conducter & Gesture management (controls Drawbot)
        robot = Conducter(speed=config.speed)

        # Init data writer
        dw = DataWriter()

        # Start Nebula AI Factory after conducter starts data moving
        nebula.endtime = time.time() + config.duration_of_piece
        self.hivemind.running = True
        robot.main_loop()
        nebula.main_loop()
        dw.main_loop()

        # Visualiser
        if config.viz:
            viz = Visualiser(self.hivemind)
            viz.make_viz()


if __name__ == "__main__":
    Main()
