#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:08:57 2023

@author: janus
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk


class ImageMarker:
    def __init__(self, images):
        self.root = tk.Tk()
        self.images = images
        self.current_image = 0
        self.colors = ['red', 'green', 'blue', 'yellow', 'black']
        self.color_index = 0
        self.img_label = tk.Label(self.root)
        self.img_label.pack(fill='both', expand=True)
        self.load_image()
        self.root.bind("<Left>", self.previous_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<space>", self.mark_image)

    def load_image(self):
        for i, image in enumerate(self.images):
            image_current = Image.open(image)
            print(image_current)
            image_current = image_current.resize((500, 500), Image.Resampling.LANCZOS)
            self.img = ImageTk.PhotoImage(image_current)
            self.img_label.config(image=self.img)

    def previous_image(self, event=None):
        self.current_image = max(0, self.current_image - 1)
        self.load_image()

    def next_image(self, event=None):
        self.current_image = min(len(self.images) - 1, self.current_image + 1)
        self.load_image()

    def mark_image(self, event=None):
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        self.img_label.config(bg=color)

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    files = filedialog.askopenfilenames(title="Select images",
                                        filetypes=(("Image files", "*.jpg;*.png"),("All files", "*.*")))
    print(files)
    if not files:
        messagebox.showwarning("Warning", "No images selected")
    else:
        app = ImageMarker(files)
        app.run()