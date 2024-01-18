import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(train_history, val_history, title, filepath, step_size=1):
    ## PLOT LOSS
    x = torch.arange(len(train_history)) * step_size
    fig = plt.figure(figsize=(4, 4), dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(x, train_history, alpha=0.7, lw=0.4, label='Training Loss')
    ax.plot(x, val_history, alpha=0.7, lw=0.4, label='Validation Loss')
    ax.legend(loc=1, prop={'size': 4})
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.tick_params(axis='both', which='minor', labelsize=4)
    plt.title(title, fontsize=6, fontweight="bold")
    plt.xlabel("Epoch", fontsize=5)
    plt.ylabel("Cross Entropy Loss", fontsize=5)
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    plt.savefig(filepath)
    plt.close()
    return

def boxplot(list, title, filepath, labels=None):
    ## INITIALIZE FIGURE
    fig = plt.figure(figsize=(4, 4), dpi=400)
    ax = fig.add_subplot(111)
    if labels is None: labels = [f"layer_{i}" for i in range(len(list))]
    ## DRAW PLOT
    ax.boxplot(list, labels=labels)
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.tick_params(axis='both', which='minor', labelsize=4)
    ## FORMAT PLOT
    plt.title(title, fontsize=6, fontweight="bold")
    plt.xlabel("Layer", fontsize=5)
    plt.ylabel("Value", fontsize=5)
    plt.xticks(rotation=90)
    plt.tight_layout()
    ## SAVE AND RETURN
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    plt.savefig(filepath)
    plt.close()


class Animation:

    def __init__(self, title, filepath, update_func, xlim=None, ylim=None, dpi=400, fps=5):
        self.filepath = filepath
        self.video = None
        self.width = None
        self.height = None
        self.layers = None
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.fps = fps
        self.update_func = update_func
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        ## INITIALIZE FIGURE
        self.fig = plt.figure(figsize=(4, 4), dpi=dpi)
        self.ax = self.fig.add_subplot(111)    


    def format(self):
        self.ax.set_title(self.title, fontsize=6, fontweight="bold")
        self.ax.set_xlabel("Layer", fontsize=5)
        self.ax.set_ylabel("Value", fontsize=5)
        self.ax.tick_params(axis='both', which='major', labelsize=4)
        self.ax.tick_params(axis='both', which='minor', labelsize=4)
        if self.xlim: self.ax.set_xlim(self.xlim)
        if self.ylim: self.ax.set_ylim(self.ylim)

    def __call__(self, **kwargs):
        ## UPDATE FIGURE
        self.ax.cla()
        self.format()
        self.update_func(self.ax, **kwargs)

        ## SET AXIS FORMATTING
        self.ax.set_xticks(self.ax.get_xticks())
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=-45, ha='right')    

        ## DRAW FRAME
        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        ## INITIALIZE VIDEO
        if self.video is None:
            self.height, self.width, self.layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))

        ## WRITE FRAME
        self.video.write(frame)
    
    def __del__(self):
        try:
            self.video.release()
            cv2.destroyAllWindows()
        except Exception:
            pass


def update_boxplot(ax, data, labels):
    ax.boxplot(data, labels=labels)
    