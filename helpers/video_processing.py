import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import threading
import queue
import time
from PIL import Image
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Data update thread
class DataUpdater(threading.Thread):
    def __init__(self, df, fig_queue, update_event):
        threading.Thread.__init__(self)
        self.df = df
        self.idx = 0
        self.fig_queue = fig_queue
        self.update_event = update_event

    def run(self):
        while True:
            if self.update_event.is_set():
                timestamp, _ = self.fig_queue.get()
                if self.df.loc[self.idx, "timestamp"] <= timestamp:
                    new_data = []
                    while self.idx < len(self.df) and self.df.loc[self.idx, "timestamp"] <= timestamp:
                        new_data.extend(self.df.loc[self.idx, "data"])
                        self.idx += 1
                    fig, axs = plt.subplots(6, 1, figsize=(6, 9))  # creating 6 subplots
                    for i, ax in enumerate(axs):
                        ax.clear()
                        ax.plot(new_data)
                    plt.close(fig)
                    self.fig_queue.put(fig)
                    self.update_event.clear()


# Frame processing thread
class FrameProcessor(threading.Thread):
    def __init__(self, video_path, fig_queue, update_event, display_queue):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(video_path)
        self.fig_queue = fig_queue
        self.update_event = update_event
        self.display_queue = display_queue

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                break
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # current timestamp
            self.fig_queue.put((timestamp, frame))
            self.update_event.set()
            fig = self.fig_queue.get()
            plot_img = fig_to_img(fig)
            plot_img = np.array(plot_img)
            plot_img = plot_img[:, :, ::-1].copy()  # Convert RGB to BGR
            self.display_queue.put((frame, plot_img))

def fig_to_img(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image


def main(video_path, df, speed):
    fig_queue = queue.Queue()
    display_queue = queue.Queue()
    update_event = threading.Event()

    data_updater = DataUpdater(df, fig_queue, update_event)
    frame_processor = FrameProcessor(video_path, fig_queue, update_event, display_queue)

    data_updater.start()
    frame_processor.start()

    while True:
        frame, plot_img = display_queue.get()
        cv2.imshow("Video", frame)
        cv2.imshow("Plots", plot_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def generate_timestamps(length, start_time, sample_rate):
    time_offset = np.arange(length) / sample_rate
    timestamps = pd.to_datetime(start_time, unit='s', origin='unix') + pd.to_timedelta(time_offset, unit='s')
    return timestamps

def process_file(file_path, file_name):
    with open(file_path, 'r') as file:
        # only take the first timestamp
        start_time = float(file.readline().split(',')[0].strip())
        df = pd.DataFrame()
        if 'IBI' not in file_name:
            sample_rate = float(file.readline().split(',')[0].strip())
            data = pd.read_csv(file, header=None).values.tolist()
            df['timestamp'] = generate_timestamps(len(data), start_time, sample_rate)
        else:
            data = pd.read_csv(file, header=None, names=['time', 'ibi']).values.tolist()
            df['timestamp'] = pd.to_datetime([item[0] + start_time for item in data], unit='s', origin='unix')
        df['datatype'] = file_name.split('.')[0]
        df['data'] = data
        return df

def load_session_data(participant_id, session_id):
    data_dir = f'Data/{participant_id}/{participant_id}_{session_id}'
    data_frames = []

    expected_files = ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'TEMP.csv']

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file in expected_files:
                file_path = subdir + os.sep + file
                df = process_file(file_path, file)
                data_frames.append(df)

    final_df = pd.concat(data_frames)
    final_df.sort_values(by=['timestamp'], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    return final_df

if __name__ == "__main__":
    video_file = 'Data/P9/P9_sess1/RPReplay_Final1682089728.MP4'
    df = load_session_data('P9', 'sess1')
    main(video_file, df, speed=1)

