import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import easyocr
import pandas as pd
from multiprocessing import Pool
from IPython.display import display, clear_output
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

class VideoAnnotator:
    def __init__(self, folder, user, sess):
        
        full_path = os.path.join(folder + '/' + user, user + '_' + sess)
        # Find video file and extract timestamp
        for file_name in os.listdir(full_path):
            if file_name.endswith(".MP4"):
                video_file = os.path.join(full_path, file_name)
                # Assuming timestamp is the number after "Final" in the filename
                timestamp_match = re.search(r"Final(\d+)", file_name)
                if timestamp_match:
                    self.video_timestamp = int(timestamp_match.group(1))*1000
                break
        
        self.status = user + ', ' + sess
        self.cap = cv2.VideoCapture(video_file)
        self.reader = easyocr.Reader(['en'],gpu = False)
        self.df = pd.DataFrame(columns=['timestamp', 'status', 'ocr_string'])
        
    def run(self, speed=1):
        
        prev_ocr = ''
        prev_ocr_gover = ''
        
        #ROIs where we see pause/resume status and game over text
        x1, y1, x2, y2 = 750, 600, 1200, 750 
        x3, y3, x4, y4 = 725, 900, 1225, 1150
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if not ret:
                break
        
            if frame_count % speed == 0:
            
                # Extract the ROI from the image
                roi = frame[y1:y2, x1:x2]
                roi_gameover = frame[y3:y4, x3:x4]

                #OCR it
                ocr_output = self.reader.readtext(roi)
                if len(ocr_output):
                    ocr_result = ocr_output[0][1].lower()
                else:
                    ocr_result = ''

                ocr_output_gover = self.reader.readtext(roi_gameover)
                if len(ocr_output_gover):
                    ocr_result_gover = ocr_output_gover[0][1].lower()
                else:
                    ocr_result_gover = ''

                #check if OCR has changed, indicating an update in game status
                if (prev_ocr != ocr_result or prev_ocr_gover != ocr_result_gover):

                    prev_ocr = ocr_result
                    prev_ocr_gover = ocr_result_gover

                    status = 'NOTIFICATION'

                    if (ocr_result == '' and ocr_result_gover == ''):
                        status='PLAYING'
                    elif('game' in ocr_result_gover or 'over' in ocr_result_gover):
                        status='GAME_OVER'
                    elif('start' in ocr_result or 'resume' in ocr_result):
                        status='PAUSED'
                    elif('falling' in ocr_result or 'blocks' in ocr_result or 'player' in ocr_result_gover):
                        status='BOOT_MENU'
                    elif('latest' in ocr_result or 'public' in ocr_result or 'score' in ocr_result):
                        status='SCOREBOARD'
                    elif(ocr_result != ''):
                        status='UNKNOWN'

                    new_row = {
                    'timestamp': self.cap.get(cv2.CAP_PROP_POS_MSEC) + self.video_timestamp,
                    'status': status,          # An example status
                    'ocr_string': ocr_result + ' | ' + ocr_result_gover
                    }
                    
                    self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

            frame_count += 1

        self.cap.release()
        
        return self.df


def process_video(args):
    folder, user, session = args
    try:
        print(f'Starting P{user} Session #{session}')
        video_meta_path = os.path.join(folder, f'P{user}', f'P{user}_sess{session}', 'video_meta.pkl')
        if not Path(video_meta_path).is_file():
            va = VideoAnnotator(folder, 'P' + str(user), 'sess' + str(session))
            video_metadata = va.run(speed=8)
            video_metadata.to_pickle(video_meta_path)
            return f"Completed user {user}, session {session}"
        else:
            return f"Skipping user {user}, session {session} - 'video_meta.pkl' already exists"
    except Exception as e:
        return f"Error processing video for user {user} session {session}: {e}"


if __name__=='__main__':
    
    # Define your folder and list of users and sessions
    folder = '/Volumes/Secondary/PhDStudy_Results'
    users = list(range(1, 26))  # Users 1 to 25
    sessions = [1, 2]  # Sessions 1 and 2

    # Create a list of all combinations of users and sessions
    tasks = [(folder, user, session) for user in users for session in sessions]

    # Create a multiprocessing pool and process all videos in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:  # Change the number 4 to control how many videos are processed at once
        futures = {executor.submit(process_video, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            print(future.result())