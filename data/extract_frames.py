"""
Helper script for extracting frames from the UCF-101 dataset
"""

import av
import glob
import os
import time
import tqdm
import datetime
import argparse


def extract_frames(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


prev_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="UCF-101", help="Path to UCF-101 dataset")
    opt = parser.parse_args()
    print(opt)

    time_left = 0
    video_paths = glob.glob(os.path.join(opt.data_path, "*", "*.avi"))
    for i, video_path in enumerate(video_paths):
        sequence_type, sequence_name = video_path.split("/")[-2:]
        sequence_name = sequence_name.split(".avi")[0]
        sequence_path = os.path.join(f"{opt.data_path}-frames", sequence_type, sequence_name)

        if os.path.exists(sequence_path):
            continue

        os.makedirs(sequence_path, exist_ok=True)

        # Extract frames
        for j, frame in enumerate(
            tqdm.tqdm(
                extract_frames(video_path, time_left),
                desc=f"[{i}/{len(video_paths)}] {sequence_name} : ETA {time_left}",
            )
        ):
            frame.save(os.path.join(sequence_path, f"{j}.jpg"))

        # Determine approximate time left
        videos_left = len(video_paths) - (i + 1)
        time_left = datetime.timedelta(seconds=videos_left * (time.time() - prev_time))
        prev_time = time.time()
