import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=5):
    """Extracts frames from a video at a specified frame rate."""
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frame_count = 0

    while success:
        if count % frame_rate == 0:  # Capture every Nth frame
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}.jpg"), image)
            frame_count += 1
        
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"Extracted {frame_count} frames from {video_path}.")

# Process real and fake videos
for category in ["Real", "Fake"]:
    video_folder = os.path.join("Video_Dataset", category)
    output_folder = os.path.join("Frame_Dataset", category)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(video_folder):
        print(f"Warning: {video_folder} does not exist. Creating directory...")
        os.makedirs(video_folder)
        continue

    for filename in os.listdir(video_folder):
        if filename.endswith((".mp4", ".avi", ".mov")):
            extract_frames(os.path.join(video_folder, filename), output_folder)
