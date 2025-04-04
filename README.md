🧠 Deepfake Detection using Xception Model

This project uses a pre-trained Xception model to detect whether a given video is real or fake by analyzing extracted frames from the video.

📁 Project Structure

Deepfake-Detection/
│
├── app.py                # Streamlit UI to upload and analyze videos
├── main.py               # Model training script using Xception
├── videoframes.py        # Script to extract frames from videos
├── my_xception_model.keras  # Trained model file
├── Frame_Dataset/        # Contains extracted frames from videos (Real/Fake)
│   ├── Real/
│   └── Fake/
├── Video_Dataset/        # Source videos for testing
│   ├── Real/
│   └── Fake/
└── README.md             # You're here!

✅ Requirements

Install required Python libraries using:

pip install -r requirements.txt

If requirements.txt is not available, you can manually install:

pip install tensorflow opencv-python scikit-learn streamlit

🚀 How to Run

1. Prepare Dataset

Place real and fake videos in Video_Dataset/Real/ and Video_Dataset/Fake/.

Use videoframes.py to extract frames into Frame_Dataset/Real/ and Frame_Dataset/Fake/.

python videoframes.py

2. Train the Model

python main.py

This will save the trained model as my_xception_model.keras.

3. Launch the Web App

streamlit run app.py

This opens a browser where you can upload a video and get its classification.

🧪 Features

Uses Xception as a base CNN for transfer learning.

Handles both training and inference modes.

Real-time prediction with Streamlit UI.

Adjustable prediction threshold (e.g., 0.3 for better sensitivity).

🛠️ Configuration

Model input size: 128 x 128

Training: 50 epochs, with early stopping

Optimizer: Adam, learning rate 0.0001

Classes: Real (0) and Fake (1)

📈 Improvements

Add more diverse fake samples.

Fine-tune the Xception layers.

Include more advanced temporal models for better accuracy.
