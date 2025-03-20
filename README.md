# Tennis Match Analysis using Computer Vision

## Overview
This project uses computer vision and deep learning to analyze tennis matches. It employs YOLO for object detection, keypoint estimation for court detection, and tracking algorithms to follow player movements and ball trajectories. The final output overlays detected features and statistics on the video.

## Features
- **Tennis Ball Detection**: Uses YOLO to detect and track the ball.
- **Player Tracking**: Identifies and tracks players throughout the match.
- **Court Keypoint Detection**: Identifies court lines and calculates player positions relative to the court.
- **Shot Speed Estimation**: Calculates the velocity of shots based on ball movement.
- **Mini Court Visualization**: Displays real-time positions of players and the ball on a small overlayed court.

## Installation
To set up the environment, install the following dependencies:

```sh
pip install torch torchvision ultralytics opencv-python pandas numpy matplotlib
```

## Data Preparation
This project uses a pre-trained YOLO model fine-tuned on a custom tennis dataset. To download the dataset from Roboflow:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolov5")
```

## Training
To train the model on a custom dataset, run:

```sh
yolo task=detect mode=train model=yolov5l6.pt data=/path/to/data.yaml epochs=100 imgsz=640
```

## Running the Notebook
1. Load the trained YOLO model.
2. Process an input video for analysis.
3. Draw keypoints, player stats, and ball trajectories.
4. Save the output video.

Execute the following in the notebook:

```python
from ultralytics import YOLO
model = YOLO("/path/to/your/trained/model.pt")
results = model.predict("/path/to/input_video.mp4", save=True)
```

## Output
- **Processed Video**: Annotated video with ball tracking, player movements, and stats.
- **CSV Logs**: Data on shot speeds, player movements, and ball bounces.

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is open-source and available under the MIT License.

