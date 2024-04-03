Welcome to my project on real-time drowsiness detection using state-of-the-art technologies - YOLO, Python, and PyTorch. In this project, 
I aim to develop a robust computer vision system capable of detecting signs of drowsiness in real-time video streams.
Drowsiness detection is a critical aspect in various domains such as driver monitoring systems, workplace safety, and healthcare.
By leveraging deep learning techniques and object detection methods, I endeavor to create a solution that can identify drowsiness in individuals and enable 
timely intervention to prevent accidents or adverse events.

Project Objectives:

Install and import necessary dependencies for the project, including YOLO, PyTorch, and OpenCV.
Load a pre-trained YOLOv5 model for object detection.
Make detections on sample images and real-time video streams using the loaded model.
Collect training images for custom drowsiness detection using a webcam.
Annotate collected images using the LabelImg tool for training data preparation.
Train a custom YOLOv5 model using the collected and annotated images.
Evaluate the performance of the custom-trained model on sample images and real-time video streams.
Implement real-time drowsiness detection using the custom-trained model and a webcam.
Project Workflow:

Installation and Dependency Setup: I start by installing and importing all the necessary dependencies required for the project, including Torch, Matplotlib, NumPy, and OpenCV. Additionally, I clone the YOLOv5 repository from GitHub and install the required dependencies using the provided requirements.txt file.

Model Loading: I load a pre-trained YOLOv5 model (yolov5s) using the torch.hub.load() function from the Ultralytics repository. This pre-trained model serves as the foundation for drowsiness detection in both sample images and real-time video streams.

Detections on Sample Images: I make detections on a sample image using the loaded pre-trained model. The results are printed and displayed using Matplotlib, showcasing the model's capability to identify objects, including drowsiness signs, in static images.

Real-Time Detections: I implement real-time drowsiness detection using a webcam. By capturing video frames in real-time, I feed them to the pre-trained model for detections. The results are displayed using OpenCV, enabling live monitoring and intervention for drowsiness detection.

Custom Model Training: To enhance the drowsiness detection capabilities, I collect and annotate training images using a webcam. These images are then used to train a custom YOLOv5 model, tailored specifically for drowsiness detection.

Evaluation and Implementation: Once the custom model is trained, I evaluate its performance on sample images and real-time video streams. Finally, I implement real-time drowsiness detection using the custom-trained model, enabling practical application in various scenarios.

Conclusion:
Through this project, I aim to develop a reliable and efficient system for real-time drowsiness detection, leveraging the power of YOLO, Python, and PyTorch. By combining deep learning techniques with real-world data collection and model training, I aspire to contribute towards enhancing safety and well-being in critical domains such as driver monitoring, workplace safety, and healthcare.
