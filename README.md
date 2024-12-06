Instructions how to run 
Install the Packages:
pip install torch==1.11.0+cu113
pip install numpy
pip install opencv-python
pip install Pillow
pip install matplotlib
pip install scipy
pip install tensorboard
pip install pycocotools
pip install albumentations
pip install pyyaml
pip install tqdm
pip install gitpython
pip install git+https://github.com/ultralytics/yolov5.git


1.System Requirements
•	Operating System: Windows/Linux/macOS.
•	Python: Version 3.7 or later.
2. Install Required Software
•	Python: Install Python from the official website.
•	Git: Install Git for cloning repositories 
3. Set Up the Environment
1.	Open a terminal or command prompt.
2.	Create a virtual environment (optional but recommended):
Run this in terminal
python -m venv yolov5-env
source yolov5-env/bin/activate    # For Linux/Mac
yolov5-env\Scripts\activate      # For Windows
4. Clone the YOLOv5 Repository
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
5. Install Required Dependencies
pip install -r requirements.txt
6. Download Pretrained YOLOv5 Weights
•	Download the YOLOv5 weights (e.g., yolov5s.pt) directly:
python detect.py --weights yolov5s.pt --source 0  # To download weights during execution
7. Running Object Detection
•	On Images: 
python detect.py --weights yolov5s.pt --source path/to/image.jpg
•	On a Video:
python detect.py --weights yolov5s.pt --source path/to/video.mp4
•	Real-Time from Webcam:
python detect.py --weights yolov5s.pt --source 0
8. Customize the Run
To save results in a specific directory:
python detect.py --weights yolov5s.pt --source path/to/image.jpg --project results --name run1
9.Adjust confidence threshold:
python detect.py --weights yolov5s.pt --source path/to/image.jpg --conf-thres 0.4

10 Troubleshooting
•	Issue: CUDA not found or GPU unavailable.
o	Ensure CUDA and cuDNN are correctly installed.
o	Use CPU by adding --device cpu in the command:
python detect.py --weights yolov5s.pt --source path/to/image.jpg --device cpu
Missing Dependencies:
•	Re-run pip install -r requirements.txt.
Cleanup (Optional)
•	Deactivate the virtual environment:
deactivate
