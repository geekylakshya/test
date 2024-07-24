from Detector import *

#https://github.com/ultralytics/ultralytics?tab=readme-ov-file

modelPath = "models\yolov8m.pt"

confidence_threshold = 0.5
max_objects = int(input("Please enter the maximum number of objects to detect in a frame: "))

detector = Detector()
detector.loadModel(modelPath)
detector.predictVideo(confidence_threshold, max_objects)

