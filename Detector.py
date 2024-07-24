import cv2  
import numpy as np  
import time  
import hashlib  
import math  
from ultralytics import YOLO  
from realsense_camera import RealsenseCamera  
import cvzone 

def get_color_from_class_name(class_name: str) -> tuple:
    """generate a color from a class name"""
    hash_object = hashlib.md5(class_name.encode())  #create an md5 hash object
    hash_hex = hash_object.hexdigest()  #get the hexadecimal representation of the hash
    r = int(hash_hex[0:2], 16)  #extract the red component
    g = int(hash_hex[2:4], 16)  #extract the green component
    b = int(hash_hex[4:6], 16)  #extract the blue component
    return (r, g, b)

class Detector:
    def __init__(self):
        """initialize the detector with the RealSense camera"""
        self.colorList = []  #list to store colors for each class
        self.model = None
        self.rs_camera = RealsenseCamera()  #initialize the RealSense camera

    def loadModel(self, modelPath: str):
        """load a pretrained YOLO model"""
        print("Loading Model....")
        self.model = YOLO(modelPath)  #load the YOLO model from the specified path
        self.classesList = list(self.model.names.values())  #get the list of class names
        self.colorList = [get_color_from_class_name(class_name) for class_name in self.classesList]  #generate colors for each class
        print("Model loaded successfully...")

    def draw_corner_lines(self, image: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int, class_color: tuple, scale_factor: float = 0.1, thickness: int = 5):
        """draw corner lines on an image"""
        line_width = min(int((xmax - xmin) * scale_factor), int((ymax - ymin) * scale_factor))  #calculate line width

        #top-left corner
        cv2.line(image, (xmin, ymin), (xmin + line_width, ymin), class_color, thickness=thickness)
        cv2.line(image, (xmin, ymin), (xmin, ymin + line_width), class_color, thickness=thickness)
        
        #top-right corner
        cv2.line(image, (xmax, ymin), (xmax - line_width, ymin), class_color, thickness=thickness)
        cv2.line(image, (xmax, ymin), (xmax, ymin + line_width), class_color, thickness=thickness)
        
        #bottom-left corner
        cv2.line(image, (xmin, ymax), (xmin + line_width, ymax), class_color, thickness=thickness)
        cv2.line(image, (xmin, ymax), (xmin, ymax - line_width), class_color, thickness=thickness)
        
        #bottom-right corner
        cv2.line(image, (xmax, ymax), (xmax - line_width, ymax), class_color, thickness=thickness)
        cv2.line(image, (xmax, ymax), (xmax, ymax - line_width), class_color, thickness=thickness)

    def calculateDimensions(self, bbox, distance: float) -> tuple:
        """calculate the dimensions of a detected object"""
        xmin, ymin, xmax, ymax = bbox.xyxy[0].cpu().numpy()  #extract bounding box coordinates
        width_pixels = xmax - xmin  #calculate width in pixels
        height_pixels = ymax - ymin  #calculate height in pixels

        width_real = (width_pixels * distance) / self.rs_camera.intrinsics.fx  #calculate real width
        height_real = (height_pixels * distance) / self.rs_camera.intrinsics.fy  #calculate real height

        return width_real, height_real

    def calculate_distance_bw_obj(self, obj1: dict, obj2: dict) -> float:
        """calculate the distance between two detected objects"""
        x1, y1, d1, w1 = obj1['x_center'], obj1['y_center'], obj1['distance'], obj1['width']  #extract object 1 center coordinates, distance, and width
        x2, y2, d2, w2 = obj2['x_center'], obj2['y_center'], obj2['distance'], obj2['width']  

        z1 = d1  #set z1 to object 1 distance
        z2 = d2  #set z2 to object 2 distance

        x1_real = (x1 - self.rs_camera.intrinsics.ppx) * z1 / self.rs_camera.intrinsics.fx  #calculate object 1 real x coordinate
        y1_real = (y1 - self.rs_camera.intrinsics.ppy) * z1 / self.rs_camera.intrinsics.fy  #calculate object 1 real y coordinate
        x2_real = (x2 - self.rs_camera.intrinsics.ppx) * z2 / self.rs_camera.intrinsics.fx  #calculate object 2 real x coordinate
        y2_real = (y2 - self.rs_camera.intrinsics.ppy) * z2 / self.rs_camera.intrinsics.fy  #calculate object 2 real y coordinate

        center_distance = math.sqrt((x2_real - x1_real)**2 + (y2_real - y1_real)**2 + (z2 - z1)**2)  #calculate the distance(euclidean) between the centers of the objects
        edge_distance = center_distance - (w1 / 2) - (w2 / 2)  #calculate the edge-to-edge distance
        return edge_distance

    def createBoundingBox(self, image: np.ndarray, depth_image: np.ndarray, confidence_threshold: float = 0.3, max_objects: int = 8) -> tuple:
        """create bounding boxes around detected objects"""
        results = self.model(image)  #get detection results
        bboxes = results[0].boxes  #extract bounding boxes
        object_info = []  #to store object information

        if len(bboxes) > 0:
            bbox_with_distance = []  #to store bounding boxes with distances

            #calculate distances for each bounding box
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox.xyxy[0].cpu().numpy()
                conf = bbox.conf.cpu().item()

                if conf > confidence_threshold:
                    x_center = int((xmin + xmax) / 2)
                    y_center = int((ymin + ymax) / 2)
                    
                    #average depth around center point to reduce noise
                    kernel_size = 5
                    try:
                        depth_values = depth_image[y_center - kernel_size:y_center + kernel_size, x_center - kernel_size:x_center + kernel_size]
                        if depth_values.size == 0:
                            continue
                        distance = np.median(depth_values) * 0.001  #convert mm to meters
                    except IndexError:
                        continue
                    
                    bbox_with_distance.append((bbox, distance))

            #sortbounding boxes based on distance
            bbox_with_distance.sort(key=lambda x: x[1]) #show max_object nearer to camera

            count = 0  #counter to keep track ofnumber of objects processed/detected
            for bbox, distance in bbox_with_distance:
                if count >= max_objects:
                    break  #exit loop if count of detected objects reaches maximum

                xmin, ymin, xmax, ymax = bbox.xyxy[0].cpu().numpy()  #extract bounding box coordinates
                conf = bbox.conf.cpu().item()  #extract confidence score
                class_idx = int(bbox.cls.cpu().item())  #extract class index

                if conf > confidence_threshold:
                    classLabelText = self.classesList[class_idx]  #get class label text
                    classColor = self.colorList[class_idx]  #get class color

                    x_center = int((xmin + xmax) / 2)  #calculate x center of bounding box
                    y_center = int((ymin + ymax) / 2)  #calculate y center of bounding box

                    width_real, height_real = self.calculateDimensions(bbox, distance)  #calculate real dimensions of object

                    ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)  #convert coordinates to integers

                    #draw the bounding box
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2)  

                    #draw bold corners on bounding box
                    self.draw_corner_lines(image, xmin, ymin, xmax, ymax, classColor, scale_factor=0.1, thickness=5)

                    object_info.append({
                        'class': classLabelText,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width_real,
                        'height': height_real,
                        'distance': distance
                    })
                    
                    #show class labels on bounded boxes
                    displayText = f'{classLabelText.upper()}, {distance:.2f} M, W: {width_real:.2f} M, H: {height_real:.2f} M'
                    label_size, _ = cv2.getTextSize(displayText, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                    cvzone.putTextRect(
                        image, displayText, (xmin+8, ymin + label_size[1]+2),  #image and starting position of the rectangle
                        scale=1, thickness=2,  #font scale and thickness
                        colorT=(255, 255, 255), colorR=classColor,  #text color and rectangle color
                        font=cv2.FONT_HERSHEY_PLAIN,  #font type
                        offset=6,
                        border=1, colorB=classColor  #border thickness and color
                    )

                    count += 1  #increment the counter for each object processed

        return image, object_info

    def predictVideo(self, confidence_threshold: float = 0.5, max_objects: int = 8):
        """predict objects in a video stream and display the results"""
        
        startTime = time.time()  #record the start time for FPS calculation

        try:
            while True:
                ret, bgr_frame, depth_frame = self.rs_camera.get_frame_stream()
                if not ret:
                    break
                
                #fps calculation
                currentTime = time.time()  #get the current time
                fps = 1 / (currentTime - startTime)  #calculate fps
                startTime = currentTime  #update the start time
                
                #normalize and convert depth frame for accurate visualization
                depth_image = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)  #normalize to full range of uint16
                depth_image = depth_image.astype(np.uint8)  #convert back to uint8 for display

                bboxImage, object_info = self.createBoundingBox(bgr_frame, depth_frame, confidence_threshold, max_objects)

                if len(object_info) >= 2:
                    
                    distance_between_objects = self.calculate_distance_bw_obj(object_info[0], object_info[1])
                    distance_text = f"Distance Between Two Objects: {abs(distance_between_objects):.2f} M"

                    #show distance between two detected objects
                    cvzone.putTextRect(
                        bboxImage, distance_text, (20, 80),  #image, text and starting position of the rectangle
                        scale=0.65, thickness=2,  #font scale and thickness
                        colorT=(255, 255, 255), colorR=(0, 0, 0),  #text color and rectangle color
                        font=cv2.FONT_HERSHEY_SIMPLEX,  #font type
                        border=1, colorB=(255, 255, 255)  #border thickness and color
                    )
                    
                #show fps
                cvzone.putTextRect(
                        bboxImage, f"FPS: {int(fps)}", (20, 30),
                        scale=1, thickness=1,
                        colorT=(255, 76, 76), colorR=(243, 254, 184),
                        font=cv2.FONT_HERSHEY_PLAIN,
                        border=1, colorB=(255, 178, 44)
                    )
                
                #show BGR and Depth Frame
                cv2.imshow("BGR Frames", bboxImage)
                cv2.imshow("Depth Frames", depth_image)

                
                key = cv2.waitKey(1)  #wait for a key press for 1 ms
                if key == 27:  #if pressed key is 'Esc' (ASCII value 27)
                    break  #exit loop

        except Exception as e:
            print(f"Exception occurred: {str(e)}")

        finally:
            self.rs_camera.release()
            cv2.destroyAllWindows()
