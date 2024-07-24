# import pyrealsense2 as rs
# import numpy as np

# class RealsenseCamera:
#     def __init__(self):
#         # Configure depth and color streams
#         print("Loading Intel Realsense Camera...")
#         self.pipeline = rs.pipeline()
#         config = rs.config()
#         config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#         # Start streaming
#         profile = self.pipeline.start(config)
#         align_to = rs.stream.color
#         self.align = rs.align(align_to)

#         # Get intrinsics
#         self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

#     def get_frame_stream(self):
#         # Wait for a coherent pair of frames: depth and color
#         frames = self.pipeline.wait_for_frames()
#         aligned_frames = self.align.process(frames)
#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         if not depth_frame or not color_frame:
#             # If there is no frame, probably camera not connected, return False
#             print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
#             return False, None, None

#         # Apply filter to fill the Holes in the depth image
#         spatial = rs.spatial_filter()
#         spatial.set_option(rs.option.holes_fill, 3)
#         filtered_depth = spatial.process(depth_frame)
#         hole_filling = rs.hole_filling_filter()
#         filled_depth = hole_filling.process(filtered_depth)

#         # Create colormap to show the depth of the Objects
#         colorizer = rs.colorizer()
#         depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(filled_depth.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         return True, color_image, depth_image

#     def release(self):
#         self.pipeline.stop()


import pyrealsense2 as rs
import numpy as np

class RealsenseCamera:
    def __init__(self):
        #configure depth and color streams
        print("loading intel realsense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  #configure color stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #configure depth stream

        #start streaming
        profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)  #align depth frame to color frame

        #get intrinsics(parameters of camera)
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()  #get camera intrinsics

    def get_frame_stream(self):
        #wait for a coherent pair of frames(depth and color)
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)  #align frames
        depth_frame = aligned_frames.get_depth_frame()  #get aligned depth frame
        color_frame = aligned_frames.get_color_frame()  #get aligned color frame

        if not depth_frame or not color_frame:
            #if there is no frame, probably camera not connected, return false
            print("error, impossible to get the frame, make sure that the intel realsense camera is correctly connected")
            return False, None, None

        #apply filter to fill holes in depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3) #spatial filter option for filling holes in depth image
        filtered_depth = spatial.process(depth_frame) #apply spatial filter to fill holes
        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)  #apply hole filling filter

        #create colormap to show depth of objects
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())  #create depth colormap

        #convert images to numpy arrays
        depth_image = np.asanyarray(filled_depth.get_data())  #convert filled depth frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())  #convert color frame to numpy array

        return True, color_image, depth_image  #return success flag and images

    def release(self):
        self.pipeline.stop()  #stop camera pipeline