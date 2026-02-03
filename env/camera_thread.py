import threading

import os
import gin
import numpy as np
from loop_rate_limiters import RateLimiter

gin.parse_config_file("config/settings.gin")
from config.settings import EnvSettings
from scipy.spatial.transform import Rotation as R
import torch
from utils.lg_tracker import LGTrackerCAM 
from utils.polar_vs import PolarVS
config = EnvSettings()
def set_self_affinity(core):
    tid = threading.get_native_id()  # Linux TID
    os.sched_setaffinity(tid, {core})


class CameraThread:
    def __init__(self, pipeline, qRgb, qDepth, qNn, device, size, scale,parser, fps=10):
        """
        camera: your camera interface object, e.g. with .get_image()
        encoder: your encoder object or function
        fps: desired frame rate
        """
        self.device = device
        self.scale = scale
        self.size = size
        self.qNn = qNn
        self.pipeline = pipeline 
        self.qRgb = qRgb
        self.qDepth = qDepth
        self.parser = parser
        self.fps = fps
        self.rate_limiter = RateLimiter(frequency = fps)
        self.latest_twist = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = threading.Thread(target=self.run)
        self.vs  = PolarVS()
        self.init = False
    def request_reinit(self) : 
        self.init =False
    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
    #@profile
    def run(self):
        set_self_affinity(1)
        self.pipeline.start()

        # Enable IR dots & flood
        self.device.setIrLaserDotProjectorIntensity(1.0)
        self.device.setIrFloodLightIntensity(0.0)
        while self.running:
            twist = self.get_twist()
            with self.lock : 

                self.latest_twist = twist

            self.rate_limiter.sleep()
    def get_latest(self):
        with self.lock:
            return self.latest_twist
    
    def get_twist(self): 
        twist = None
        rgb_frame = self.qRgb.get()     # HxWx3 RGB image
        features = self.qNn.get()
        depth_frame = self.qDepth.get()  # HxW depth in millimeters
        intrinsics_matrix = rgb_frame.getTransformation().getSourceIntrinsicMatrix()
        source_width, source_height =  rgb_frame.getTransformation().getSourceSize()
        fx = intrinsics_matrix[0][0] *self.size[0]/(self.scale * source_width )
        fy = intrinsics_matrix[1][1] *self.size[1]/(self.scale * source_height )
        cx = intrinsics_matrix[0][2]*self.size[0]/(self.scale * source_width )
        cy = intrinsics_matrix[1][2]*self.size[1]/(self.scale * source_height )
        rgb_frame = rgb_frame.getCvFrame()
        depth_frame = depth_frame.getCvFrame()/ 1000
        #depth_frame = np.ones_like(depth_frame)
        rgb_torch = torch.from_numpy(rgb_frame).moveaxis(-1,0)/255
        if not self.init : 
            self.init = True
            self.parser.setTrigger()
        else :
            points = np.array([[f.position.x, f.position.y] for f in features.trackedFeatures])
            n  = len(points)
            points, pointsd = points[0:n:2] ,points[1:n:2]
            if points is not None and pointsd is not None:
                points_polar = self.vs.points2polar(points,depth_frame, fx,fy,cx,cy)
                points_polar_d = self.vs.points2polar(pointsd,np.ones_like(depth_frame), fx,fy,cx,cy)
                twist = self.vs.servo(points_polar, points_polar_d)
        if twist is None : 
            twist = np.zeros(2)
        else : 
            twist = np.array(twist)[[0,-1]]
        
        return twist