import depthai as dai
from gymnasium import Wrapper
import torch
import numpy as np
from env.camera_thread import CameraThread

def create_pipeline():
    device = dai.Device()
    size = np.array([640, 400])
    pipeline = dai.Pipeline(device)
    scale = 1
    rgbd = pipeline.create(dai.node.RGBD).build(
        True, dai.node.StereoDepth.PresetMode.FAST_ACCURACY, 
        size=tuple(size //scale)
    )
    qRgbd = rgbd.rgbd.createOutputQueue()
    return pipeline, qRgbd, device, size,scale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RaspiImageWrapper(Wrapper):
    def __init__(self, env, image_every=10):
        super().__init__(env=env)
        pipeline, qRgbd, device,size , scale= create_pipeline()
        self.camera_thread = CameraThread(
            pipeline = pipeline, qRgbd= qRgbd, size= size, scale = scale ,device = device,fps=10
        )
        self.camera_thread.start()
        self.request_reinit = False

    def step(self, action):
        s, r, d, t, i = self.env.step(action)
        if self.request_reinit : 
            self.camera_thread.request_reinit()
            self.request_reinit = False
        i['vs_twist'] = self.camera_thread.get_latest()
        return s, r, d, t, i

    def reset(self, **kwargs):
        s, i = self.env.reset(**kwargs)
        i['vs_twist'] = self.camera_thread.get_latest()
        return s, i
