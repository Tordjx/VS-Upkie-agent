import depthai as dai
from gymnasium import Wrapper
import torch
import numpy as np
from env.camera_thread import CameraThread
from depthai_nodes.node import ParsingNeuralNetwork
model = 'xfeat:mono-640x480'
def create_pipeline() : 
    device = dai.Device()
    pipeline = dai.Pipeline(device)
    size = np.array([320, 240])
    scale = 1
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    stereo = pipeline.create(dai.node.StereoDepth)

    # Linking
    colorOut = color.requestOutput(size, fps = 10)
    monoLeftOut = monoLeft.requestOutput(size, fps = 10)
    monoRightOut = monoRight.requestOutput(size, fps = 10)
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    stereo.setRectification(False)
    stereo.setExtendedDisparity(False)
    stereo.setLeftRightCheck(True)

    nn = pipeline.create(ParsingNeuralNetwork).build(
        color,    # Pass the RGBD node
        model,   # Your model identifier
        fps = 10
    )
    qNn = nn.out.createOutputQueue()
    qRgb = colorOut.createOutputQueue()
    qDepth = stereo.depth.createOutputQueue()
    parser = nn.getParser(0)
    parser.setMaxKeypoints(128)
    return pipeline, qRgb, qDepth, qNn, device, size, scale,parser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RaspiImageWrapper(Wrapper):
    def __init__(self, env, image_every=10):
        super().__init__(env=env)
        pipeline, qRgb, qDepth, qNn, device, size, scale,parser= create_pipeline()
        self.camera_thread = CameraThread(
            pipeline = pipeline, qRgb= qRgb, qDepth = qDepth , qNn = qNn, size= size, scale = scale ,device = device,parser = parser,fps=10
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
