import depthai as dai
import torch
import numpy as np
from depthai_nodes.node import ParsingNeuralNetwork
from utils.polar_vs import PolarVS
model = 'xfeat:mono-320x240'
def create_pipeline() : 
    device = dai.Device()
    pipeline = dai.Pipeline(device)
    size = np.array([320, 240])
    scale = 1
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setOutputSize(size[0],size[1])
    # Linking
    colorOut = color.requestOutput(size, fps = 10, type = dai.ImgFrame.Type.BGR888p)
    monoLeftOut = monoLeft.requestOutput(size, fps = 10)
    monoRightOut = monoRight.requestOutput(size, fps = 10)
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    stereo.setRectification(False)
    stereo.setExtendedDisparity(False)
    stereo.setLeftRightCheck(True)
    
    nn = pipeline.create(ParsingNeuralNetwork).build(
        colorOut,    # Pass the RGBD node
        model,   # Your model identifier
    )
    qNn = nn.out.createOutputQueue()
    qRgb = colorOut.createOutputQueue()
    qDepth = stereo.depth.createOutputQueue()
    parser = nn.getParser(0)
    parser.setMaxKeypoints(256)
    return pipeline, qRgb, qDepth, qNn, device, size, scale,parser
pipeline, qRgb, qDepth, qNn, device, size, scale,parser = create_pipeline()
vs  = PolarVS()
def get_twist(): 
    twist = None
    rgb_frame = qRgb.get()     # HxWx3 RGB image
    features = qNn.get()
    depth_frame = qDepth.get()  # HxW depth in millimeters
    intrinsics_matrix = rgb_frame.getTransformation().getSourceIntrinsicMatrix()
    source_width, source_height =  rgb_frame.getTransformation().getSourceSize()
    fx = intrinsics_matrix[0][0] *size[0]/(scale * source_width )
    fy = intrinsics_matrix[1][1] *size[1]/(scale * source_height )
    cx = intrinsics_matrix[0][2]*size[0]/(scale * source_width )
    cy = intrinsics_matrix[1][2]*size[1]/(scale * source_height )
    rgb_frame = rgb_frame.getCvFrame()
    depth_frame = depth_frame.getCvFrame()/ 1000
    #depth_frame = np.ones_like(depth_frame)
    rgb_torch = torch.from_numpy(rgb_frame).moveaxis(-1,0)/255
    if not init : 
        init = True
        parser.setTrigger()
    else :
        points = np.array([[f.position.x, f.position.y] for f in features.trackedFeatures])
        n  = len(points)
        points, pointsd = points[0:n:2] ,points[1:n:2]
        if points is not None and pointsd is not None:
            points_polar = vs.points2polar(points,depth_frame, fx,fy,cx,cy)
            points_polar_d = vs.points2polar(pointsd,np.ones_like(depth_frame), fx,fy,cx,cy)
            twist = vs.servo(points_polar, points_polar_d)
    if twist is None : 
        twist = np.zeros(2)
    else : 
        twist = np.array(twist)[[0,-1]]
    
    return twist

pipeline.start()


###############

import matplotlib.pyplot as plt
from collections import deque

history_len = 50  # keep last 200 iterations

# translation histories
tx_hist = deque(maxlen=history_len)
ty_hist = deque(maxlen=history_len)
tz_hist = deque(maxlen=history_len)

# rotation histories
rx_hist = deque(maxlen=history_len)
ry_hist = deque(maxlen=history_len)
rz_hist = deque(maxlen=history_len)

# iteration counter
iter_hist = deque(maxlen=history_len)
iteration = 0
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Translation
line_tx, = ax[0].plot([], [], 'r-', label='tx')
#line_ty, = ax[0].plot([], [], 'g-', label='ty')
#line_tz, = ax[0].plot([], [], 'b-', label='tz')
ax[0].set_ylabel('Translation [units]')
ax[0].legend()
ax[0].set_title('Translation Twist Components')

# Rotation
#line_rx, = ax[1].plot([], [], 'r-', label='roll')
#line_ry, = ax[1].plot([], [], 'g-', label='pitch')
line_rz, = ax[1].plot([], [], 'b-', label='yaw')
ax[1].set_ylabel('Rotation [rad]')
ax[1].set_xlabel('Iteration')
ax[1].legend()
ax[1].set_title('Rotation Twist Components')
import cv2
init = False
twist_ema = 0
while True:  
    twist = None
    rgb_frame = qRgb.get()     # HxWx3 RGB image
    features = qNn.get()
    depth_frame = qDepth.get()  # HxW depth in millimeters
    intrinsics_matrix = rgb_frame.getTransformation().getSourceIntrinsicMatrix()
    source_width, source_height =  rgb_frame.getTransformation().getSourceSize()
    fx = intrinsics_matrix[0][0] *size[0]/(scale * source_width )
    fy = intrinsics_matrix[1][1] *size[1]/(scale * source_height )
    cx = intrinsics_matrix[0][2]*size[0]/(scale * source_width )
    cy = intrinsics_matrix[1][2]*size[1]/(scale * source_height )
    rgb_frame = rgb_frame.getCvFrame()
    depth_frame = depth_frame.getCvFrame()/ 1000
    depth_frame = np.where(depth_frame==0, 1,depth_frame)
    if not init : 
        init = True
        parser.setTrigger()
    else :
        points = np.array([[f.position.x, f.position.y] for f in features.trackedFeatures])
        n  = len(points)
        pointsd, points = points[0:n:2] ,points[1:n:2]
        if points is not None and pointsd is not None:

            for pt in points:
                x, y = pt.ravel()
                cv2.circle(rgb_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            points_polar = vs.points2polar(points,depth_frame, fx,fy,cx,cy)
            points_polar_d = vs.points2polar(pointsd,depth_frame, fx,fy,cx,cy)
            twist = vs.servo(points_polar, points_polar_d)
        if twist is not None:
            twist=  np.array(twist)
            alpha = 0.05
            twist_ema = np.clip(alpha * twist + (1 - alpha) * twist_ema , -1,1)
            # Update histories
            iteration += 1
            iter_hist.append(iteration)
            tx_hist.append(twist_ema[0])
            #ty_hist.append(twist_ema[1])
            #tz_hist.append(twist_ema[2])
            #rx_hist.append(twist_ema[3])
            #ry_hist.append(twist_ema[4])
            rz_hist.append(twist_ema[5])

            # Update translation plot
            line_tx.set_data(iter_hist, tx_hist)
            #line_ty.set_data(iter_hist, ty_hist)
            #line_tz.set_data(iter_hist, tz_hist)
            ax[0].relim()
            ax[0].autoscale_view()

            # Update rotation plot
            #line_rx.set_data(iter_hist, rx_hist)
            #line_ry.set_data(iter_hist, ry_hist)
            line_rz.set_data(iter_hist, rz_hist)
            ax[1].relim()
            ax[1].autoscale_view()

            # Draw the plots
            plt.pause(0.001)

        # Normalize depth to 0-255 for visualization
        depth_norm = np.clip(depth_frame/4  * 255, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        cv2.imshow("RGB Image", rgb_frame)
        cv2.imshow("Depth Image", depth_colored)
        #cv2.imshow("gray",gray)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            init = False
            twist_ema = 0
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

cv2.destroyAllWindows()
