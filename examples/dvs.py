import cv2
import depthai as dai
import numpy as np
import visp as vp
size = np.array([640, 400])
# --- Create DepthAI pipeline ---
pipeline = dai.Pipeline()
rgbd = pipeline.create(dai.node.RGBD).build(
    True, dai.node.StereoDepth.PresetMode.FAST_ACCURACY, 
    size=tuple(size //4)
)
qRgbd = rgbd.rgbd.createOutputQueue()
pipeline.start()

# DVS visual servoing
servo = vp.vs.Servo()
servo.setServo(vp.vs.Servo.EYEINHAND_CAMERA)
servo.setInteractionMatrixType(vp.vs.Servo.CURRENT)
servo.setLambda(0.2)  # tune gain

init = False
twist_offset = None

###############

import matplotlib.pyplot as plt
from collections import deque

history_len = 200  # keep last 200 iterations

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
line_ty, = ax[0].plot([], [], 'g-', label='ty')
line_tz, = ax[0].plot([], [], 'b-', label='tz')
ax[0].set_ylabel('Translation [units]')
ax[0].legend()
ax[0].set_title('Translation Twist Components')

# Rotation
line_rx, = ax[1].plot([], [], 'r-', label='roll')
line_ry, = ax[1].plot([], [], 'g-', label='pitch')
line_rz, = ax[1].plot([], [], 'b-', label='yaw')
ax[1].set_ylabel('Rotation [rad]')
ax[1].set_xlabel('Iteration')
ax[1].legend()
ax[1].set_title('Rotation Twist Components')


###############
while True:
    inRgbd = qRgbd.get()
    rgb_frame = inRgbd.getRGBFrame().getCvFrame()
    depth_frame = inRgbd.getDepthFrame().getCvFrame()/1000.0  # meters

    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    if not init:
        init = True
        # --- Desired image ---
        Id = vp.core.ImageGray(gray)

        # Build desired visual feature
        sId = vp.visual_features.FeatureLuminance() 
        sId.init(h, w, np.mean(depth_frame))  # Z can be 1.0 for normalized interaction matrix
        sId.buildFrom(Id)

        # Build current feature
        sI = vp.visual_features.FeatureLuminance()
        sI.init(h, w, np.mean(depth_frame))
        sI.buildFrom(Id)

        # Add to servo task
        servo.addFeature(sI, sId)

    else:

        # Update current feature
        sI.init(h,w, np.mean(depth_frame))
        sI.buildFrom( vp.core.ImageGray(gray))

        # Compute camera velocity
        twist = servo.computeControlLaw()
        if twist_offset is None:
            twist_offset = twist

        if twist is not None:
            # Compute difference from the offset
            twist_diff = twist  # should be 6 elements: [tx, ty, tz, rx, ry, rz]

            # Update histories
            iteration += 1
            iter_hist.append(iteration)
            tx_hist.append(twist_diff[0])
            ty_hist.append(twist_diff[1])
            tz_hist.append(twist_diff[2])
            rx_hist.append(twist_diff[3])
            ry_hist.append(twist_diff[4])
            rz_hist.append(twist_diff[5])

            # Update translation plot
            line_tx.set_data(iter_hist, tx_hist)
            line_ty.set_data(iter_hist, ty_hist)
            line_tz.set_data(iter_hist, tz_hist)
            ax[0].relim()
            ax[0].autoscale_view()

            # Update rotation plot
            line_rx.set_data(iter_hist, rx_hist)
            line_ry.set_data(iter_hist, ry_hist)
            line_rz.set_data(iter_hist, rz_hist)
            ax[1].relim()
            ax[1].autoscale_view()

            # Draw the plots
            plt.pause(0.001)



    cv2.imshow("RGB Image", rgb_frame)
    depth_norm = np.clip(depth_frame / 4 * 255, 0, 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imshow("Depth Image", depth_colored)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('r'):
        init = False
        twist_offset = None
        servo = vp.vs.Servo()
        servo.setServo(vp.vs.Servo.EYEINHAND_CAMERA)
        servo.setInteractionMatrixType(vp.vs.Servo.CURRENT)
        servo.setLambda(0.2)
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
