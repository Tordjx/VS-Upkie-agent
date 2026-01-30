

import cv2
import depthai as dai
import numpy as np
import visp as vp
import cv2
import numpy as np

class KLTTracker:
    def __init__(self, max_corners=500, quality_level=0.1, min_distance=5, win_size=10):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.win_size = win_size
        self.prev_gray = None
        self.points = None
        self.status = None
    def init_tracking(self, gray_frame):
        # detect features to track
        self.points = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        if self.points is not None:
            self.points = cv2.cornerSubPix(
                gray_frame,
                self.points,
                winSize=(self.win_size, self.win_size),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
            )
        self.prev_gray = gray_frame.copy()

    def track(self, gray_frame):
        if self.points is None or len(self.points) == 0:
            return []

        new_points, status, _ =cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_frame,
            self.points, None,
            winSize=(21, 21),    # larger window
            maxLevel=3,          # pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )


        # only keep points that were successfully tracked
        good_new = new_points
        self.status = status
        self.points = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray_frame.copy()
        return self.points,status
class VisualServoPoints:
    def __init__(self, lambda_gain=0.2):
        self.task = vp.vs.Servo()
        self.task.setServo(vp.vs.Servo.EYEINHAND_CAMERA)
        self.task.setInteractionMatrixType(vp.vs.Servo.CURRENT)
        self.task.setLambda(lambda_gain)

        self.p = []      # current features
        self.pd = []     # desired features
        self.active_ids = None
        self.initialized = False

    def set_desired(self, points3d, status):
        status = status if status is not None else np.ones(len(points3d), bool)
        valid = status & (points3d[:,2] > 0)

        self.active_ids = np.where(valid)[0]

        for i in self.active_ids:
            x, y, Z = points3d[i]

            pd = vp.visual_features.FeaturePoint()
            pd.set_xyZ(x, y, Z)

            p = vp.visual_features.FeaturePoint()
            p.set_xyZ(x, y, Z)

            self.task.addFeature(p, pd)
            self.p.append(p)
            self.pd.append(pd)

        self.initialized = True

    def servo(self, points3d, status):
        if not self.initialized:
            return None

        status = status.ravel().astype(bool)
        valid = status & (points3d[:,2] > 0)

        # only update features that still exist
        j = 0
        for i in self.active_ids:
            if not valid[i]:
                continue

            x, y, Z = points3d[i]
            self.p[j].set_xyZ(x, y, Z)
            j += 1

        if j < 4:
            return None   # not enough constraints

        return self.task.computeControlLaw()


    def points2d_to_3d(self,pts2d,depth, fx,fy,cx,cy):
        pts3d = []

        h, w = depth.shape

        for u, v in pts2d:
            ui, vi = int(round(u)), int(round(v))

            if ui < 0 or ui >= w or vi < 0 or vi >= h:
                pts3d.append([0,0,0])
                continue

            Z = depth[vi, ui] 
            if Z <= 0 or np.isnan(Z):
                pts3d.append([0,0,0])
                continue

            X = (u - cx)/ fx
            Y = (v - cy)/ fy

            pts3d.append([X, Y, Z])

        return np.array(pts3d)
    def se2_twist(self,twist) :
        return np.array([twist[0], twist[-1]])

qualityLevel=0.001      # detect weaker corners
minDistance=7           # allow denser points
maxCorners=1000         # yes, really
vs = VisualServoPoints()
tracker = KLTTracker(quality_level=qualityLevel, min_distance=minDistance, max_corners=maxCorners)
# --- Create DepthAI pipeline -
size = np.array([640, 400])
# --- Create DepthAI pipeline ---
pipeline = dai.Pipeline()
rgbd = pipeline.create(dai.node.RGBD).build(
    True, dai.node.StereoDepth.PresetMode.FAST_ACCURACY, 
    size=tuple(size //4)
)
qRgbd = rgbd.rgbd.createOutputQueue()
print(dir(dai.node.StereoDepth.PresetMode))
# Color map for depth visualization
colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]  # make zero-disparity pixels black
pipeline.start()
init = False
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
twist = None
while True:
    inRgbd = qRgbd.get()  # Get RGB-D frame
    rgb_frame = inRgbd.getRGBFrame()     # HxWx3 RGB image
    depth_frame = inRgbd.getDepthFrame()  # HxW depth in millimeters
    intrinsics_matrix = rgb_frame.getTransformation().getSourceIntrinsicMatrix()
    fx = intrinsics_matrix[0][0]
    fy = intrinsics_matrix[1][1]
    cx = intrinsics_matrix[0][2]
    cy = intrinsics_matrix[1][2]
    rgb_frame = rgb_frame.getCvFrame()
    depth_frame = depth_frame.getCvFrame()/ 1000
    #depth_frame = np.ones_like(depth_frame)
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    if not init : 
        init = True
        tracker.init_tracking(gray)
        points = np.array(tracker.points)[:,0]
        points3d = vs.points2d_to_3d(points,depth_frame, fx,fy,cx,cy)
        vs.set_desired(points3d,None)
    else :
        tracker.track(gray)
        status  = np.array(tracker.status)[:,0]
        points = np.array(tracker.points)[:,0]
        if points is not None:
            for pt in points:
                x, y = pt.ravel()
                cv2.circle(rgb_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        points3d = vs.points2d_to_3d(points,depth_frame, fx,fy,cx,cy)
        twist = vs.servo(points3d,status)
    if twist is not None : 
        # Update histories
        iteration += 1
        iter_hist.append(iteration)
        tx_hist.append(twist[0])
        ty_hist.append(twist[1])
        tz_hist.append(twist[2])
        rx_hist.append(twist[3])
        ry_hist.append(twist[4])
        rz_hist.append(twist[5])

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

    # Normalize depth to 0-255 for visualization
    depth_norm = np.clip(depth_frame/4  * 255, 0, 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    cv2.imshow("RGB Image", rgb_frame)
    cv2.imshow("Depth Image", depth_colored)
    #cv2.imshow("gray",gray)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        init = False
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
