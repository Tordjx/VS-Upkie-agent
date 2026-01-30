

import cv2
import depthai as dai
import numpy as np
import visp as vp

me = vp.me.Me()
def create_line_tracker(): 
    opt_me_range = 20
    opt_me_sample_step = int(3)
    opt_me_threshold = 50
    me.setRange(opt_me_range)
    #me.setThreshold(opt_me_threshold)
    me.setSampleStep(opt_me_sample_step)
    line = vp.me.MeLine()
    line.setMePy(me)
    return  line
def abc_to_points(A,B,C, height, width):
    if (abs(A) < abs(B)) :
        i1 = 0
        j1 = (-A * i1 - C) / B
        i2 = height - 1.0
        j2 = (-A * i2 - C) / B
    
    else :  
        j1 = 0
        i1 = -(B * j1 + C) / A
        j2 = width - 1.0
        i2 = -(B * j2 + C) / A

    return[int(j1),int(i1)] ,[int(j2),int(i2)]
# --- Create DepthAI pipeline ---
pipeline = dai.Pipeline()
rgbd = pipeline.create(dai.node.RGBD).build(
      True, dai.node.StereoDepth.PresetMode.ACCURACY, 
      size=(640, 400)
)
qRgbd = rgbd.rgbd.createOutputQueue()

# Color map for depth visualization
colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]  # make zero-disparity pixels black
pipeline.start()
vs = None
init = False
line = create_line_tracker()
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
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9,9), 0)
    if not init : 
        init = True
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=20)

        # pick the longest line
        longest_line = max(lines, key=lambda l: np.linalg.norm([l[0][0]-l[0][2], l[0][1]-l[0][3]]))
        x1, y1, x2, y2 = longest_line[0]

        p1 = vp.core.ImagePoint(y1,x1)
        p2 = vp.core.ImagePoint(y2,x2)
        print(p1,p2)
        line.initTracking(vp.core.ImageGray(gray), p1, p2)
    else :
        line.track(vp.core.ImageGray(gray))
        abc = line.get_ABC()
        A,B,C = abc[0],abc[1],abc[2]
        height, width = gray.shape
        p1,p2 = abc_to_points(A,B,C, height, width)
        cv2.line(
            gray,
            p1, p2,
            color=(0, 255, 0),  # green line
            thickness=2
        )
    # Normalize depth to 0-255 for visualization
    depth_norm = np.clip(depth_frame  * 255, 0, 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    #cv2.imshow("RGB Image", rgb_frame)
    #cv2.imshow("Depth Image", depth_colored)
    cv2.imshow("gray",gray)
    cv2.imshow('canny',edges)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
