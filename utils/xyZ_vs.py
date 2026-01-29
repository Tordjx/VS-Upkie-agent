import visp as vp
import numpy as np

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
        if status is None : 
            status = np.ones(points3d.shape[0])
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