import visp as vp
import numpy as np

class PolarVS:
    def __init__(self, lambda_gain=0.2):
        self.lambda_gain = lambda_gain

        self.p = []      # current features
        self.pd = []     # desired features
        self.dof = vp.core.ColVector([1,0,0,0,0,1])
    def servo(self,points_polar, points_polar_desired):
        self.task = vp.vs.Servo()
        self.task.setCameraDoF(self.dof)
        self.task.setServo(vp.vs.Servo.EYEINHAND_CAMERA)
        self.task.setInteractionMatrixType(vp.vs.Servo.CURRENT)
        self.task.setLambda(self.lambda_gain)
        if len(points_polar)< 4 :
            return None
        for i in range(len(points_polar)):
            rho, theta, z = points_polar[i]
            rhod, thetad, zd = points_polar_desired[i]
            pd = vp.visual_features.FeaturePointPolar()
            pd.buildFrom(rhod, thetad, zd)

            p = vp.visual_features.FeaturePointPolar()
            p.buildFrom(rho, theta, z)

            self.task.addFeature(p, pd)
            self.p.append(p)
            self.pd.append(pd)

        return self.task.computeControlLaw()
    def points2polar(self , points , depth, fx,fy , cx, cy) :
        polar = []
        for (u, v) in points:
            u,v = int(u) , int(v)
            z = depth[v,u]
            if z > 0 :
                x = (u - cx) / fx
                y = (v - cy) / fy
                rho = np.sqrt(x*x + y*y)
                theta = np.arctan2(y, x)
                polar.append((rho, theta, z))
        return polar

