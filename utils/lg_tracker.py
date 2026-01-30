import torch
import lightglue

class LGTracker :

    def __init__(self) : 
        self.featsd = None
        self.feats = None
        self.extractor = lightglue.SuperPoint(max_num_keypoints=2048,resize = False).eval().cuda()  # load the extractor
        self.matcher = lightglue.LightGlue(features='superpoint',filter_threshold= 0.1).eval().cuda()  # load the matcher
    def init_tracking(self, imaged) : 
        with torch.no_grad():
            self.featsd = self.extractor.extract(imaged)
    def track(self, image):
        with torch.no_grad():
            self.feats = self.extractor.extract(image)
            matches01 = self.matcher({'image0': self.featsd, 'image1': self.feats})
            feats0, feats1, matches01 = [lightglue.utils.rbd(x) for x in [self.featsd, self.feats, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)
            pointsd = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
            points = feats1['keypoints'][matches[..., 1]].cpu().numpy()   # coordinates in image #1, shape (K,2)
            return pointsd, points