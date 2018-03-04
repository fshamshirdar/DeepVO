from PIL import Image
import os
import os.path
import numpy as np
import random
import math
import datetime

import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class VisualOdometryDataLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None,
                 loader=default_image_loader):
        self.base_path = datapath
        self.sequence = '01'
        self.sequence_path = os.path.join(self.base_path, 'sequences', self.sequence)
        self.pose_path = os.path.join(self.base_path, 'poses')

        self.timestamp = self.load_timestamps()
        self.poses = self.load_poses(self.sequence)

        self.transform = transform
        self.loader = loader
        self.size = len(self.timestamp)

    def load_poses(self, sequence):
        with open(os.path.join(self.base_path, 'poses/',  sequence + '.txt')) as f:
            poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
        return poses

    def load_timestamps(self):
        timestamp_file = os.path.join(self.sequence_path, 'times.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                t = datetime.timedelta(seconds=float(line))
                timestamps.append(t)
        return timestamps

    def get_image(self, index):
        image_path = os.path.join(self.sequence_path, 'image_2', '%06d' % index + '.png')
        return self.loader(image_path)

    def __getitem__(self, index):
        # img1 = self.loader(os.path.join(self.base_path, anchor_data_index, anchor_path))
        img1 = self.get_image(index)
        img2 = self.get_image(index+1)
        odom = self.get_ground_6d_poses(self.poses[index])
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, odom

    def __len__(self):
        return self.size-1

    def isRotationMatrix(self, R):
        """ Checks if a matrix is a valid rotation matrix
            referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        """
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        """ calculates rotation matrix to euler angles
            referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
        """
        assert(self.isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if  not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z], dtype=np.float32)

    def get_ground_6d_poses(self, p):
        """ For 6dof pose representaion """
        pos = np.array([p[3], p[7], p[11]])
        R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
        angles = self.rotationMatrixToEulerAngles(R)
        return np.concatenate((pos, angles))

if __name__ == "__main__":
    db = VisualOdometryDataLoader("/data/KITTI/dataset/")
    img1, img2, odom = db[1]
    print (odom)

    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(img1)
    axarr[0,1].imshow(img2)
    plt.show()
