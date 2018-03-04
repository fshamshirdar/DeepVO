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
        # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        self.sequences = ['01']

        # self.timestamps = self.load_timestamps()
        self.size = 0
        self.sizes = []
        self.poses = self.load_poses()

        self.transform = transform
        self.loader = loader

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/',  sequence + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)

                self.size = self.size + len(poses)
                self.sizes.append(len(poses))
        return all_poses

    """ 
    def load_timestamps(self, sequence_path):
        for sequence in self.sequences:
            timestamp_file = os.path.join(self.sequence_path, 'times.txt')

            # Read and parse the timestamps
            timestamps = []
            with open(timestamp_file, 'r') as f:
                for line in f.readlines():
                    t = datetime.timedelta(seconds=float(line))
                    timestamps.append(t)
            return timestamps
    """

    def get_image(self, sequence, index):
        image_path = os.path.join(self.base_path, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
        return self.loader(image_path)

    def __getitem__(self, index):
        sequence = 0
        sequence_size = 0
        for size in self.sizes:
            if index < size-1:
                sequence_size = size
                break
            index = index - (size-1)
            sequence = sequence + 1

        next_index = random.randint(index-10,index+10)
        if next_index > size-1 or next_index < 0:
            next_index = index+1

        img1 = self.get_image(self.sequences[sequence], index)
        img2 = self.get_image(self.sequences[sequence], next_index)
        pose1 = self.get_ground_6d_poses(self.poses[sequence][index])
        pose2 = self.get_ground_6d_poses(self.poses[sequence][next_index])
        odom = pose2 - pose1
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, odom

    def __len__(self):
        return self.size-len(self.sequences)

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
