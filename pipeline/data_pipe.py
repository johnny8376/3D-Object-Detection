!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar
!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data

import numpy as np
import pandas as pd
import pyquaternion as Quaternion
import lyft
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk import LyftDataset

class LyftDataset(torch.utils.data.Dataset):
    def __init__(self, data:pandas.DataFrame):
        self.data = data
        self.SENSOR = 'LIDAR_TOP'
        self.lyftdata = LyftDataset(data_path='.', json_path='data/', verbose=True)

    @classmethod
    def from_csv(cls,csv):
        train = pd.read_csv(csv)
        return cls(train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        coords, feat = self.get_pointcloud(idx)
        annotations = self.get_annotations(idx)
        return (coords, feat), annotations

    def get_pointcloud(self,idx):
        token = self.get_token(idx)
        sample = self.lyftdata.get('sample',token)
        sensor = self.lyftdata.get('sample_data',sample['data'][self.SENSOR])
        pointcloud = LidarPointCould.from_file(Path(sensor['filename']))
        feat = np.ones((pointcloud.shape[0],1)) if self.intensity == False else pointcloud[:,3]
        return ponintcloud[:,:3], feat
    
    def get_annotaions_from_string(idx):
        columns = ["center_x", "center_y", "center_z", "width", "length", "height", "yaw", "name"]
        token = self.get_token(idx)

        bbox_string = self.data.iloc[idx]['PredictionString'].strip().split(" ")
        new_shape = (int(len(bbox_string) / num_parameters_in_bbox), num_parameters_in_bbox)
        boxes = np.array(bbox_string).reshape(new_shape)
        dft = pd.DataFrame(boxes, columns=columns)
        dft["center_x"] = dft["center_x"].astype(float)
        dft["center_y"] = dft["center_y"].astype(float)
        dft["center_z"] = dft["center_z"].astype(float)
        dft["width"] = dft["width"].astype(float)
        dft["length"] = dft["length"].astype(float)
        dft["height"] = dft["height"].astype(float)
        dft["yaw"] = dft["yaw"].astype(float)
        return dft

    def get_annotations(self,idx):
        token = self.get_token(idx)                
        sample = self.lyftdata.get('sample',token)
        yaw = lambda rotation: Quaternion(rotation).yaw_pitch_roll[0]
        extraction = lambda ann: (ann['translation'],ann['size'],yaw(ann['rotation']))
        annotations = map(extraction,sample['anns'])
        return [*annotations]

    def get_token(self,idx):
        return self.data.iloc[idx]['Id']
