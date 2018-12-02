import numpy as np 
from util import joints

def relative_joint_transform(data):
    # data numpy: video_number x frames x 20 x 3
    new_data = np.copy(data)
    for video in range(data.shape[0]):
        for frame in range(data.shape[1]):
            for i, pair in enumerate(joints):
                vector = data[video, frame, pair[1], :] - data[video, frame, pair[0], :]
                new_data[video, frame, pair[1], :] = vector
    return new_data

def relative_joint_transform_rev(data):
    new_data = np.copy(data)
    for video in range(data.shape[0]):
        for frame in range(data.shape[1]):
            for i, pair in enumerate(joints):
                vector = new_data[video, frame, pair[0], :] + data[video, frame, pair[1], :] 
                new_data[video, frame, pair[1], :] = vector
    return new_data

def relative_time_transform(data):
    new_data = np.copy(data)
    for video in range(data.shape[0]):
        for frame in range(1, data.shape[1]):
            vector = data[video, frame, :, :] - data[video, frame - 1, :, :]
            new_data[video, frame, :, :] = vector
    return new_data 

def relative_time_transform_rev(data):
    new_data = np.copy(data)
    for video in range(data.shape[0]):
        for frame in range(1, data.shape[1]):
            vector = data[video, frame, :, :] + new_data[video, frame - 1, :, :]
            new_data[video, frame, :, :] = vector
    return new_data


