import numpy as np 
import os

def load_data(name):
    lines = open(name).readlines()
    lines = [[float(x) for x in line.strip().split(' ')] \
            for line in lines]
    lines = np.array(lines).reshape([-1, 20, 4]) # 20*frames x 4
    # print(lines.shape) # frames x 20 x 4
    return lines

def load_all(datafolder='./data/'):
    #  items_num x frames x 20 x 4
    all_data = []
    all_class = []
    all_subject = []
    max_length = 100
    for name in os.listdir(datafolder):
        filename = os.path.join(datafolder, name)
        class_id = int(filename.split('_')[0].split('a')[-1])
        subject_id = int(filename.split('_')[1].split('s')[-1])
        old_data = load_data(filename)
        data = np.zeros([max_length, 20, 4])
        for x in range(20):
            for y in range(4):
                data[:,x,y] = np.interp(np.arange(max_length), np.arange(old_data.shape[0]), old_data[:,x,y])
        # zeros = np.zeros([max_length - data.shape[0], 20, 4])
        # data = np.concatenate([data, zeros], axis=0)
        all_data.append(data)
        all_class.append(class_id)
        all_subject.append(subject_id)
    return np.stack(all_data), np.stack(all_class), np.stack(all_subject)


