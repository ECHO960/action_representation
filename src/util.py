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
        data = load_data(filename)
        zeros = np.zeros([max_length - data.shape[0], 20, 4])
        data = np.concatenate([data, zeros], axis=0)
        all_data.append(data)
        all_class.append(class_id)
        all_subject.append(subject_id)
    return np.stack(all_data), np.stack(all_class), np.stack(all_subject)


