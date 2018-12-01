import numpy as np 
import os

import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D

joints = [[3, 20], [3, 1], [1, 8], [8, 10], [10, 12], [3, 2], [2, 9], [9, 11], [11, 13],\
        [3, 4], [4, 7], [7, 5], [5, 14], [14, 16], [16, 18], [7, 6], [6, 15], [15, 17], [17, 19]]

def load_data(name):
    lines = open(name).readlines()
    lines = [[float(x) for x in line.strip().split(' ')] \
            for line in lines]
    lines = np.array(lines).reshape([-1, 20, 4]) # 20*frames x 4
    # print(lines.shape) # frames x 20 x 4
    return lines

def load_all(datafolder='../data/'):
    #  video_number x frames x 20 x 4
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
                data[:,x,y] = np.interp(np.linspace(0, old_data.shape[0]-1, max_length), np.arange(old_data.shape[0]), old_data[:,x,y])
        all_data.append(data)
        all_class.append(class_id)
        all_subject.append(subject_id)
    return np.stack(all_data), np.stack(all_class), np.stack(all_subject)

def load_one(datafolder='../data/', label=1, subject=1, instant=1):
    name = "a%02d_s%02d_e%02d_skeleton3D.txt"%(label, subject, instant)
    filename = os.path.join(datafolder, name)
    old_data = load_data(filename)
    max_length = 100
    data = np.zeros([max_length, 20, 4])
    for x in range(20):
        for y in range(4):
            data[:,x,y] = np.interp(np.linspace(0, old_data.shape[0]-1, max_length), np.arange(old_data.shape[0]), old_data[:,x,y])
    return np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))

def animate_skeleton(data_sequence):
    """
    data_sequence: a Frame x 20 x 4
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
    def update_graph(num, points, lines):
        current_frame = data_sequence[num,:,:]
        points.set_data (current_frame[:,0], current_frame[:,1])
        points.set_3d_properties(current_frame[:,2])

        for idx, each in enumerate(lines):
            joint_1 = joints[idx][0] - 1
            joint_2 = joints[idx][1] - 1
            each.set_data(current_frame[(joint_1,joint_2),0], current_frame[(joint_1,joint_2),1])
            each.set_3d_properties(current_frame[(joint_1,joint_2),2])
        title.set_text('3D Test, time={}'.format(num))
        return title, graph


    points, = ax.plot(data_sequence[0,:,0], data_sequence[0,:,1], data_sequence[0,:,2],linestyle="", marker="o")
    lines = []
    for each in joints:
        joint_1 = each[0] - 1
        joint_2 = each[1] - 1
        graph, = ax.plot(data_sequence[0,(joint_1, joint_2),0], data_sequence[0,(joint_1, joint_2),0], data_sequence[0,(joint_1, joint_2),0], c= 'r')
        lines.append(graph)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, 100, fargs= (points, lines),
                               interval=40, blit=False)

    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0, 4.0])
    ax.set_zlabel('Z')
    ax.set_title('3D Test')
    plt.show()

if __name__ == '__main__':
    data = load_one()
    animate_skeleton(np.squeeze(data))