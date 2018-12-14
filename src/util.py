import numpy as np 
import os
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D

joints = [[2, 19], [2, 0], [0, 7], [7, 9], [9, 11], [2, 1], [1, 8], [8, 10], [10, 12],\
        [2, 3], [3, 6], [6, 4], [4, 13], [13, 15], [15, 17], [6, 5], [5, 14], [14, 16], [16, 18]]
length = [0.26058642, 0.20741401, 0.26374705, 0.2848494,  0.06721901, 0.20688813, \
        0.26339344, 0.28399782, 0.06754426, 0.2775788, 0.21709026, 0.18817224, \
        0.57511103,0.42697567, 0.16832953, 0.18817167, 0.57247651, 0.42698327, 0.16832914]

def load_data(name):
    lines = open(name).readlines()
    lines = [[float(x) for x in line.strip().split(' ')] \
            for line in lines]
    lines = np.array(lines).reshape([-1, 20, 4]) # 20*frames x 4
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
    all_data = preProcess(np.stack(all_data))
    return all_data, np.stack(all_class), np.stack(all_subject)

def load_one(datafolder='../data/', label=1, subject=1, instant=1):
    name = "a%02d_s%02d_e%02d_skeleton3D.txt"%(label, subject, instant)
    filename = os.path.join(datafolder, name)
    old_data = load_data(filename)
    max_length = 100
    data = np.zeros([max_length, 20, 4])
    for x in range(20):
        for y in range(4):
            data[:,x,y] = np.interp(np.linspace(0, old_data.shape[0]-1, max_length), np.arange(old_data.shape[0]), old_data[:,x,y])
    data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
    data = preProcess(data)
    return data

def preProcess(data):
    #  video_number x frames x 20 x 4
    # length normalization
    data = data[:,:,:,0:3]
    new_data = np.copy(data)
    global length 
    for video in range(data.shape[0]):
        for frame in range(data.shape[1]):
            for i, pair in enumerate(joints):
                vector = data[video, frame, pair[1], :] - data[video, frame, pair[0], :]
                vector = vector / np.sqrt(np.square(vector).sum()) * length[i]
                new_data[video, frame, pair[1], :] = new_data[video, frame, pair[0], :] + vector
    # position normalization
    for video in range(data.shape[0]):
        for frame in range(data.shape[1]):
            new_data[video, frame, :, :] -= new_data[video, frame, 2, :]
    return new_data[:,:,:,[0,2,1]]

def animate_skeleton(data_sequence, save_file = None, caption = '3D Test'):
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
            joint_1 = joints[idx][0]
            joint_2 = joints[idx][1]
            each.set_data(current_frame[(joint_1,joint_2),0], current_frame[(joint_1,joint_2),1])
            each.set_3d_properties(current_frame[(joint_1,joint_2),2])
        title.set_text('{}, time={}'.format(caption, num))
        return title, graph


    points, = ax.plot(data_sequence[0,:,0], data_sequence[0,:,1], data_sequence[0,:,2],linestyle="", marker="o")
    lines = []
    for each in joints:
        joint_1 = each[0]
        joint_2 = each[1]
        graph, = ax.plot(data_sequence[0,(joint_1, joint_2),0], data_sequence[0,(joint_1, joint_2),0], data_sequence[0,(joint_1, joint_2),0], c= 'r')
        lines.append(graph)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, 100, fargs= (points, lines),
                               interval=40, blit=False)


    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-1.5, .5])
    ax.set_zlabel('Z')

    if save_file is not None:
        ani.save(save_file)
    plt.show()

def static_skeleton(data_sequence, stride = 10,offset = np.array([0.5, 0.5, 0])):
    """
    Generate a bunch of static skeleton plot
    stride: for every stride frame display a skeleton
    offset: for every stride frame, move the skeleton by offset amount accroding to x, y, z
    """
    totla_frame = data_sequence.shape[0]
    num_of_skeleton = int(totla_frame/stride)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d([-1.5 + offset[0]/2 * num_of_skeleton, 1.0 + offset[0]/2 * num_of_skeleton])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.5 + offset[0]/2 * num_of_skeleton, 1.0 + offset[0]/2 * num_of_skeleton])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-1.5, .5])
    ax.set_zlabel('Z')
    ax.set_title('3D Test')
    for idx,j in enumerate(range(0, totla_frame, stride)):
        current_offset = idx * offset
        data_sequence_offset = data_sequence[j,:,:] + current_offset
        ax.plot(data_sequence_offset[:,0], data_sequence_offset[:,1], data_sequence_offset[:,2],linestyle="", marker="o",c='b')
        for each in joints:
            joint_1 = each[0]
            joint_2 = each[1]
            graph, = ax.plot(data_sequence_offset[(joint_1, joint_2),0], \
                            data_sequence_offset[(joint_1, joint_2),1], \
                            data_sequence_offset[(joint_1, joint_2),2], c= 'r')
    ax.grid(False)
    # # and later in the code:
    # ax.get_proj = make_get_proj(ax, 1.2, 1.2, 1)
    # ax.set_aspect(1.0)

    plt.show()

if __name__ == '__main__':
    data = load_one(label=1, subject=1, instant=1)
    # animate_skeleton(np.squeeze(data)[50:])
    static_skeleton(np.squeeze(data)[50:])