import util
import numpy as np
root_bone = (2,19)
def bone_hoerarchy_from_joints(joints, root_bone):
    #Joints hierarchy map
    hierarchy_map = {}
    for parent_joint, child_joint in joints:
        hierarchy_map[child_joint] = parent_joint

    #Bone hierarchy map
    hierarchy = {}
    for parent_joint, child_joint in joints:
        if (parent_joint, child_joint) == root_bone:
            hierarchy[root_bone] = None
        elif parent_joint not in hierarchy_map:
            hierarchy[(parent_joint, child_joint)] = (root_bone[1], root_bone[0])
        else:
            hierarchy[(parent_joint, child_joint)] = (hierarchy_map[parent_joint], parent_joint)
    return hierarchy


def axis_angle_transform(data_sequence):
    """
        data_sequence: Number of Actions x Frame x 20 x 4

        util.joints: list of [parent_node, child_node]
    """
    bone_length = {}
    hierarchy = bone_hoerarchy_from_joints(util.joints, root_bone)
    all_rotations = []
    all_axis = []
    correspondance = {}

    
    for parent_joint, child_joint in util.joints:
        if hierarchy[(parent_joint, child_joint)] is None: #This is root vector, we should store this
            root_joints_pair = (data_sequence[:,:,child_joint,:] - data_sequence[:,:,parent_joint,:])
        else:
            parent_bone = hierarchy[(parent_joint, child_joint)]
            children_bone = (parent_joint, child_joint)

            parent_vector = data_sequence[:,:,parent_bone[0],:] - data_sequence[:,:,parent_bone[1],:]
            children_vector = data_sequence[:,:,children_bone[1],:] - data_sequence[:,:,children_bone[0],:]

            axis_of_rotation = np.cross(parent_vector, children_vector)
            axis_of_rotation = axis_of_rotation/np.sqrt(np.sum(axis_of_rotation ** 2, axis =2,keepdims=True)) #normalize axis

            angle_of_rotation = np.sum(parent_vector * children_vector, axis = 2, keepdims = True) #dot product
            children_bone_magnitude = np.sqrt(np.sum(children_vector ** 2, axis = 2, keepdims = True))
            parent_bone_magnitude = np.sqrt(np.sum(parent_vector ** 2, axis = 2, keepdims = True))
            magnitude =  parent_bone_magnitude * children_bone_magnitude
            angle_of_rotation = angle_of_rotation/magnitude
            angle_of_rotation = np.arccos(angle_of_rotation)
            all_rotations.append(angle_of_rotation)
            all_axis.append(axis_of_rotation)


            bone_length[parent_bone] = np.mean(parent_bone_magnitude)
            bone_length[children_bone] = np.mean(children_bone_magnitude)

            correspondance[(parent_joint, child_joint)] = len(all_axis) - 1

    return root_joints_pair, np.array(all_axis), np.array(all_rotations), bone_length, correspondance

# def transform_arccos(data_sequence):
#     """
#         data_sequence: Number of Actions x Frame x 20 x 4

#         util.joints: list of [parent_node, child_node]
#     """
#     root_joints_pair, axis_of_rotation, angle_of_rotation, bone_length = transform(data_sequence)
#     return root_joints_pair, axis_of_rotation, np.arccos(angle_of_rotation), bone_length

def axis_angle_transform_rev(root_joints_pair, data_sequence, bone_length, correspondance, origin = np.zeros(3)):
    """
        root_joints_pair: the root joints (frame, 3), roog joints is root node - its children node

        data_sequence: (number of bones, time frame, 3)

        the bone_length

        util.joints: list of [parent_node, child_node]
    """
    total_frame = data_sequence.shape[1]
    hierarchy = bone_hoerarchy_from_joints(util.joints, root_bone)

    joint_pairs_to_visit = [(root_bone[1], root_bone[0])]
    
    second_root_joints = origin + root_joints_pair
    
    reconstructed_points = {root_bone[0]:np.tile(origin, (100,1)), root_bone[1]:second_root_joints}

    while len(joint_pairs_to_visit):

        current_bone = joint_pairs_to_visit.pop()
        print(reconstructed_points)
        for parent_joint, child_joint in util.joints:
            if hierarchy[(parent_joint, child_joint)] == current_bone:
                rotation_origin = reconstructed_points[current_bone[1]]
                rotation_point = reconstructed_points[current_bone[0]]

                rotation_point_normalize = rotation_point - rotation_origin

                rotation_vector = data_sequence[correspondance[(parent_joint, child_joint)], :, :]

                rotated_point_normalize = np.zeros(rotation_point.shape)
                for frame in range(total_frame):
                    single_rotateion_vector = rotation_vector[frame,:]
                    R = rodrigues(single_rotateion_vector.reshape(3,1))
                    rotated_point_normalize[frame,:] = R.dot(rotation_point_normalize[frame,:])

                rotated_point_normalize = rotated_point_normalize/np.sqrt(np.sum(rotated_point_normalize ** 2, axis = 1, keepdims = True)) *\
                                          bone_length[(parent_joint, child_joint)]
                rotated_point = rotated_point_normalize + rotation_origin

                reconstructed_points[child_joint] = rotated_point

                joint_pairs_to_visit.append((parent_joint, child_joint))

    all_points = np.zeros((data_sequence.shape[1], 20, 3))
    for i in range(20):
        all_points[:,i,:] = reconstructed_points[i]

    return all_points

def rodrigues(r):
    theta = np.linalg.norm(r)
    if np.isclose(theta,0):
        R = np.eye(3)
    else:
        u = r/theta
        ux = np.array([[0, -u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1],u[0], 0]])
        R = np.eye(3)*np.cos(theta) + (1-np.cos(theta)) * u.dot(u.T) + np.sin(theta) * ux
    return R


# def reverse_transform(data_sequence):
# all_data = []
# for i in [1,2,3,5,6,7,8,9,10]:
#     for j in range(1,4):
#         data = util.load_one('../data/', 2, i, j)
#         all_data.append(data)
# all_data = np.squeeze(np.array(all_data))

# print(bone_hoerarchy_from_joints(util.joints, (2,19)))    
# root_joints, all_axis, all_rotations, bone_lengtha = axis_angle_transform(all_data)
# print(root_joints.shape)
# print(all_axis.shape)
# print(all_rotations.shape)
# print((all_axis * all_rotations).shape)
if __name__ == '__main__':
    



data = util.load_one()

root_joints_pair, all_axis, all_rotations, bone_length, correspondance = axis_angle_transform(data)

data_sequence = all_axis * all_rotations
data_sequence = np.mean(data_sequence,axis = 1)
print(data_sequence.shape)

all_points = axis_angle_transform_rev(np.squeeze(root_joints_pair), data_sequence, bone_length, correspondance)

print(all_points.shape)
util.animate_skeleton(all_points)