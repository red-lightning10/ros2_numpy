from .registry import converts_from_numpy, converts_to_numpy
from geometry_msgs.msg import Transform, Vector3, Quaternion, Point, Pose
# removed tf_transformations as transformations, due to compatibility issues with numpy>2
from . import numpify
import numpy as np
from scipy.spatial.transform import Rotation as R

# numpy utils
def translation_matrix(translation):
    """ Create a translation matrix """
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return matrix

def quaternion_matrix(quaternion):
    """ Create a rotation matrix from a quaternion """
    r = R.from_quat(quaternion)
    matrix = np.eye(4)
    matrix[:3, :3] = r.as_matrix()
    return matrix

def translation_from_matrix(matrix):
    """ Extract translation from transformation matrix """
    return matrix[:3, 3]

def quaternion_from_matrix(matrix):
    """ Extract quaternion from transformation matrix """
    r = R.from_matrix(matrix[:3, :3])
    return r.as_quat()

# basic types
@converts_to_numpy(Vector3)
def vector3_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg.x, msg.y, msg.z, 0])
    else:
        return np.array([msg.x, msg.y, msg.z])

@converts_from_numpy(Vector3)
def numpy_to_vector3(arr):
    if arr.shape[-1] == 4:
        assert np.all(arr[...,-1] == 0)
        arr = arr[...,:-1]

    if len(arr.shape) == 1:
        return Vector3(**dict(zip(['x', 'y', 'z'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Vector3(**dict(zip(['x', 'y', 'z'], v))), axis=-1,
            arr=arr)

@converts_to_numpy(Point)
def point_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg.x, msg.y, msg.z, 1])
    else:
        return np.array([msg.x, msg.y, msg.z])

@converts_from_numpy(Point)
def numpy_to_point(arr):
    if arr.shape[-1] == 4:
        arr = arr[...,:-1] / arr[...,-1]

    if len(arr.shape) == 1:
        return Point(**dict(zip(['x', 'y', 'z'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Point(**dict(zip(['x', 'y', 'z'], v))), axis=-1, arr=arr)

@converts_to_numpy(Quaternion)
def quat_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])

@converts_from_numpy(Quaternion)
def numpy_to_quat(arr):
    assert arr.shape[-1] == 4

    if len(arr.shape) == 1:
        return Quaternion(**dict(zip(['x', 'y', 'z', 'w'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Quaternion(**dict(zip(['x', 'y', 'z', 'w'], v))),
            axis=-1, arr=arr)


# compound types
# all of these take ...x4x4 homogeneous matrices

@converts_to_numpy(Transform)
def transform_to_numpy(msg):
    return np.dot(
        translation_matrix(numpify(msg.translation)),
        quaternion_matrix(numpify(msg.rotation))
    )

@converts_from_numpy(Transform)
def numpy_to_transform(arr):
    shape, rest = arr.shape[:-2], arr.shape[-2:]
    assert rest == (4,4)

    if len(shape) == 0:
        trans = translation_from_matrix(arr)
        quat = quaternion_from_matrix(arr)
        return Transform(
            translation=Vector3(**dict(zip(['x', 'y', 'z'], trans))),
            rotation=Quaternion(**dict(zip(['x', 'y', 'z', 'w'], quat)))
        )
    else:
        res = np.empty(shape, dtype=np.object_)
        for idx in np.ndindex(shape):
            res[idx] = Transform(
                translation=Vector3(
                    **dict(
                        zip(['x', 'y', 'z'],
                        translation_from_matrix(arr[idx])))),
                rotation=Quaternion(
                    **dict(
                        zip(['x', 'y', 'z', 'w'],
                        quaternion_from_matrix(arr[idx]))))
            )
@converts_to_numpy(Pose)
def pose_to_numpy(msg):
    return np.dot(
        translation_matrix(numpify(msg.position)),
        quaternion_matrix(numpify(msg.orientation))
    )

@converts_from_numpy(Pose)
def numpy_to_pose(arr):
    shape, rest = arr.shape[:-2], arr.shape[-2:]
    assert rest == (4,4)

    if len(shape) == 0:
        trans = translation_from_matrix(arr)
        quat = quaternion_from_matrix(arr)

        return Pose(
            position=Point(**dict(zip(['x', 'y', 'z'], trans))),
            orientation=Quaternion(**dict(zip(['x', 'y', 'z', 'w'], quat)))
        )
    else:
        res = np.empty(shape, dtype=np.object_)
        for idx in np.ndindex(shape):
            res[idx] = Pose(
                position=Point(
                    **dict(
                        zip(['x', 'y', 'z'],
                        translation_from_matrix(arr[idx])))),
                orientation=Quaternion(
                    **dict(
                        zip(['x', 'y', 'z', 'w'],
                        quaternion_from_matrix(arr[idx]))))
            )
