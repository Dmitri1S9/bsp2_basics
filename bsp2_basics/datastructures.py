from typing import Tuple
import numpy as np
    
def define_structures() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Defines the two vectors v1 and v2 as well as the matrix M determined by your matriculation number.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.
    register_number = '12344433'
    register_dict = {i: int(register_number[ord(i) - 65]) for i in 'ABCDEFGH'}

    v1 = np.array((register_dict['D'], register_dict['A'], register_dict['C']))
    v2 = np.array((register_dict['F'], register_dict['B'], register_dict['E']))
    M = np.array([
        [register_dict['D'], register_dict['B'], register_dict['C']],
        [register_dict['B'], register_dict['G'], register_dict['A']],
        [register_dict['E'], register_dict['H'], register_dict['F']]
    ])

    ### END STUDENT CODE

    return v1, v2, M

def sequence(M : np.ndarray) -> np.ndarray:
    """
        Defines a vector given by the minimum and maximum digit of your matriculation number. Step size = 0.25.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.
    min_in_matrix = np.min(M)
    max_in_matrix = np.max(M)

    result = np.arange(min_in_matrix, max_in_matrix + 0.25, 0.25)

    ### END STUDENT CODE


    return result

def matrix(M : np.ndarray) -> np.ndarray:
    """
        Defines the 15x9 block matrix as described in the task description.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    r = np.zeros((15,9), dtype=int)
    for i in range(15):
        for j in range(9):
            if (i // 3 % 2 == 0) == (j // 3 % 2 == 0):
                r[i,j] = M[i % 3, j % 3]

    ### END STUDENT CODE

    return r


def dot_product(v1:np.ndarray, v2:np.ndarray) -> float:
    """
        Dot product of v1 and v2.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    r = sum([v1[i] * v2[i] for i in range(3)])
    # r = np.dot(v1, v2)

    ### END STUDENT CODE

    return r

def cross_product(v1:np.ndarray, v2:np.ndarray) -> np.ndarray:
    """
        Cross product of v1 and v2.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    r = np.array((
        (v1[1] * v2[2]) - (v1[2] * v2[1]),
        (v1[2] * v2[0]) - (v1[0] * v2[0]),
        (v1[0] * v2[1]) - (v1[1] * v2[0])
    ))
    # r = np.cross(v1, v2)
    ### END STUDENT CODE

    return r

def vector_X_matrix(v:np.ndarray, M:np.ndarray) -> np.ndarray:
    """
        Defines the vector-matrix multiplication v*M.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.
    r = np.array((
        v[0] * M[0, 0] + v[1] * M[1, 0] + v[2] * M[2, 0],
        v[0] * M[0, 1] + v[1] * M[1, 1] + v[2] * M[2, 1],
        v[0] * M[0, 2] + v[1] * M[1, 2] + v[2] * M[2, 2]
    ))
    # r = v @ M
    ### END STUDENT CODE

    return r

def matrix_X_vector(M:np.ndarray, v:np.ndarray) -> np.ndarray:
    """
        Defines the matrix-vector multiplication M*v.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.
    r = np.array((
        M[0, 0] * v[0] + M[0, 1] * v[1] + M[0, 2] * v[2],
        M[1, 0] * v[0] + M[1, 1] * v[1] + M[1, 2] * v[2],
        M[2, 0] * v[0] + M[2, 1] * v[1] + M[2, 2] * v[2]
    ))
    # r = M @ v
    ### END STUDENT CODE

    return r

def matrix_X_matrix(M1:np.ndarray, M2:np.ndarray) -> np.ndarray:
    """
        Defines the matrix multiplication M1*M2.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.
    r = np.array([
        [
            M1[0, 0] * M2[0, 0] + M1[0, 1] * M2[1, 0] + M1[0, 2] * M2[2, 0],
            M1[0, 0] * M2[0, 1] + M1[0, 1] * M2[1, 1] + M1[0, 2] * M2[2, 1],
            M1[0, 0] * M2[0, 2] + M1[0, 1] * M2[1, 2] + M1[0, 2] * M2[2, 2]
        ],
        [
            M1[1, 0] * M2[0, 0] + M1[1, 1] * M2[1, 0] + M1[1, 2] * M2[2, 0],
            M1[1, 0] * M2[0, 1] + M1[1, 1] * M2[1, 1] + M1[1, 2] * M2[2, 1],
            M1[1, 0] * M2[0, 2] + M1[1, 1] * M2[1, 2] + M1[1, 2] * M2[2, 2]
        ],
        [
            M1[2, 0] * M2[0, 0] + M1[2, 1] * M2[1, 0] + M1[2, 2] * M2[2, 0],
            M1[2, 0] * M2[0, 1] + M1[2, 1] * M2[1, 1] + M1[2, 2] * M2[2, 1],
            M1[2, 0] * M2[0, 2] + M1[2, 1] * M2[1, 2] + M1[2, 2] * M2[2, 2]
        ]
    ])
    # r = M1 @ M2
    ### END STUDENT CODE

    return r

def matrix_Xc_matrix(M1:np.ndarray, M2:np.ndarray) -> np.ndarray:
    """
        Defines the element-wise matrix multiplication M1*M2 (Hadamard Product).
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.
    r = np.array([
        [M1[0, 0] * M2[0, 0], M1[0, 1] * M2[0, 1], M1[0, 2] * M2[0, 2]],
        [M1[1, 0] * M2[1, 0], M1[1, 1] * M2[1, 1], M1[1, 2] * M2[1, 2]],
        [M1[2, 0] * M2[2, 0], M1[2, 1] * M2[2, 1], M1[2, 2] * M2[2, 2]]
    ])
    # r = M1 * M2
    ### END STUDENT CODE
    

    return r


