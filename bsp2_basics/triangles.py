from typing import List, Tuple

import numpy
import numpy as np

def define_triangle() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.
    register_number = '12344433'
    register_dict = {i: int(register_number[ord(i) - 65]) for i in 'ABCDEFGH'}

    P1 = np.array([1 + register_dict['C'], - 1 - register_dict['A'], - 1 - register_dict['E']])
    P2 = np.array([- 1 - register_dict['G'], - 1 - register_dict['B'], - 1 - register_dict['H']])
    P3 = np.array([- 1 - register_dict['D'], - 1 - register_dict['F'], - 1 - register_dict['B']])
    ### END STUDENT CODE

    return P1, P2, P3

def define_triangle_vertices(P1:np.ndarray, P2:np.ndarray, P3:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    P1P2 = P2 - P1
    P2P3 = P3 - P2
    P3P1 = P1 - P3
    ### END STUDENT CODE

    return P1P2, P2P3, P3P1

def compute_lengths(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> List[float]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    norms = [np.sqrt(P1P2[0] ** 2 + P1P2[1] ** 2 + P1P2[2] ** 2),
             np.sqrt(P2P3[0] ** 2 + P2P3[1] ** 2 + P2P3[2] ** 2),
             np.sqrt(P3P1[0] ** 2 + P3P1[1] ** 2 + P3P1[2] ** 2)
    ]
    # norms2 = [np.linalg.norm(P1P2), np.linalg.norm(P2P3), np.linalg.norm(P3P1)]
    # if norms != norms2:
    #     raise ValueError('Incorrect lengths')

    ### END STUDENT CODE

    return norms

def compute_normal_vector(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    n = np.cross(P2P3, P1P2)
    n_normalized = n / np.linalg.norm(n)
    ### END STUDENT CODE

    return n, n_normalized

def compute_triangle_area(n:np.ndarray) -> float:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    area = np.linalg.norm(n) / 2
    ### END STUDENT CODE

    return area

def compute_angles(P1P2:np.ndarray,P2P3:np.ndarray,P3P1:np.ndarray) -> Tuple[float, float, float]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    alpha = np.degrees(np.arccos(np.dot(P1P2, -P3P1) / np.linalg.norm(P1P2) / np.linalg.norm(P3P1)))
    beta = np.degrees(np.arccos(np.dot(P2P3, -P1P2) / np.linalg.norm(P1P2) / np.linalg.norm(P2P3)))
    gamma = np.degrees(np.arccos(np.dot(P3P1, -P2P3) / np.linalg.norm(P3P1) / np.linalg.norm(P2P3)))
    ### END STUDENT CODE

    return alpha, beta, gamma

