from scipy.spatial import ConvexHull
import numpy as np


def is_dominated(x, y):
    for j in range(len(x)):
        if x[j] > y[j]:
            return False
    return True


def belongs_to_positive_hull(vertex, hull):

    for vertex2 in hull:
        if not np.array_equal(vertex, vertex2):
            for i in range(len(vertex)):
                if is_dominated(vertex, vertex2):
                    #print(vertex, " is dominated by ", vertex2)
                    return False
    return True


def get_hull(points, CCS=True):

    aux_needed = False

    vertices = []
    for vertex in points:

        if CCS:
            if belongs_to_positive_hull(vertex, points):
                # print(vertex, " approves positive exam")
                vertices.append(vertex)

    points = vertices

    if aux_needed:
        # Compute  auxiliary points
        dim_maxs = np.max(points, axis=0)
        for i in range(len(dim_maxs)):
            aux_point = -100000*np.ones(len(dim_maxs))
            aux_point[i] = dim_maxs[i]

            # Add auxiliary points
            points = np.append(points, np.array([aux_point]), axis=0)

    # Compute new hull
    try:
        hull = ConvexHull(points)
        hull_points = [points[vertex] for vertex in hull.vertices]
    except:
        hull_points = points

    # Remove auxiliary points of new hull
    vertices = []
    for vertex in hull_points:

        if CCS:
            if belongs_to_positive_hull(vertex, hull_points):
                # print(vertex, " approves positive exam")
                vertices.append(vertex)
        else:
            if np.linalg.norm(vertex) < 10000:
                vertices.append(vertex)

    return np.array(vertices)



def translate_hull(point, gamma, hull):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Translation and scaling operation of convex hulls (definition 1 of the paper).

    :param point: a 2-D numpy array
    :param gamma: a real number
    :param hull: a set of 2-D points, they need to be numpy arrays
    :return: the new convex hull, a new set of 2-D points
    """

    if len(hull) == 0:
        hull = [point]
    else:
        for i in range(len(hull)):
            hull[i] = np.multiply(hull[i], gamma, casting="unsafe")
            if point == []:
                pass
            else:
                hull[i] = np.add(hull[i], point,casting="unsafe")
    return hull


def sum_hulls(hull_1, hull_2):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Sum operation of convex hulls (definition 2 of the paper)

    :param hull_1: a set of 2-D points, they need to be numpy arrays
    :param hull_2: a set of 2-D points, they need to be numpy arrays
    :return: the new convex hull, a new set of 2-D points
    """
    if len(hull_1) == 0:
        return hull_2
    elif len(hull_2) == 0:
        return hull_1

    new_points = None

    for i in range(len(hull_1)):
        if new_points is None:
            new_points = translate_hull(hull_1[i].copy(), 1,  hull_2.copy())
        else:
            new_points = np.concatenate((new_points, translate_hull(hull_1[i].copy(), 1, hull_2.copy())), axis=0)

    return get_hull(new_points)

def max_q_value(weight, hull):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Extraction of the Q-value (definition 3 of the paper)

    :param weight: a weight vector, can be simply a list of floats
    :param hull: a set of 2-D points, they need to be numpy arrays
    :return: a real number, the best Q-value of the hull for the given weight vector
    """
    scalarised = []

    for i in range(len(hull)):
        f = np.dot(weight,hull[i])
        #print(f)
        scalarised.append(f)

    scalarised = np.array(scalarised)

    return np.max(scalarised)


if __name__ == "__main__":

    points = np.random.rand(42, 2)   # 15 random points in 2-D

    puntitos = -1*np.array([[0, 0, 4], [0, 5, 3], [1, 7, 0], [2, 1, 4], [3, 4, 5], [4, 2, 3], [4, 4, 6], [4, 6, 7], [5, 0, 2], [6, 4, 1], [6, 5, 1], [6, 7, 0], [7, 4, 3]])
    print(puntitos)

    v_function = [[3., 2.],
                  [5., -1.5],
                  [-20., 4.]]

    v_functionE = [[-3., 0.],
                   [4., 2.],
                   [5., 3.]]

    v_function = np.array(v_function)
    v_functionE = np.array(v_functionE)

    vertices = get_hull(puntitos)

    print("resulting vertices")
    print(vertices)
    import matplotlib.pyplot as plt

    plt.plot(vertices[:,0], vertices[:,1], 'k-')
    #max_q_value([1.0,0.4],vertices)
    plt.show()
