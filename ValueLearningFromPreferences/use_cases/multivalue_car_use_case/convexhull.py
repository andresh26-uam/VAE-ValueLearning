from scipy.spatial import ConvexHull
import numpy as np


def is_dominated(x, y):
    for j in range(len(x)):
        if x[j] > y[j]:
            return False
    return True
    #return (np.asarray(x) <= y).all()

def belongs_to_positive_hull(vertex, hull):

    for vertex2 in hull:
        if not np.array_equal(vertex, vertex2):
            for i in range(len(vertex)):
                if is_dominated(vertex, vertex2):
                    #print(vertex, " is dominated by ", vertex2)
                    return False
    return True


def non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            jiii = (np.asarray(solutions[is_efficient]) <= c).all(axis=1)
            is_efficient[is_efficient] = np.invert(jiii)
            # keep the point itself, otherwise we would get an empty list
            is_efficient[i] = 1

    return solutions[is_efficient]



def get_hull(points, CCS=True, epsilon=0.0):

    aux_needed = False

    vertices = []
    if CCS:
        #for vertex in points:
        #    if belongs_to_positive_hull(vertex, points):
                # print(vertex, " approves positive exam")
        #        vertices.append(vertex)

        #points = vertices

        points = non_dominated(np.array(points))
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


    vertices = non_dominated(np.array(hull_points))

    if CCS and epsilon > 0:
        if len(vertices) > 4:

            allowed_points = np.ones(len(vertices))

            for i in range(len(vertices)):
                for j in range(len(vertices)):
                    for k in range(len(vertices)):
                        if i != j and j != k and i != k and allowed_points[i] and allowed_points[j] and allowed_points[k]:
                            p1 = vertices[i]
                            p2 = vertices[j]
                            p3 = vertices[k]

                            d1 = np.linalg.norm(p3 - p1)
                            d2 = np.linalg.norm(p2 - p1)
                            d3 = np.linalg.norm(p3 - p2)
                            #print(p1, p2, p3, d2 + d3 - d1)

                            if np.abs((d2 + d3) - d1) < epsilon:
                                #print(p1, p2, p3, d2+d3-d1)
                                allowed_points[j] = 0

            new_vertices = list()
            for i in range(len(vertices)):
                if allowed_points[i]:
                    new_vertices.append(vertices[i])

            vertices = new_vertices

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

        hull = np.multiply(hull, gamma, casting="unsafe")

        if len(point) > 0:
            hull = np.add(hull, point, casting="unsafe")

        #for i in range(len(hull)):
        #    hull[i] = np.multiply(hull[i], gamma, casting="unsafe")
        #    if point == []:
        #        pass
        #    else:
        #        hull[i] = np.add(hull[i], point,casting="unsafe")
    return hull




def sum_hulls(hull_1, hull_2, epsilon=0.0):
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

    return get_hull(new_points, epsilon=epsilon)

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

    puntitos = [[  8.625,-10.,0.],
        [  9.5,-25.,0.],
                [9.625,-30,0.]]

    v_function = [[3., 2.],
                  [5., -1.5],
                  [-20., 4.]]

    v_functionE = [[-3., 0.],
                   [4., 2.],
                   [5., 3.]]

    v_function = np.array(v_function)
    v_functionE = np.array(v_functionE)

    vertices = get_hull(puntitos,epsilon=1.0)

    print("resulting vertices")
    print(vertices)
    import matplotlib.pyplot as plt

    plt.plot(vertices[:,0], vertices[:,1], 'k-')
    #max_q_value([1.0,0.4],vertices)
    plt.show()