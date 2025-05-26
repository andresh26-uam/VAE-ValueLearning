from scipy.spatial import ConvexHull
import numpy as np


def get_hull(points):
    """

    Get_hull calculates the positive convex hull of a set of points, limiting it to only consider weights of the form
    (1, x, x) with x >= 0. If the number of points is too small to calculate the convex hull, the program will simply
    return the original points.

    :param points: set of 2-D points, they need to be numpy arrays
    :return: new set of 2-D points, the vertices of the calculated convex hull
    """
    try:
        hull = ConvexHull(points)
        vertices = []
        for vertex in hull.vertices:
            #print(points[vertex])
            vertices.append(points[vertex])

        vertices = np.array(vertices)

        best_individual = np.argmax(vertices[:, 0])

        #Calculating best ethical
        best_ethical = -1
        chosen_ethical = np.max(vertices[:, 1])

        where_ethical = np.argwhere(vertices[:, 1] == chosen_ethical)[:, 0]
        chosen_individual = np.max(vertices[where_ethical][:, 0])

        for i in range(len(vertices)):
            if vertices[i][0] == chosen_individual and vertices[i][1] == chosen_ethical:
                best_ethical = i

        #print(best_individual, best_ethical)

        if best_ethical < best_individual:
            vertices = np.concatenate((vertices[best_individual:], vertices[:best_ethical+1]),0)
        else:
            vertices = vertices[best_individual:best_ethical + 1]

        #print()
        #print(vertices)
        return vertices
    except:
        return points


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
    points = np.random.rand(15, 2)   # 15 random points in 2-D

    vertices = get_hull(points)

    import matplotlib.pyplot as plt

    plt.plot(vertices[:,0], vertices[:,1], 'k-')
    max_q_value([1.0,0.4],vertices)
    plt.show()
