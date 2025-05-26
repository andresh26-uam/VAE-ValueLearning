import numpy as np
import itertools

def get_lex_coverage_set_from_hull(hull, lexes):

    lex_set = list()

    for l in lexes:
        lex_set.append(lex_max(hull, l))

    return np.unique(lex_set, axis=0)


def get_lex_set_from_hull(hull, lexes):
    lex_set = list()

    for l in lexes:
        lex_set.append(lex_max(hull, l))

    return lex_set


def get_convex_hull(n_obstacles, n_pedestrians, n_stochastic):
    return eval("ch_" + str(n_obstacles) + "_" + str(n_pedestrians) + "_" + str(n_stochastic))


def reverse_lex(lex):

    def find_n(n, lex):

        for i in range(len(lex)):
            if n == lex[i]:
                return i

    return [find_n(i, lex) for i in range(len(lex))]


def lex_max(hull, lex, iWantIndex=False):

    hull = np.array(hull)

    ordered_hull = hull[:, lex]
    antilex = reverse_lex(lex)

    for i in range(len(lex)):
        first_max = np.max(ordered_hull, axis=0)[i]
        next_hull = []

        for j in ordered_hull:
            if j[i] >= first_max:
                next_hull.append(j)

        ordered_hull = next_hull

    lex_point = np.array(ordered_hull[0])
    lex_point = lex_point[antilex]

    if iWantIndex:
        for i in range(len(hull)):
            its_the_same = True
            for j in range(len(lex_point)):
                its_the_same *= lex_point[j] == hull[i][j]

            if its_the_same:
                return i

    return lex_point


ch_1_1_0 = [[  8.,   0.,   0.],
 [  9.,   0., -10.],
 [ 10., -10.,   0.]]

ch_1_2_0 = [[  6.,   0.,   0.],
 [  9.,   0., -10.],
 [ 10., -10.,   0.]]

ch_1_1_1 = [[  8.25,   0.,     0. ],
 [  9.,     0.,    -7.5 ],
 [  9.25, -10.,     0.  ],
 [ 10.,   -10.,    -7.5 ]]

ch_1_2_1 = [[  7.,     0.,     0.  ],
 [  7.75,   0.,    -2.5 ],
 [  8.25,  -5.,     0.  ],
 [  8.25,   0.,    -5.  ],
 [  8.75, -10.,     0.  ],
 [  9.,     0.,   -10.],
 [  9.5,  -10.,    -2.5 ],
 [ 10.,   -10.,    -5.]]

ch_1_2_2 = [[  7.5625,   0.,       0.  ],
 [  8.125,    0.,      -1.875 ],
 [  8.375,   -5.,       0.   ],
 [  8.375,    0.,      -3.125 ],
 [  8.75,     0.,      -5.625 ],
 [  9.,       0.,      -7.5   ],
 [  9.125,  -10.,       0.    ],
 [  9.5,    -10.,      -1.25  ],
 [  9.75,   -10.,      -2.5   ],
 [ 10.,     -10.,      -5.    ]]

ch_2a_1_0 = [[  7.,   0.,   0.],
 [  8.,   0., -10.],
 [ 10., -10.,   0.]]

ch_2a_2_0 = [[  5.,   0.,   0.],
 [  8.,   0., -10.],
 [ 10., -10.,   0.]]

ch_2a_1_1 = [[  7.25,   0.,     0.  ],
 [  8.,     0.,    -7.5 ],
 [  9.25, -10.,     0.  ],
 [ 10.,   -10.,    -7.5 ]]

ch_2a_2_1 = [[  6.5,    0.,     0.  ],
 [  7.25,   0.,    -2.5 ],
 [  8.,     0.,   -10.  ],
 [  8.25,  -5.,     0.  ],
 [  8.75, -10.,     0.  ],
 [  9.5, -10.,    -2.5 ],
 [ 10.,   -10.,    -5.  ]]

ch_2a_2_2 = [[  6.9375,   0.,       0.,    ],
 [  7.3125,   0.,      -1.25  ],
 [  7.625,    0.,      -3.75  ],
 [  7.8125,  -2.5,      0.,    ],
 [  8.,       0.,      -7.5   ],
 [  8.125,   -2.5,     -2.5   ],
 [  8.25,    -2.5,     -3.75  ],
 [  8.375,   -5.,       0.,    ],
 [  9.125,  -10.,       0.,    ],
 [  9.5,    -10.,      -1.25  ],
 [  9.75,   -10.,      -2.5   ],
 [ 10.,     -10.,      -5.,    ]]


ch_2b_1_0 = [[  8.,   0.,   0.],
 [  9.,   0., -10.],
 [ 10., -10.,   0.]]

ch_2b_2_0 = [[  6.,   0.,   0.],
 [  9.,   0., -10.],
 [ 10., -10.,   0.]]

ch_2b_1_1 = [[  8.25,   0.,     0.  ],
 [  9.,     0.,    -7.5 ],
 [  9.25, -10.,     0.  ],
 [ 10.,   -10.,    -7.5 ]]

ch_2b_2_1 = [[  7.,     0.,     0.  ],
 [  7.75,   0.,    -2.5 ],
 [  8.25,  -5.,     0.  ],
 [  8.25,   0.,    -5.  ],
 [  8.75, -10.,     0.  ],
 [  9.,     0.,   -10.  ],
 [  9.5,  -10.,    -2.5 ],
 [ 10.,   -10.,    -5.  ]]

ch_2b_2_2 = [[  7.5625,   0.,       0.,    ],
 [  8.125,    0.,      -1.875 ],
 [  8.375,   -5.,       0.,    ],
 [  8.375,    0.,      -3.125 ],
 [  8.75,    0.,      -5.625 ],
 [  9.,       0.,      -7.5   ],
 [  9.125,  -10.,       0.,    ],
 [  9.5,    -10.,      -1.25  ],
 [  9.75,   -10.,      -2.5   ],
 [ 10.,     -10.,      -5.,    ]]

ch_3_1_0 = [[  6.,   0.,   0.],
 [  7.,   0., -20.],
 [  9., -10.,   0.],
 [ 10., -30.,   0.]]

ch_3_2_0 = [[  4.,   0.,   0.],
 [  7.,   0., -20.],
 [  9., -10.,   0.],
 [ 10., -30.,   0.]]

ch_3_1_1 = [[  6.25,   0.,     0.  ],
 [  7.,     0.,   -15.  ],
 [  8.25, -10.,     0.  ],
 [  9.,   -10.,    -7.5 ],
 [  9.25, -30.,     0.  ],
 [ 10.,   -30.,    -7.5 ]]

ch_3_2_1 = [[  5.5,    0.,     0.  ],
 [  6.25,   0.,    -5.  ],
 [  6.5,    0.,    -7.5 ],
 [  7.,     0.,   -15.  ],
 [  8.,   -10.,     0.  ],
 [  8.5,  -20.,     0.  ],
 [  8.75, -30.,     0.  ],
 [  9.,   -10.,    -5.  ],
 [  9.5,  -30.,    -2.5 ],
 [ 10.,   -30.,    -5.  ]]

ch_3_2_2 = [[  5.75,     0.,       0.,    ],
 [  5.9375,   0.,      -0.9375],
 [  6.3125,   0.,      -3.4375],
 [  6.5,      0.,      -5.3125],
 [  6.75,     0.,      -9.0625],
 [  7.,       0.,     -13.75  ],
 [  8.25,   -10.,       0.,    ],
 [  8.75,   -10.,      -2.5   ],
 [  9.,     -25.,       0.,    ],
 [  9.,     -10.,      -5.,    ],
 [  9.125,  -30.,       0.,    ],
 [  9.5,    -30.,      -1.25  ],
 [  9.75,   -30.,      -2.5   ],
 [ 10.,     -30.,      -5.,    ]]

if __name__ == "__main__":

    all_hulls = list()

    for i in [1, "2a", "2b", 3]:
        for j in [1, 2]:
            for k in range(3):
                if k <= j:
                    all_hulls.append(get_convex_hull(i, j, k))

    lexes = list(itertools.permutations([0, 1, 2]))

    print(lexes)

    #hull = get_convex_hull("2b", 2, 1)
    hull = np.load("v_function.npy", allow_pickle=True)[43][45][31]

    print(hull)

    lex_set = get_lex_set_from_hull(hull, lexes)
    for point in lex_set:
        print(point)
    print(len(np.unique(lex_set, axis=0)))

