import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle


def visualize_point_clouds(point_clouds, boxes=None, c=['b', 'r'], view=None, store_path=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.axes.zaxis.set_ticklabels([])

    for i in range(len(point_clouds)):
        points = point_clouds[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c[i], s=10, alpha=0.3)

    X, Y, Z = point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.grid(False)

    if view is None:
        view = 0, 0
    ax.view_init(view[0], view[1])
    plt.show()

    # plt.pause(5)

    if store_path:
        fig.savefig(store_path, bbox_inches='tight')


def visualize_neighbors(anchors, queries, idx, neighbors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c='g', s=80)
    ax.scatter(anchors[neighbors, 0], anchors[neighbors, 1], anchors[neighbors, 2], c='r', s=80)
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], alpha=0.2)
    ax.set_aspect('equal')

    plt.show()