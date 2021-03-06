import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np


class DebugVisualizer:
    """
    visualizes the data
    """

    def __init__(self):
        """
        Initializes the DebugVisualizer
        """

        # create the list of connections in the human body
        # 0 Root
        # 1 left hip 5 right hip
        # 2 left knee 6 right knee
        # 3 left ankle 7 right ankle
        # 4 left toe 8 right toe
        # 9 torso 10 chest
        # 11 neck 12 nose
        # 13 left shoulder 17 right shoulder
        # 14 left elbow 18 right elbow
        # 19 left wrist 15 right wrist
        # 16 left thumb 20 right thumb
        self.humanSkeleton = [
            [1, 5],
            [0, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [5, 6],
            [1, 2],
            [6, 7],
            [2, 3],
            [13, 17],
            [17, 18],
            [13, 14],
            [14, 15],
            [18, 19]
        ]

        self.humanSkeleton_19_joints = [
            [0, 1],
            [0, 3],
            [3, 4],
            [4, 5],
            [0, 2],
            [2, 6],
            [6, 7],
            [7, 8],
            [2, 12],
            [12, 13],
            [13, 14],
            [0, 9],
            [9, 10],
            [10, 11]
        ]

        # add the joints and the marker colors
        # setup the frame layout
        self.fig = plt.figure()
        self.dpi = 2000
        self.fig.set_size_inches(19.20, 10.80, True)
        self.ax = p3.Axes3D(self.fig)
        self.ax.view_init(elev=130, azim=-90)
        # self.ax._axis3don = False
        self.ax.set_xlim3d(-200, 200)
        self.ax.set_ylim3d(0, 200)
        self.ax.set_zlim3d(-200, 200)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # set up the writer
        # Set up formatting for the movie files
        Writer = anim.writers['ffmpeg']
        self.writer = Writer(fps=24, metadata=dict(artist='Kurel002'), bitrate=self.dpi)

    def add_skeleton(self, joints, color):
        """
        Adds a new skeleton to the plot
        :param joints: Joints Tensor of shape (57,seqLength)
        :param color: color of the skeleton
        :return: points, bones references to the new plot
        """

        jointList = []
        points = []
        for i in range(0, len(joints), 3):
            x = joints[i][0]
            y = joints[i + 1][0]
            z = joints[i + 2][0]
            jointList.append((x, y, z))
            graph, = self.ax.plot([x], [y], [z], linestyle="", c=color, marker="o")
            points.append(graph)

        bones = []
        for start, end in self.humanSkeleton:
            xs = [jointList[start][0], jointList[end][0]]
            ys = [jointList[start][1], jointList[end][1]]
            zs = [jointList[start][2], jointList[end][2]]
            graph, = self.ax.plot(xs, ys, zs, c=color)
            bones.append(graph)

        return points, bones

    def update_skeleton(self, frame, points, bones, joints):
        """
        Updates the skeleton positions
        :param frame: frame number
        :param points: references to all the points in the plot
        :param bones: references to all the bones in the plot
        :param joints: list of all joint sequences to be added
        :return:
        """

        jointList = []
        for i in range(len(points)):
            graph = points[i]
            x = joints[3 * i][frame]
            y = joints[3 * i + 1][frame]
            z = joints[3 * i + 2][frame]
            jointList.append((x, y, z))
            graph.set_data([x], [y])
            graph.set_3d_properties([z])

        for i in range(len(bones)):
            start, end = self.humanSkeleton[i]
            graph = bones[i]
            xs = [jointList[start][0], jointList[end][0]]
            ys = [jointList[start][1], jointList[end][1]]
            zs = [jointList[start][2], jointList[end][2]]
            graph.set_data(xs, ys)
            graph.set_3d_properties(zs)

    def update_plot(self, frame, joints, points, bones):

        for i in range(len(points)):
            self.update_skeleton(frame, points[i], bones[i], joints[i])

    def create_animation(self, jointsList, fileName):
        """
        Creates the animation
        :param jointsList: List of all sequences to be added to the visualization
        :param fileName: Name of the file to store the video in
        :return:
        """

        # create a list of colors
        colors = ['r', 'g', 'b', 'y']

        pointsList = []
        bonesList = []
        for i in range(len(jointsList)):

            joints = jointsList[i]
            if i < 3:
                color = colors[i]
            else:
                color = colors[3]

            points, bones = self.add_skeleton(joints, color)
            pointsList.append(points)
            bonesList.append(bones)

        frames = len(jointsList[0][0])

        # create the animation
        ani = anim.FuncAnimation(
            self.fig,
            self.update_plot,
            frames,
            fargs=(jointsList, pointsList, bonesList),
            interval=16,
            repeat=True
        )

        # save only if filename is passed
        if fileName:
            print(fileName + '.mp4')
            #matplotlib.use("Agg")
            ani.save(fileName + '.mp4', writer=self.writer)
        else:
            plt.show(self.fig)

        plt.close(self.fig)

    def visualize_points(self, positions):
        """
        Debug function for visualizing the rest skeleton
        :param positions: (F, J, 3) numpy array
        :return:
        """
        ax = plt.axes(projection='3d')

        points = positions[0]
        for i in range(points.shape[0]):
            ax.scatter(points[i][0], points[i][1], points[i][2])
            ax.text(points[i][0], points[i][1], points[i][2], str(i))

        bones = self.humanSkeleton
        for i in range(len(bones)):
            start, end = bones[i]
            xs = [points[start][0], points[end][0]]
            ys = [points[start][1], points[end][1]]
            zs = [points[start][2], points[end][2]]
            ax.plot3D(xs, ys, zs)

        ax.set_xlim3d(-200, 200)
        ax.set_ylim3d(0, 200)
        ax.set_zlim3d(-300, 100)

        # Hide grid lines
        # ax.grid(False)

        # Hide axes ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def conv_debug_visual_form(self, rest_targets):
        """
        Function for converting the input into the required format
        :param rest_targets: Input skeleton (F, J, 3)
        :return: rest_targets (3J, F)
        """
        rest_targets = rest_targets.reshape(rest_targets.shape[0], -1)  # (F, 3J)
        rest_targets = np.swapaxes(rest_targets, 0, 1)  # (3J, F)

        return rest_targets
