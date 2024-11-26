from Localization import *
import numpy as np
from DifferentialDriveSimulatedRobot import *


class DR_3DOFDifferentialDrive(Localization):
    """
    Dead Reckoning Localization for a Differential Drive Mobile Robot.
    """
    def __init__(self, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`prlab.DR_3DOFDifferentialDrive` class.

        :param args: Rest of arguments to be passed to the parent constructor
        """

        super().__init__(index, kSteps, robot, x0, *args)  # call parent constructor

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.t_1 = 0.0  # t_1 is the previous time at which we iterated the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn

    def Localize(self, xk_1, uk):  # motion model
        """
        Motion model for the 3DOF (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`) Differential Drive Mobile robot using as input the readings of the wheel encoders (:math:`u_k=[n_L~n_R]^T`).

        :parameter xk_1: previous robot pose estimate (:math:`x_{k-1}=[x_{k-1}~y_{k-1}~\psi_{k-1}]^T`)
        :parameter uk: input vector (:math:`u_k=[u_{k}~v_{k}~w_{k}~r_{k}]^T`)
        :return xk: current robot pose estimate (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`)
        """
         
        # Store previous state and input for Logging purposes
        self.etak_1 = xk_1  # store previous state
        self.uk = uk  # store input

        # TODO: to be completed by the student

        # Variables obtention
        R =  self.robot.wheelRadius 
        ppr = self.robot.pulse_x_wheelTurns # number of ticks that the encoder register per each turn of a wheel
        l = self.wheelBase # distance between wheels
        
        # Store previous pose
        x_k_1 = xk_1[0][0] # x position with respect to the world frame
        y_k_1 = xk_1[1][0] # y position with respect to the world frame
        theta_k_1 = xk_1[2][0] # theta with respect to the world frame (robot orientation)
        
        # Retrieve pulses from encoders
        pulses, _ = self.GetInput()
        pl, pr = pulses  # pulses of left and right wheel
        pl = pl[0] # get the values
        pr = pr[0]

        # Compute individual wheel displacements
        dl = 2 * np.pi * R * pl / ppr  # Left displacement         
        dr = 2 * np.pi * R * pr / ppr  # Right displacemente 
            
        # Compute robot displacemente
        d = (dl + dr)/2

        # Compute change in theta for the actual iteration
        d_theta_k = (dr - dl) / l

        # Add the change of theta
        theta_k = theta_k_1 + d_theta_k
        
        # Compute the new x and y positions with respect to the world frame
        x_k = x_k_1 + (d*np.cos(theta_k)) 
        y_k = y_k_1 + (d*np.sin(theta_k))
        
        # Define the new updated pose vector
        new_pose = np.array([[x_k],
                             [y_k],
                             [theta_k]])
        
        #new_pose = np.array([[self.robot.xsk[0]], [self.robot.xsk[1]],[self.robot.xsk[2]]])

        return new_pose

    def GetInput(self):

        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """
    
        # TODO: to be completed by the student
        
        # retreive pulses from robot encoders
        encoders_reading = self.robot.ReadEncoders()
        pulses, covariance = encoders_reading

        return (pulses, covariance)


if __name__ == "__main__":

    # feature map. Position of 2 point features in the world frame.
    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0=np.zeros((6,1))   # initial simulated robot pose
    robot = DifferentialDriveSimulatedRobot(xs0, M) # instantiate the simulated robot object

    kSteps = 5000 # number of simulation steps
    xsk_1 = xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)] # index of the state vector used for plotting

    x0=Pose3D(np.zeros((3,1)))
    dr_robot=DR_3DOFDifferentialDrive(index,kSteps,robot,x0)
    dr_robot.LocalizationLoop(x0, np.array([[0.5, 0.03]]).T)

    exit(0)