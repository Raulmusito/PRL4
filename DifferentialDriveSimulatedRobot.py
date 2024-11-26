from SimulatedRobot import *
from IndexStruct import *
from Pose import *
import scipy
from roboticstoolbox.mobile.Animations import *
import numpy as np
import argparse


class DifferentialDriveSimulatedRobot(SimulatedRobot):
    """
    This class implements a simulated differential drive robot. It inherits from the :class:`SimulatedRobot` class and
    overrides some of its methods to define the differential drive robot motion model.
    """
    def __init__(self, xs0, map=[],*args):
        """
        :param xs0: initial simulated robot state :math:`\\mathbf{x_{s_0}}=[^Nx{_{s_0}}~^Ny{_{s_0}}~^N\psi{_{s_0}}~]^T` used to initialize the  motion model
        :param map: feature map of the environment :math:`M=[^Nx_{F_1},...,^Nx_{F_{nf}}]`

        Initializes the simulated differential drive robot. Overrides some of the object attributes of the parent class :class:`SimulatedRobot` to define the differential drive robot motion model:

        * **Qsk** : Object attribute containing Covariance of the simulation motion model noise.

        .. math::
            Q_k=\\begin{bmatrix}\\sigma_{\\dot u}^2 & 0 & 0\\\\
            0 & \\sigma_{\\dot v}^2 & 0 \\\\
            0 & 0 & \\sigma_{\\dot r}^2 \\\\
            \\end{bmatrix}
            :label: eq:Qsk

        * **usk** : Object attribute containing the simulated input to the motion model containing the forward velocity :math:`u_k` and the angular velocity :math:`r_k`

        .. math::
            \\bf{u_k}=\\begin{bmatrix}u_k & r_k\\end{bmatrix}^T
            :label: eq:usk

        * **xsk** : Object attribute containing the current simulated robot state

        .. math::
            x_k=\\begin{bmatrix}{^N}x_k & {^N}y_k & {^N}\\theta_k & {^B}u_k & {^B}v_k & {^B}r_k\\end{bmatrix}^T
            :label: eq:xsk

        where :math:`{^N}x_k`, :math:`{^N}y_k` and :math:`{^N}\\theta_k` are the robot position and orientation in the world N-Frame, and :math:`{^B}u_k`, :math:`{^B}v_k` and :math:`{^B}r_k` are the robot linear and angular velocities in the robot B-Frame.

        * **zsk** : Object attribute containing :math:`z_{s_k}=[n_L~n_R]^T` observation vector containing number of pulses read from the left and right wheel encoders.
        * **Rsk** : Object attribute containing :math:`R_{s_k}=diag(\\sigma_L^2,\\sigma_R^2)` covariance matrix of the noise of the read pulses`.
        * **wheelBase** : Object attribute containing the distance between the wheels of the robot (:math:`w=0.5` m)
        * **wheelRadius** : Object attribute containing the radius of the wheels of the robot (:math:`R=0.1` m)
        * **pulses_x_wheelTurn** : Object attribute containing the number of pulses per wheel turn (:math:`pulseXwheelTurn=1024` pulses)
        * **Polar2D_max_range** : Object attribute containing the maximum Polar2D range (:math:`Polar2D_max_range=50` m) at which the robot can detect features.
        * **Polar2D\_feature\_reading\_frequency** : Object attribute containing the frequency of Polar2D feature readings (50 tics -sample times-)
        * **Rfp** : Object attribute containing the covariance of the simulated Polar2D feature noise (:math:`R_{fp}=diag(\\sigma_{\\rho}^2,\\sigma_{\\phi}^2)`)

        Check the parent class :class:`prpy.SimulatedRobot` to know the rest of the object attributes.
        """
        super().__init__(xs0, map,*args) # call the parent class constructor

        # Initialize the motion model noise
        self.Qsk = np.diag(np.array([0.1 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2]))  # simulated acceleration noise
        self.usk = np.zeros((3, 1))  # simulated input to the motion model

        # Inititalize the robot parameters
        self.wheelBase = 0.5  # distance between the wheels
        self.wheelRadius = 0.1  # radius of the wheels
        self.pulse_x_wheelTurns = 1024  # number of pulses per wheel turn

        # Initialize the sensor simulation
        self.encoder_reading_frequency = 1  # frequency of encoder readings
        self.Re= np.diag(np.array([22 ** 2, 22 ** 2]))  # covariance of simulated wheel encoder noise

        self.Polar2D_feature_reading_frequency = 50  # frequency of Polar2D feature readings
        self.Polar2D_max_range = 50  # maximum Polar2D range, used to simulate the field of view
        self.Rfp = np.diag(np.array([1 ** 2, np.deg2rad(5) ** 2]))  # covariance of simulated Polar2D feature noise

        self.xy_feature_reading_frequency = 50  # frequency of XY feature readings
        self.xy_max_range = 50  # maximum XY range, used to simulate the field of view
        self.xy_range_std = 0.25 # standard deviation of simulated ranges noise

        self.yaw_reading_frequency = 1  # frequency of Yasw readings
        self.v_yaw_std = np.deg2rad(5)  # std deviation of simulated heading noise

    def fs(self, xsk_1, usk):  # input velocity motion model with velocity noise
        """ Motion model used to simulate the robot motion. Computes the current robot state :math:`x_k` given the previous robot state :math:`x_{k-1}` and the input :math:`u_k`:

        .. math::
            \\eta_{s_{k-1}}&=\\begin{bmatrix}x_{s_{k-1}} & y_{s_{k-1}} & \\theta_{s_{k-1}}\\end{bmatrix}^T\\\\
            \\nu_{s_{k-1}}&=\\begin{bmatrix} u_{s_{k-1}} &  v_{s_{k-1}} & r_{s_{k-1}}\\end{bmatrix}^T\\\\
            x_{s_{k-1}}&=\\begin{bmatrix}\\eta_{s_{k-1}}^T & \\nu_{s_{k-1}}^T\\end{bmatrix}^T\\\\
            u_{s_k}&=\\nu_{d}=\\begin{bmatrix} u_d& r_d\\end{bmatrix}^T\\\\
            w_{s_k}&=\\dot \\nu_{s_k}\\\\
            x_{s_k}&=f_s(x_{s_{k-1}},u_{s_k},w_{s_k}) \\\\
            &=\\begin{bmatrix}
            \\eta_{s_{k-1}} \\oplus (\\nu_{s_{k-1}}\\Delta t + \\frac{1}{2} w_{s_k}) \\\\
            \\nu_{s_{k-1}}+K(\\nu_{d}-\\nu_{s_{k-1}}) + w_{s_k} \\Delta t
            \\end{bmatrix} \\quad;\\quad K=diag(k_1,k_2,k_3) \\quad k_i>0\\\\
            :label: eq:fs

        Where :math:`\\eta_{s_{k-1}}` is the previous 3 DOF robot pose (x,y,yaw) and :math:`\\nu_{s_{k-1}}` is the previous robot velocity (velocity in the direction of x and y B-Frame axis of the robot and the angular velocity).
        :math:`u_{s_k}` is the input to the motion model contaning the desired robot velocity in the x direction (:math:`u_d`) and the desired angular velocity around the z axis (:math:`r_d`).
        :math:`w_{s_k}` is the motion model noise representing an acceleration perturbation in the robot axis. The :math:`w_{s_k}` acceleration is the responsible for the slight velocity variation in the simulated robot motion.
        :math:`K` is a diagonal matrix containing the gains used to drive the simulated velocity towards the desired input velocity.

        Finally, the class updates the object attributes :math:`xsk`, :math:`xsk\_1` and  :math:`usk` to made them available for plotting purposes.

        **To be completed by the student**.

        :parameter xsk_1: previous robot state :math:`x_{s_{k-1}}=\\begin{bmatrix}\\eta_{s_{k-1}}^T & \\nu_{s_{k-1}}^T\\end{bmatrix}^T`
        :parameter usk: model input :math:`u_{s_k}=\\nu_{d}=\\begin{bmatrix} u_d& r_d\\end{bmatrix}^T`
        :return: current robot state :math:`x_{s_k}`
        """

        # TODO: to be completed by the student

        wsk = np.random.multivariate_normal(np.zeros(3),self.Qsk,1).T  # acceleration noise calculated with the Qsk cov mat
        
        etask_1 = Pose3D(xsk_1[0:3]) # get the pose values from the last state
        nusk_1  = np.array(xsk_1[3:6]) # get the velocity values from the last state

        K = np.diag([1.0,1.0,1.0]) # gain matrix for the difference between prvious state vel and desired vel. without any correlation

        vd = np.array([[usk[0][0]], # reshape of the desired velocity to match the 3x1 size requiered to do the difference with the previous el. 
                       [0],
                       [usk[1][0]]])
        
        
        
        # Build actual state vector [[current pose][current vel]]
        self.xsk = np.vstack((etask_1.oplus(nusk_1*self.dt + (1/2)*(self.dt**2)*wsk), 
                                    (nusk_1 + K@(vd - nusk_1)+wsk*self.dt)))

        if self.k % self.visualizationInterval == 0:
                self.PlotRobot()
                self.xTraj.append(self.xsk[0, 0])
                self.yTraj.append(self.xsk[1, 0])
                self.trajectory.pop(0).remove()
                self.trajectory = plt.plot(self.xTraj, self.yTraj, marker='.', color='orange', markersize=1)

        self.k += 1
 
        return self.xsk


    def ReadEncoders(self):
        """ Simulates the robot measurements of the left and right wheel encoders.

        **To be completed by the student**.

        :return zsk,Rsk: :math:`zk=[n_L~n_R]^T` observation vector containing number of pulses read from the left and right wheel encoders. :math:`R_{s_k}=diag(\\sigma_L^2,\\sigma_R^2)` covariance matrix of the read pulses.
        """

        # TODO: to be completed by the student

        if self.k % self.encoder_reading_frequency == 0:
            # Current velocities
            v = self.xsk[3] # the linear velocity in xsk is with respect to the robot frame, so it has just 1 component
            w = self.xsk[5] # angular velocity

            # Robot Dimensions
            l = self.wheelBase # distance between wheels
            r = self.wheelRadius # Radius of a wheel
            ppr = self.pulse_x_wheelTurns # number of ticks that the encoder register per each turn of a wheel

            dt = self.dt
            
            # Compute individual wheel rotational speed
            wl = (v - (l/2) * w) / r    # left wheel angular velocity   
            wr = (v + (l/2) * w) / r    # right wheel angular velocity

            # Compute pulses per wheel encoder, and noise
            zsk = np.array([wl*dt*ppr/(2*np.pi),  
                            wr*dt*ppr/(2*np.pi)])
            
            noise = np.random.multivariate_normal(np.zeros(2),self.Re,1).T # vector of noise of the size 2x1 with the noise for the left and right encoder
            zsk = zsk + noise # add noise to each reading

            # Covariance matrix
            Rsk = self.Re

            return zsk, Rsk

    def ReadCompass(self):
        """ Simulates the compass reading of the robot.

        :return: yaw and the covariance of its noise *R_yaw*
        """

        if self.k % self.yaw_reading_frequency == 0: # provide readings at the predefined frequency
            yaw = self.xsk[2] # get the orientation of the robot with respect to the world frame
            
            # Obtain yaw and compute noise
            R_yaw = self.v_yaw_std**2 # variance
            yaw += np.random.normal(0, self.v_yaw_std)

            return yaw, R_yaw

    def ReadRanges(self):
        """ Simulates reading the distances to the features in the environment.

        return: a list of ranges measurements and the covariance of the noise of the readings 
        """
        ''' Note: You can simulate that the reading includes the id of the feature and the distance to it
                  this will help you to match the readings with the map features and avoid data association problems
        
           Note: define a new class variable to keep the desired measurement covariance (set some meaningful value for it)
        '''
        
        # Provide readings at the predefined frequency
        if self.k % self.xy_feature_reading_frequency != 0:
            return [], []
        
        readings = [] # empty list to store the readings
        ranges_var = self.xy_range_std**2 # variance of the noise of the readings
        robot_x = self.xsk[0][0] # get the x position of the robot
        robot_y = self.xsk[1][0] # get the y position of the robot
        
        for i in range(len(self.M)):
            # print('ReadRanges()>>> feature:',self.M[i])
            feature_x = self.M[i][0][0] # get the x position of the feature
            feature_y = self.M[i][1][0] # get the y position of the feature

            # compute distance with noise
            d = np.sqrt((feature_x - robot_x)**2 + (feature_y - robot_y)**2) # compute the distance between the robot and the feature
            d += np.random.normal(0, self.xy_range_std) # add noise to the reading

            if d < self.xy_max_range: # check if the reading is within the max range
                readings.append((i,d)) # i is the id of the feature, d is the distance to the feature

        return readings, ranges_var

    def PlotRobot(self):
        """ Updates the plot of the robot at the current pose """

        self.vehicleIcon.update([self.xsk[0], self.xsk[1], self.xsk[2]])
        plt.pause(0.0000001)
        return


### TESTING

if __name__ == "__main__":

    # copy the map creation in the main.py code

    # feature map. Position of 2 point features in the world frame.
    M2D = [np.array([[-40, 5]]).T,
            np.array([[-5, 40]]).T,
            np.array([[-5, 25]]).T,
            np.array([[-3, 50]]).T,
            np.array([[-20, 3]]).T,
            np.array([[40,-40]]).T]
    xs0 = np.zeros((6,1))   # initial simulated robot pose
    
    # instance of robot
    roberto = DifferentialDriveSimulatedRobot(xs0, M2D) # instantiate the simulated robot object
    xsk_1 = np.zeros((6, 1))  # initial simulated robot pose

    # parsing testing shape to perform trajectory
    ## ex. run in terminal: >> python DifferentialDriveSimulatedRobot.py "circle"
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", type=str, help="The test shape type string")
    args = parser.parse_args()

    if args.shape == "circle":
        # Circle
        usk = np.array([[5],
                        [0.2]])
        for i in range(1000):
            xsk = roberto.fs(xsk_1,usk)
            xsk_1 = xsk
        # pause for capture
        usk = np.array([[0],
                        [0.0]])
        for i in range(1000):
            xsk = roberto.fs(xsk_1,usk)
            xsk_1 = xsk
    elif args.shape == "8":
        # Eight
        for j in range(2):
            usk = np.array([[5],
                        [0.2]])
            for i in range(314):
                xsk_1 = roberto.fs(xsk_1,usk)

            usk = np.array([[5],
                            [-0.2]])
            for i in range(314):
                xsk_1 = roberto.fs(xsk_1,usk)
        # pause for capture
        usk = np.array([[0],
                        [0.0]])
        for i in range(1000):
            xsk_1 = roberto.fs(xsk_1,usk)
    else:
        print("No circle nor 8 shape given")