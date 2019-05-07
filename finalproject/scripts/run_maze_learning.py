#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
import pickle
#import rosparam
# import our training environment
import my_turtlebot2_maze

def loadQtable(file_name,nfiles): # load table
        if nfiles==1:
            with open(file_name, 'rb') as handle:
                Q = pickle.load(handle)
        else :
            for i in range(nfiles):
                with open(file_name+str(i), 'rb') as handle:
                    K=pickle.load(handle)
                    for i in range(len(K)):
                        Q.update({K[i][0]:K[i][1]})
        return Q

if __name__ == '__main__':

    rospy.init_node('example_turtlebot2_maze_qlearn', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('MyTurtleBot2Maze-v0')
    rospy.loginfo("Gym environment done")

    #rosparam.load_file("../config/turtlebot2_openai_qlearn_params.yaml")
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('finalproject')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot2/alpha")
    Epsilon = 10
    Gamma = rospy.get_param("/turtlebot2/gamma")
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot2/nepisodes")
    nsteps = rospy.get_param("/turtlebot2/nsteps")

    running_step = rospy.get_param("/turtlebot2/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    qlearn.q = loadQtable("../Qvalue/maze_",1)

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do

    # Initialize the environment and get first state of the robot
    observation = env.reset()
    state = ''.join(map(str, observation))
    done = False

    # Show on screen the actual situation of the robot
    # env.render()
    # for each episode, we test the robot for nsteps
    while not done:
        # Pick an action based on the current state
        action = qlearn.chooseAction(state)
        
        observation, reward, done, info = env.step(action)

        nextState = ''.join(map(str, observation))

    env.close()