
# An implementation of table based Q-Learning on openAI's CartPole game.
#   We only use the angle and the angular velocity of the pole as overbasions
# and find hyper parameters for the model.

import gym
import numpy as np
import random
import math
from time import sleep
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from threading import Thread
plt.close()
plt.close()
plt.close()


######################### PARAMETERS ################################
INIT_EXPLORE_RATE_ER = 1.0
TIME_CONSTANT_ER =  10 + 90
DECAY_POINT_ER = 100 + 100 + 100 + 100 + 100
MIN_EXPLORE_RATE = 0.005 - 0.004 - 0.0008

INIT_LEARNING_RATE = 0.5
TIME_CONSTANT_LR = 40
DECAY_POINT_LR = 270 + 50
MIN_LEARNING_RATE = 0.2 - 0.04 - 0.03 - 0.03 - 0.03

DISCOUNT_FACTOR = 0.99
NUM_BUCKETS = ( 1, 1, 6, 3 )
# STATE_BOUNDS[ 1 ] = [ -0.5, 0.5 ]
# STATE_BOUNDS[ 3 ] = [ - math.radians( 50 ), math.radians( 50 ) ]
NUM_EPISODES = int( 4000 * 1 )
MAX_T = 1000 + 4000 + 3000 # the muximum number of steps in each episode.
SOLVED_T = 400
STREAK_TO_END = 120
DEBUG_MODE = True
#####################################################################

############################ ENVIRONMENT COFIGS #####################
# env = gym.make( 'CartPole-v0' )
# NUM_ACTIONS = env.action_space.n
# STATE_BOUNDS = list( zip( env.observation_space.low,
# env.observation_space.high ) )
# q_table = np.zeros( NUM_BUCKETS + ( NUM_ACTIONS, ) )
#####################################################################

def get_explore_rate( episode ) :
    init = INIT_EXPLORE_RATE_ER
    tc = TIME_CONSTANT_ER
    if episode < DECAY_POINT_ER:
        return init
    # return init * np.exp( - episode / tc )
    return init * ( MIN_EXPLORE_RATE + np.exp( - episode / tc ) )

def get_learning_rate( episode ) :
    init = INIT_LEARNING_RATE
    tc = TIME_CONSTANT_LR
    if episode < DECAY_POINT_LR :
        return init
    return init * ( MIN_LEARNING_RATE +  np.exp( - episode / tc ) )

def plot_lr() :
    t = np.arange( 1, NUM_EPISODES ) * 1.
    y = np.vectorize( get_learning_rate )( t )
    # plt.close()
    plt.plot( t, y )
    plt.show()

def plot_er() :
    t = np.arange( 1, NUM_EPISODES ) * 1.
    y = np.vectorize( get_explore_rate )( t )
    # plt.close()
    plt.plot( t, y )
    plt.show()

# streaks_list = []
# steps_list = []
def start_training( thread_num = 0 ) :
    ## SET THE ENVIRONMENT ######################
    env = gym.make( 'CartPole-v0' )
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list( zip( env.observation_space.low,
    env.observation_space.high ) )
    q_table = np.zeros( NUM_BUCKETS + ( NUM_ACTIONS, ) )
    STATE_BOUNDS[ 1 ] = [ -0.5, 0.5 ]
    STATE_BOUNDS[ 3 ] = [ - math.radians( 50 ), math.radians( 50 ) ]
    #############################################

    ## START TRAINING ###########################
    streaks_list = []
    steps_list = []

    learning_rate = get_learning_rate( 0 )
    explore_rate = get_explore_rate( 0 )
    discount_factor = DISCOUNT_FACTOR
    num_streaks = 0
    for episode in range( NUM_EPISODES ) :
        obv = env.reset()
        state_0 = state_to_bucket( obv, STATE_BOUNDS )
        for t in range( MAX_T ) :
            # env.render()
            action = select_action( state_0, explore_rate, q_table, env )
            obv, reward, done, _ = env.step( action )
            state = state_to_bucket( obv, STATE_BOUNDS )
            best_q = np.amax( q_table[ state ] )
            q_table[ state_0 ][ action ] = \
                ( 1 - learning_rate ) * q_table[ state_0 ][ action ] + \
                learning_rate * ( reward + discount_factor * best_q )
            state_0 = state

            if done :
                print 'Thread %02d Episode %04d finished after %04d time steps, num_streaks = %04d' %\
                        ( thread_num, episode, t, num_streaks )
                if t >= SOLVED_T :
                    num_streaks += 1
                else :
                    num_streaks = 0
                break
            elif t == ( MAX_T - 1 ) :
                print 'Episode %04d finished after %04d time steps, num_streaks = %04d' %\
                        ( episode, t, num_streaks)
                num_streaks += 1
                break
        steps_list.append( t )
        streaks_list.append( num_streaks )
        explore_rate = get_explore_rate( episode )
        learning_rate = get_learning_rate( episode )
    #############################################

    ## VISUALIZE TRAINING #############
    x = np.arange( len( steps_list ) )
    y = np.array( steps_list )
    fig = plt.figure()
    ax = fig.add_subplot( 111 )
    # ax.plot( x, y,  )
    ax.scatter( x = x, y = y, c = 'black', marker = 'o', s = 10 )
    ax.plot( [ 0, len( steps_list ) ], [ SOLVED_T, SOLVED_T ], 'r-' )
    ax.plot( x, streaks_list, 'b-' )
    plt.xlabel( 'episode' )
    plt.ylabel( 'steps' )
    plt.grid()
    plt.show()

    score = np.sum( steps_list_np > SOLVED_T ).sum() * 1. / len( steps_list )
    print 'Score: {}'.format( score )
    ###################################

# def visualize_training( steps_list, streaks_list ) :
#     x = np.arange( len( steps_list ) )
#     y = np.array( steps_list )
#     fig = plt.figure()
#     ax = fig.add_subplot( 111 )
#     # ax.plot( x, y,  )
#     ax.scatter( x = x, y = y, c = 'black', marker = 'o', s = 10 )
#     ax.plot( [ 0, len( steps_list ) ], [ SOLVED_T, SOLVED_T ], 'r-' )
#     ax.plot( x, streaks_list, 'b-' )
#     plt.xlabel( 'episode' )
#     plt.ylabel( 'steps' )
#     plt.grid()
#     plt.show()

def test() :
    NUM_TESTS = 5
    MAX_STEPS = 2000
    for episode in range( NUM_TESTS ) :
        obv = env.reset()
        state = state_to_bucket( obv )
        reward_sum = 0
        for step in range( MAX_STEPS ) :
            env.render()
            action = np.argmax( q_table[ state ] )
            next_obv, reward, done, _ = env.step( action )
            state = state_to_bucket( next_obv )
            reward_sum += reward
            print 'Episode {}, Step {}, reward_sum {}'.format( episode, step, reward_sum )
            if done :
                # print 'reward_sum: {}'.format( reward_sum )
                break

def select_action( state, explore_rate, q_table, env ) :
    if random.random() < explore_rate :
        action = env.action_space.sample()
    else :
        action = np.argmax( q_table[ state ] )
    return action

def state_to_bucket( state, STATE_BOUNDS ) :
    bucket_indice = []
    for i in range( len( state ) ) :
        if state[ i ] <= STATE_BOUNDS[ i ][ 0 ] :
            bucket_index = 0
        elif state[ i ] >= STATE_BOUNDS[ i ][ 1 ] :
            bucket_index = NUM_BUCKETS[ i ] - 1
        else :
            bound_width = STATE_BOUNDS[ i ][ 1 ] - STATE_BOUNDS[ i ][ 0 ]
            offset = ( NUM_BUCKETS[ i ] - 1 ) * STATE_BOUNDS[ i ][ 0 ] / bound_width
            scaling = ( NUM_BUCKETS[ i ] - 1  ) / bound_width
            bucket_index = int( round( scaling * state[ i ] - offset ))
        bucket_indice.append( bucket_index )
    return tuple( bucket_indice )


############################ START TRAINING #########################
start_training()
# # test()
# # visualize_training()
# steps_list_np = np.array( steps_list )
# score = np.sum( steps_list_np > SOLVED_T ).sum() * 1. / len( steps_list )
# print 'Score: {}'.format( score )
#####################################################################
