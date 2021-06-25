import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")
env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

#Hyperparameters
LEARNING_RATE = 0.1 #ALPHA
DISCOUNT_FACTOR = 0.95 #future reward Vs current reward
EPISODES = 25000
SHOW_EVERY = 500
Epsilon = 0.5 # exploratory: higher epsilon more likely to perform random action
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = (Epsilon)/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

#size of states
# as states are two so 20x20
DISCRETE_OS_SIZE =[20]*len(env.observation_space.high)  
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
# print(discrete_os_win_size)

#draw samples from uniform distribution
#size : 20x20x3
q_table = np.random.uniform(low=-2, high=0,size = (DISCRETE_OS_SIZE + [env.action_space.n]))
# print(q_table.shape)

#list of rewards in each episode
ep_rewards = []
#dict : episode number , avg, min, max
aggr_ep_rewards = {'ep': [], 'avg': [],'min':[],'max':[]}


#As states are continuous
def get_discrete_state(state):
    #state = current state
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))
#print(discrete_st)
# print(q_table[8][10])
#For argmax 0, we will go for action 0
# print(np.argmax(q_table[discrete_st]))

#Mountain Car:
# 3 actions it can take: Left-0, Nothing-1, Right-2
for episode in range(EPISODES):
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else: 
        render = False

    discrete_st = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random()>Epsilon:
            action = np.argmax(q_table[discrete_st])
        else: 
            action = np.random.randint(0,env.action_space.n)

        #every step we will do this action
        #state: position, velocity
        new_state, reward,done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        episode_reward += reward
    
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            # it returns 3 dim value (6,10,2)
            current_q = q_table[discrete_st + (action, )]
            new_q = (1 - LEARNING_RATE)*current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR*max_future_q)
            q_table[discrete_st+(action, )] = new_q

        elif new_state[0] >= env.goal_position: # new_state[0] -- position
            print("We made it on episode {}".format(episode))
            q_table[discrete_st + (action, )] = 0
        
        discrete_st = new_discrete_state

        if END_EPSILON_DECAYING >= episode>= START_EPSILON_DECAYING:
            Epsilon -= epsilon_decay_value

        ep_rewards.append(episode_reward)

        if not episode % SHOW_EVERY:
            average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
            print('Ep:',ep_rewards[-SHOW_EVERY:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
            aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

            print('Episode : {},Avg:{},Min:{},Max:{}'.format(episode,average_reward,min(ep_rewards[-SHOW_EVERY:]),max(ep_rewards[-SHOW_EVERY:])))


env.close()

plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],Label ='avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],Label ='min')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],Label ='max')
plt.legend(loc = 4)
plt.show()