import os
from datetime import datetime

import torch
import numpy as np

from PPO import PPO
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from environment import OcclusionEnv
from pytorch3d.datasets import ShapeNetCore
import tqdm
useShapeNet = True

################################### Training ###################################
def train():
    print("============================================================================================")

    env_name = 'OcclusionEnv'
    ####### initialize environment hyperparameters ######

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 50                   # max timesteps in one episode
    max_training_timesteps = int(2e4)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 4        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(5e3)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(1.5e4)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 42         # set random seed if required (0 = no random seed)
    #####################################################

    print("training OcclusionEnv")

    # Set the cuda device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    shapenet_dataset = None
    if useShapeNet:
        # Load shapenet dataset
        try:
            # From Matyi's external drive
            shapenetdir = "/Volumes/MacMiklos/M/BME/2021_12-OcclusionEnvironment/Shapenet/"
            shapenet_dataset = ShapeNetCore(shapenetdir, version=2)
        except:
            try:
                shapenet_dataset = ShapeNetCore("./data/shapenet/shapenetcore", version=2)
            except:
                shapenet_dataset = ShapeNetCore("/data/shapenet/shapenetcore", version=2)

        print("Shapenetcore dataset loaded")

    env = OcclusionEnv(shapenet_dataset)
    env.renderMode = "" # 'human'

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = run_num      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward,running_reward,length\n')

    # printing and logging variables

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step(action)
            #env.render()
            #print(reward)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward.detach())
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward.detach()

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)


            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        i_episode += 1

        # log average reward till last episode
        log_avg_reward = current_ep_reward
        log_avg_reward = round(log_avg_reward.cpu().item(), 4)
        running_reward = 0.95*running_reward + 0.05*current_ep_reward if i_episode > 1 else current_ep_reward
        log_running_reward = round(running_reward.cpu().item(), 4)

        log_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward, log_running_reward, t))
        log_f.flush()

        # print average reward till last episode
        print_avg_reward = current_ep_reward
        print_avg_reward = round(print_avg_reward.cpu().item(), 2)
        print_running_reward = round(running_reward.cpu().item(), 2)

        print("Episode: {} \t Length: {} \t Timestep: {} \t Average Reward: {} \t Running Reward: {}".format(i_episode, t, time_step, print_avg_reward, print_running_reward))


    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
