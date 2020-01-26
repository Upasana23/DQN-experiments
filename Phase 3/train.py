import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import logger
from collections import deque
from agent import Agent
from config import CONFIG
from wrappers import make_env
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Only run on GPU 0

C= CONFIG

filePath = '/users/upattnai/Phase3/'
resultPath = '/users/upattnai/Phase3/results_ip_Tennis_Breakout/'
weightsPath = '/users/upattnai/Phase3/weightsTennis/ItrPrac/1000000_weights.pk1'
envName = 'IterativeP_Tennis_Breakout'

def reset_fs():
  fs = deque([], maxlen = C['frame_stack'])
  for i in range(C['frame_stack']):
    fs.append(np.zeros((84,84), dtype=np.uint8))
  return fs

def main():

  logger.configure('{}{}_logs'.format(filePath, envName))
  for k, v in C.items():
    logger.record_tabular(k, v)
  logger.dump_tabular()


  logger.log('Breakout from Pong DQN. epsilon 1.0 to 0.1 in 5e5. 10000000_weights.pkl')

  #Start the session
  sess = tf.InteractiveSession()

  with open(weightsPath,"rb") as wt:
    weights = pickle.load(wt)
  wt_cnn = weights[0]

  train_env = make_env(C['env_id'], C['noop_max'])
  eval_env = make_env(C['env_id'], C['noop_max'])

  #Intitialize variables to record outputs
  train_track = [0.0]
  eval_track = []
  best_reward = 0

  train_reward = tf.placeholder(tf.float32)
  eval_reward = tf.placeholder(tf.float32)
  train_env = make_env(C['env_id'], C['noop_max'])
  eval_env = make_env(C['env_id'], C['noop_max'])
  agent = Agent(train_env, C, wt_cnn)

  train_fs = reset_fs()
  train_s = train_env.reset()
  best_reward = 0
  train_mean = []
  eval_mean = []

  train_summary = tf.summary.scalar('train_reward', train_reward)
  eval_summary = tf.summary.scalar('eval_reward', eval_reward)
  writer = tf.summary.FileWriter('{}{}_summary'.format(filePath, envName), sess.graph)
  sess.run(tf.global_variables_initializer())


  agent.net.load_weights()
  logger.log("Loaded weights")
  agent.net.update_target_network()

  for it in range(C['iterations']):

    train_fs.append(train_s)

    train_a = agent.act(np.transpose(train_fs, (1,2,0)))
    ns, train_r, train_d, _ = train_env.step(train_a)
    #print('Iteration ',it, ' Reward ', train_r)
    train_track[-1]+= train_r
    agent.record(train_s, train_a, train_r, float(train_d), it)
    train_s = ns

    if train_d:
      if train_env.env.env.was_real_done: # train_env.env.was_real_done for MsPacman, Freeway (No Fire action)
        if len(train_track) % 100 == 0:
          mean = np.mean(train_track[-100:])
          train_mean.append(mean)
          summary = sess.run(train_summary, feed_dict={train_reward:mean})
          writer.add_summary(summary, it)
          logger.record_tabular('steps', it)
          logger.record_tabular('episode', len(train_track))
          logger.record_tabular('epsilon', 100*agent.epsilon)
          logger.record_tabular('learning rate', agent.lr)
          logger.record_tabular('Mean Reward 100 episdoes', mean)
          logger.dump_tabular()
          with open(resultPath + 'reward_atari_base.pk1', 'wb') as f:
            pickle.dump(train_track, f, protocol=pickle.HIGHEST_PROTOCOL)

        train_track.append(0.0)

      train_fs = reset_fs()
      train_s = train_env.reset()


    if (it+1)%C['eval_freq'] == 0:

      for i in range(C['eval_episodes']):
        temp_video = []
        eval_track.append(0.0)
        eval_fs = reset_fs()
        eval_s = eval_env.reset()
        while True:
          temp_video.append(eval_s)
          eval_fs.append(eval_s)
          eval_a = agent.greedy_act(np.transpose(eval_fs, (1,2,0)))
          eval_s, eval_r, eval_d, _ = eval_env.step(eval_a)
          eval_track[-1] += eval_r

          if eval_env.env.env.was_real_done:
              break
          if eval_d:
              eval_fs = reset_fs()
              eval_s = eval_env.reset()

        if eval_track[-1] > best_reward:
          best_reward = eval_track[-1]
          best_video = temp_video
          with open(resultPath + 'video_atari_base.pk1', 'wb') as f:
              pickle.dump(best_video, f, protocol=pickle.HIGHEST_PROTOCOL)

      eval_mean.append(np.mean(eval_track[-C['eval_episodes']:]))
      summary = sess.run(eval_summary, feed_dict={eval_reward:np.mean(eval_track[-C['eval_episodes']:])})
      writer.add_summary(summary, it)

    """if it == 1000000:
      outputs = agent.net.get_outputs(np.transpose(train_fs, (1,2,0)))
      with open(resultPath+ 'outputs.pk1', 'wb') as f:
        pickle.dump(outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
      with open(resultPath+ 'outputs_screen.pk1', 'wb') as f:
        pickle.dump(train_fs, f, protocol=pickle.HIGHEST_PROTOCOL)"""

    """if it%1000000 == 0:
      weights = agent.net.get_weights()
      with open(resultPath + str(it) +'_weights.pk1', 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)"""


  with open(resultPath + 'reward_atari_base.pk1', 'wb') as f:
    pickle.dump(train_track, f, protocol=pickle.HIGHEST_PROTOCOL)
  with open(resultPath + 'trainMean_atari_base.pk1', 'wb') as f:
    pickle.dump(train_mean, f, protocol=pickle.HIGHEST_PROTOCOL)
  with open(resultPath+ 'evalMean_atari_base.pk1', 'wb') as f:
    pickle.dump(eval_mean, f, protocol=pickle.HIGHEST_PROTOCOL)
  agent.net.save(filePath + '{}_model2'.format(C['env_id']))
  sess.close()


if __name__ == '__main__':
	main()
