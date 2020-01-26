# Transfer learning
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import logger
from agent import Agent
from config import CONFIG
from wrappers import make_env

C= CONFIG

filePath = '/users/upattnai/Phase2P2/PDQN/'
resultPath = '/users/upattnai/Phase2P2/PDQN/resultsBoxing_Breakout/'
weightsPath = '/users/upattnai/Phase1/BaseDQNResults/Boxing/boxing_results/10000000_weights.pk1'
envName = 'Boxing_Breakout'

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

  logger.log('Storing weights after 1000000 steps. Practice DQN with Dense 512. 3 layer trasnfer. Epsilon 1.0 to 0.1 6e5.')

  sess = tf.InteractiveSession()

  with open(weightsPath, "rb") as wt:
      weights = pickle.load(wt)

  wt_cnn =weights[0]
  train_env = make_env(C['env_id'], C['noop_max'])
  eval_env = make_env(C['env_id'], C['noop_max'])
  train_s = train_env.reset()
  agent = Agent(train_env, C, wt_cnn)

  train_reward = tf.placeholder(tf.float32)
  eval_reward = tf.placeholder(tf.float32)
  train_summary = tf.summary.scalar('train_reward', train_reward)
  eval_summary = tf.summary.scalar('eval_reward', eval_reward)
  writer = tf.summary.FileWriter('{}{}_summary'.format(filePath, envName), sess.graph)

  sess.run(tf.global_variables_initializer())
  for it in range(C['pre_iterations']):
    train_a = agent.act_pre()
    ns, train_r, train_d, _ = train_env.step(train_a)
    agent.record(train_s, train_a, train_r, float(train_d), it, True)
    train_s = ns
    if train_d:
      train_s = train_env.reset()

  logger.log('Pre-training completed')

  agent.net.initialize_online_network()
  train_track = [0.0]
  eval_track = []
  best_reward = 0

  train_fs = reset_fs()
  train_s = train_env.reset()
  best_reward = 0
  train_mean = []
  eval_mean = []


  agent.net.update_target_network()

  for it in range(C['iterations']):

    train_fs.append(train_s)

    train_a = agent.act(np.transpose(train_fs, (1,2,0)))
    ns, train_r, train_d, _ = train_env.step(train_a)
    #print('Iteration ',it, ' Reward ', train_r)
    train_track[-1]+= train_r
    agent.record(train_s, train_a, train_r, float(train_d), it, False)
    train_s = ns

    if train_d:
      if train_env.env.env.was_real_done:
        if len(train_track) % 100 == 0:
          train_mean.append(np.mean(train_track[-100:]))
          summary = sess.run(train_summary, feed_dict={train_reward:np.mean(train_track[-100:])})
          writer.add_summary(summary, it)
          logger.record_tabular('steps', it)
          logger.record_tabular('episode', len(train_track))
          logger.record_tabular('epsilon', 100*agent.epsilon)
          logger.record_tabular('learning rate', agent.lr)
          logger.record_tabular('Mean Reward 100 episdoes', np.mean(train_track[-100:]))
          logger.dump_tabular()
          with open(resultPath + 'reward_atari_practice.pk1', 'wb') as f:
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
          with open(resultPath + 'video_atari_practice.pk1', 'wb') as f:
              pickle.dump(best_video, f, protocol=pickle.HIGHEST_PROTOCOL)

      eval_mean.append(np.mean(eval_track[-C['eval_episodes']:]))
      logger.log('Evaluate mean reward: {:.2f}, max reward: {:.2f}, std: {:.2f}'.format(np.mean(eval_track[-C['eval_episodes']:]), np.max(eval_track[-C['eval_episodes']:]), np.std(eval_track[-C['eval_episodes']:])))
      summary = sess.run(eval_summary, feed_dict={eval_reward:np.mean(eval_track[-C['eval_episodes']:])})
      writer.add_summary(summary, it)
      with open(resultPath + 'eval_reward_atari_practice.pk1', 'wb') as f:
          pickle.dump(eval_track, f, protocol=pickle.HIGHEST_PROTOCOL)

    """if it%1000000 == 0:
      outputs = agent.net.get_outputs(np.transpose(train_fs, (1,2,0)))
      with open(resultPath+str(it)+'outputs.pk1', 'wb') as f:
        pickle.dump(outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
      with open(resultPath+str(it)+'outputs_screen.pk1', 'wb') as f:
        pickle.dump(train_fs, f, protocol=pickle.HIGHEST_PROTOCOL)"""

    if it%1000000 == 0:
      weights = agent.net.get_weights()
      with open(resultPath + str(it) +'_weights.pk1', 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)

  with open(resultPath + 'reward_atari_practice.pk1', 'wb') as f:
    pickle.dump(train_track, f, protocol=pickle.HIGHEST_PROTOCOL)
  with open(resultPath + 'trainMean_atari_practice.pk1', 'wb') as f:
    pickle.dump(train_mean, f, protocol=pickle.HIGHEST_PROTOCOL)
  with open(resultPath+ 'evalMean_atari_practice.pk1', 'wb') as f:
    pickle.dump(eval_mean, f, protocol=pickle.HIGHEST_PROTOCOL)
  agent.net.save(filePath + '{}_model2'.format(C['env_id']))
  sess.close()


if __name__ == '__main__':
	main()
