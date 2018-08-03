from unityagents import UnityEnvironment
import os
import tensorflow as tf
from DQN import DeepQNetwork

### Parameters
run_path = "Task14"
summary_freq = 10
maxEpisode = 150000
learning_freq = 20

env = UnityEnvironment()
print(str(env))

brain_name = env.external_brain_names[0]

tf.reset_default_graph()

summary_path = "./summaries/{}".format(run_path)

if not os.path.exists(summary_path):
	os.makedirs(summary_path)

##Q2 Start
RL = DeepQNetwork(4,
                  8,
                  learning_rate=0.001,
                  reward_decay=0.99,
                  e_greedy=0.975,
                  replace_target_iter=4,
                  memory_size=10000,
                  e_greedy_increment=None)
##Q2 End

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
summary_writer = tf.summary.FileWriter(summary_path)

def PrintAndSaveSummary(writer, episode, episodeStep, episodeReward, epsilon,lr):
    print('Epsisode:', episode,
          'length =', episodeStep)
    summary = tf.Summary()
    summary.value.add(tag="Cumulated　Reward",
                      simple_value=episodeReward)
    summary.value.add(tag="Epsilon",
                      simple_value=epsilon)
    summary.value.add(tag="Learning Rate",
                      simple_value=lr)
    summary.value.add(tag="Episode Length",
                      simple_value=episodeStep)
    writer.add_summary(summary, episode)
    writer.flush()


with tf.Session(config=config) as sess:
    sess.run(init)
    totalStep = 0
    for episode in range(maxEpisode):
        # initial observation
        episodeStep = 0
        episodeReward = 0
        info = env.reset()[brain_name]
        state = info.states[0]
        while True:
            action = RL.choose_action(state)
            new_info = env.step({brain_name:[action]})[brain_name]
            RL.store_transition(state, action, new_info.rewards[0], new_info.states[0])
            episodeReward += new_info.rewards[0]
            if (totalStep > 200) and (totalStep % learning_freq == 0):
                RL.learn()
                state = new_info.states[0]
                if new_info.local_done[0]:
                    break
            totalStep += 1
            episodeStep += 1
        if (episode % summary_freq == 0):
            PrintAndSaveSummary(summary_writer,
                                episode,
                                episodeStep,
                                episodeReward,
                                RL.epsilon,RL.lr)
        episodeReward = 0
    print('Training Completed')

