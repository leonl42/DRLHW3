
import gym
import tensorflow as tf
import random as rd
import numpy as np
def create_trajectory(model,render = False,epsilon = 0.7):
    """
    create trajectory given the model
    """
    env = gym.make("LunarLander-v2")
    env.action_space.seed(42)

    s_a_r_s = []

    observation, _ = env.reset(seed=42, return_info=True)

    for _ in range(1000):
        action = model(tf.cast(tf.expand_dims(observation,axis = 0),tf.float32))

        # epsilon greedy policy
        if rd.randint(0,100)<epsilon*100:
            new_observation, reward, done, _ = env.step(tf.argmax(tf.squeeze(action,axis = 0)).numpy())
        else:
            new_observation, reward, done, _ = env.step(rd.randint(0,3))
        s_a_r_s.append((tf.convert_to_tensor(observation),tf.argmax(tf.squeeze(action,axis = 0)),tf.convert_to_tensor(reward),tf.convert_to_tensor(new_observation)))
        observation = new_observation
        if render:
            env.render()

        if done:
            observation, _ = env.reset(return_info=True)

    env.close()

    return s_a_r_s

def sample_minibatch(data,batch_size):
    """
    return a minibatch sampled from the buffer
    """
    s_batch = tf.TensorArray(tf.float32,size = batch_size)
    a_batch = tf.TensorArray(tf.float32,size = batch_size)
    r_batch = tf.TensorArray(tf.float32,size = batch_size)
    s_new_batch = tf.TensorArray(tf.float32,size = batch_size)
    for i in range(batch_size):
        element = rd.choice(data)
        s,a,r,s_new = element

        # cast all elements to floats
        r = tf.cast(r,tf.float32)
        s = tf.cast(s,tf.float32)
        a = tf.cast(a,tf.float32)
        s_new = tf.cast(s_new,tf.float32)

        # add them to the tensor array
        s_batch = s_batch.write(i,s)
        a_batch = a_batch.write(i,a)
        r_batch = r_batch.write(i,r)
        s_new_batch = s_new_batch.write(i,s_new)
    
    # stack for batch dimension
    return s_batch.stack(),a_batch.stack(),r_batch.stack(),s_new_batch.stack()




model = tf.keras.Sequential(
    [tf.keras.layers.Dense(100,activation = "relu"),
    tf.keras.layers.Dense(100,activation = "relu"),
    tf.keras.layers.Dense(10,activation = "relu"),
    tf.keras.layers.Dense(4,activation = "relu")]
)

# target network used with polyak averaging
model_target = tf.keras.models.clone_model(model)

# initialize weights by creating trajectory
create_trajectory(model)
create_trajectory(model_target)

buffer = []
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
loss_function = tf.keras.losses.MeanSquaredError()
epsilon = 0.7
for i in range(1000):

    # apply polyak averaging
    model_target.set_weights((1-0.1)*np.array(model_target.get_weights(),dtype = object) + 0.1*np.array(model.get_weights(),dtype = object))
    
    # add new data to replay buffer
    new_data = create_trajectory(model, True if i%50 == 0 else False,epsilon)
    epsilon += 0.001
    reward = []
    for s,a,r,new_s in new_data:
        reward.append(tf.cast(r,tf.float32))
    print("round: ", i," average reward: ",tf.reduce_mean(reward))
    buffer.extend(new_data)

    # remove old data from replay buffer if too many alements are in it
    while len(buffer)>100000:
        buffer = buffer[1:]
    
    for _ in range(100):

        s,a,r,s_new = sample_minibatch(buffer,128)
        with tf.GradientTape() as tape:

            # calculate the corresponding q values
            Q_max = tf.math.reduce_max(model_target(s_new),axis=-1)
            Q_s_a = tf.gather(params = model(s),indices = tf.cast(a,tf.int32),axis = -1,batch_dims = 1)

            # apply mean squared error loss
            loss = loss_function(Q_s_a, r + tf.constant(0.99)*Q_max)

        # perform gradient descent step
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    
