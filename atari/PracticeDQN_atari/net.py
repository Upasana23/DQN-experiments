import numpy as np
import tensorflow as tf


#Parametric Rectified Linear Unit
def PReLU(x, alpha_name, name, init=0.001):
    with tf.variable_scope(name):
        _init = tf.constant_initializer(init)
        alpha = tf.get_variable(alpha_name, [], initializer=_init)
        x = tf.multiply(((1 + alpha)*x + (1 - alpha)*tf.abs(x)), 0.5)
    return x

class Net:
    def __init__(self, n_state, n_action, config):
        self.C = config
        self.n_state = n_state + [self.C['frame_stack']]
        self.n_action = n_action
        self.n_delta_s = np.prod([n_action] + self.n_state)
        self.n_delta_s_a = np.prod(self.n_state)
        self._make()
        
        
    def _make_ph(self):
        self.s_ph = tf.placeholder(tf.uint8, [None]+self.n_state) #state(frame stack)
        self.a_ph = tf.placeholder(tf.int32, [None]) #action
        self.r_ph = tf.placeholder(tf.float32, [None]) #reward
        self.d_ph = tf.placeholder(tf.float32, [None]) #destination (boolean)
        self.ns_ph = tf.placeholder(tf.uint8, [None] + self.n_state) #next state
        self.lr_ph = tf.placeholder(tf.float32, []) #learning rate


    #Builds Practice network with 3 Convolution layers, 1 Dense layer and 1 Output layer    
    def _build_net_pt(self, inp, scope_name1, scope_name2):
        inp = tf.cast(inp, tf.float32)/255.0
        with tf.variable_scope(scope_name1, reuse=tf.AUTO_REUSE):
            l = tf.layers.conv2d(inputs=inp, filters=self.C['filter1'], kernel_size=self.C['size1'], strides=self.C['strides1'], padding ="same", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))
            l1 = PReLU(l, 'alpha', 'PReLU')
            l = tf.layers.conv2d(inputs=l1, filters=self.C['filter2'], kernel_size=self.C['size2'], strides=self.C['strides2'], padding ="same",  kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))
            l2 = PReLU(l, 'alpha_1', 'PReLU_1')
            l = tf.layers.conv2d(inputs=l2, filters=self.C['filter3'], kernel_size=self.C['size3'], strides=self.C['strides3'], padding ="same", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))
            l3 = PReLU(l, 'alpha_2', 'PReLU_2')
        with tf.variable_scope(scope_name2, reuse=tf.AUTO_REUSE):
            l = tf.layers.dense(tf.layers.flatten(l3), units=self.C['units1'])
            #Additional Dense layers
            #l = tf.layers.dense(l1, units=self.C['units2'])  
            #l = tf.layers.dense(l, units=self.C['units3'])
            #l = tf.layers.dense(l, units=self.C['units4'])
            l = tf.nn.leaky_relu(l, alpha=0.01)
            delta_s = tf.layers.dense(l, self.n_delta_s)
            return delta_s, l1, l2, l3
          
    #Builds RL training network with 3 Convolution layers, 1 Dense layer and 1 Output layer      
    def _build_net(self, inp, scope_name1, scope_name2):
        inp = tf.cast(inp, tf.float32)/255.0
        with tf.variable_scope(scope_name1, reuse=tf.AUTO_REUSE):
            l = tf.layers.conv2d(inputs=inp, filters=self.C['filter1'], kernel_size=self.C['size1'], strides=self.C['strides1'], padding ="same", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))
            l1 = PReLU(l, 'alpha', 'PReLU')
            l = tf.layers.conv2d(inputs=l1, filters=self.C['filter2'], kernel_size=self.C['size2'], strides=self.C['strides2'], padding ="same",  kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))
            l2 = PReLU(l, 'alpha_1', 'PReLU_1')    
            l = tf.layers.conv2d(inputs=l2, filters=self.C['filter3'], kernel_size=self.C['size3'], strides=self.C['strides3'], padding ="same", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))           
            l3 = PReLU(l, 'alpha_2', 'PReLU_2')
        with tf.variable_scope(scope_name2, reuse=tf.AUTO_REUSE):
            l = tf.layers.dense(tf.layers.flatten(l3), units=self.C['units1'])
            #Additional Dense layers 
            #l = tf.layers.dense(l3, units=self.C['units2']) 
            #l = tf.layers.dense(l, units=self.C['units3'])
            #l = tf.layers.dense(l, units=self.C['units4'])
            l = tf.nn.leaky_relu(l, alpha=0.01)
            Q = tf.layers.dense(l, self.n_action)
            return Q, l1, l2, l3

    #Builds the graph        
    def _build_graph(self):
        s = tf.transpose(tf.reshape(self.delta_s, [-1, self.n_action, self.n_delta_s_a]), [2, 0, 1])
        s_reduced = tf.reduce_sum(s * tf.one_hot(self.a_ph, self.n_action),2)
        pred_s = tf.reshape(tf.transpose(s_reduced), [-1]+self.n_state)
        pred = tf.reduce_sum(self.s_Q * tf.one_hot(self.a_ph, self.n_action), 1) 
        best_v = tf.reduce_max(self.target_Q, 1)
        
        with tf.variable_scope('loss'):
            target = tf.clip_by_value(self.r_ph, -1, 1) + (1.-self.d_ph)*self.C['discount_factor']*tf.stop_gradient(best_v) #target for RL training
            target_s = self.ns_ph - self.s_ph #target for Practice
            loss = tf.losses.huber_loss(target, pred, reduction = tf.losses.Reduction.MEAN) #loss for RL training    
            loss_s = tf.losses.mean_squared_error(target_s, pred_s) #loss for Practice
            
        return loss, loss_s
      
    #Train variables  
    def _build_train(self):
        optimizer_pre = tf.train.AdamOptimizer(self.lr_ph) #optimizer for Practice
        optimizer = tf.train.AdamOptimizer(self.lr_ph) #optimizer for RL training
        
        if self.C['grad_clip'] is None:
            self.optimize_op = optimizer.minimize(self.loss)
        else:
            grads = optimizer.compute_gradients(self.loss, var_list=self.s_Q_global_params1+self.s_Q_global_params2) 
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (tf.clip_by_norm(grad, self.C['grad_clip']), var)
            self.optimize_op = optimizer.apply_gradients(grads)

        self.optimize_init = optimizer_pre.minimize(self.loss_s) 
        
        ops_init = []
        for o,t in zip(self.init_global_params1, self.s_Q_global_params1):
            ops_init.append(t.assign(o))
        
        self.init_online_op = tf.group(*ops_init) #operation for intializing the RL training network with weights from Practice network
        
        ops = []
        for o,t in zip(self.s_Q_global_params1, self.target_Q_global_params1):
            ops.append(t.assign(o))
            
        for o,t in zip(self.s_Q_global_params2, self.target_Q_global_params2):
            ops.append(t.assign(o))
            
        self.update_target_op = tf.group(*ops) #operation for updating target network
        
        
        
    def _make(self):
        self._make_ph()
        self.delta_s, self.o1, self.o2,self.o3 = self._build_net_pt(self.s_ph, 'pre-cnn', 'pre-dense') #Practice network
        self.s_Q, self.a1, self.a2,self.a3 = self._build_net(self.s_ph, 'online-cnn', 'online-dense') #Online RL training network
        self.target_Q, self.b1, self.b2,self.b3 = self._build_net(self.ns_ph, 'target-cnn', 'target-dense') #Target Online training network
        
        #Global parameters for the netwroks (Weights and Biases)
        self.init_global_params1 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pre-cnn')]
        self.init_global_params2 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pre-dense')] 
        self.s_Q_global_params1 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online-cnn')]
        self.s_Q_global_params2 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online-dense')]
        self.target_Q_global_params1 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-cnn')]
        self.target_Q_global_params2 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-dense')]

        self.loss, self.loss_s = self._build_graph()
        self._build_train()
        
    def initialize_online_network(self):
        tf.get_default_session().run(self.init_online_op)

    def update_target_network(self):
        tf.get_default_session().run(self.update_target_op)
        
    def action(self, s):
        return np.argmax(tf.get_default_session().run(self.s_Q, feed_dict={self.s_ph:[s]})[0])

    #training function for Practice  
    def pre_train(self, buffer, learning_rate):
        sb, ab, nsb = buffer.samplePT(self.C['batch_size'])
        tf.get_default_session().run(self.optimize_init , feed_dict ={self.s_ph: sb, self.a_ph: ab, self.ns_ph: nsb, self.lr_ph: learning_rate})
    
    #RL training     
    def train(self, buffer, learning_rate):
        sb, ab, rb, db, nsb = buffer.sample(self.C['batch_size'])
        tf.get_default_session().run(self.optimize_op, feed_dict ={self.s_ph: sb, self.a_ph: ab, self.r_ph: rb, self.d_ph:db, self.ns_ph: nsb, self.lr_ph: learning_rate})

    #Uitility function to get outputs from Convolution layers
    def get_outputs(self, state):
        o1, o2, o3, a1, a2, a3 = tf.get_default_session().run([self.o1, self.o2,self.o3, self.a1, self.a2,self.a3], feed_dict = {self.s_ph: [state]})
        return [o1, o2, o3, a1, a2, a3]
        
    def save(self, directory):
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), directory+'/model.ckpt')
        
    def load(self, directory):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), directory+'/model.ckpt')