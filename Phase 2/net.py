# Tranfer Learning network
import numpy as np
import tensorflow as tf



def PReLU(x, alpha_name, name, init=0.001):
    with tf.variable_scope(name):
        _init = tf.constant_initializer(init)
        alpha = tf.get_variable(alpha_name, [], initializer=_init)
        x = tf.multiply(((1 + alpha)*x + (1 - alpha)*tf.abs(x)), 0.5)
    return x

class Net:
    def __init__(self, n_state, n_action, config, wt):
        self.C = config
        self.n_state = n_state + [self.C['frame_stack']]
        #self.n_state = n_state
        self.n_action = n_action
        #self.n_delta_s = np.prod([n_action] + n_state)
        self.wt_cnn = wt
        self._make()
        
        
    def _make_ph(self):
        self.s_ph = tf.placeholder(tf.uint8, [None]+self.n_state) #state(frame stack)
        self.a_ph = tf.placeholder(tf.int32, [None]) #action
        self.r_ph = tf.placeholder(tf.float32, [None]) #reward
        self.d_ph = tf.placeholder(tf.float32, [None]) #destination (boolean)
        self.ns_ph = tf.placeholder(tf.uint8, [None] + self.n_state) #next state
        self.lr_ph = tf.placeholder(tf.float32, []) #learning rate
                  
          
    def _build_net(self, inp, scope_name1, scope_name2):
        inp = tf.cast(inp, tf.float32)/255.0
        with tf.variable_scope(scope_name1, reuse=tf.AUTO_REUSE):
            l = tf.layers.conv2d(inputs=inp, filters=self.C['filter1'], kernel_size=self.C['size1'], strides=self.C['strides1'], padding ="same", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))
            l1 = PReLU(l, 'alpha', 'PReLU')        
            l = tf.layers.conv2d(inputs=l1, filters=self.C['filter2'], kernel_size=self.C['size2'], strides=self.C['strides2'], padding ="same",  kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))
            l2 = PReLU(l, 'alpha_2', 'PReLU_1')    # possible
            l = tf.layers.conv2d(inputs=l2, filters=self.C['filter3'], kernel_size=self.C['size3'], strides=self.C['strides3'], padding ="same", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(2.0))
            l3 = PReLU(l, 'alpha_3', 'PReLU_2')   # possible

        with tf.variable_scope(scope_name2, reuse=tf.AUTO_REUSE):
            l = tf.layers.dense(tf.layers.flatten(l3), units=self.C['units1'])
            #l = tf.layers.dense(l3, units=self.C['units2'])
            #l = tf.layers.dense(l, units=self.C['units3'])
            #l = tf.layers.dense(l, units=self.C['units4'])
            l = tf.nn.leaky_relu(l, alpha=0.01)
            Q = tf.layers.dense(l, self.n_action)
            return Q, l1, l2, l3

            
    def _build_graph(self):
        pred = tf.reduce_sum(self.s_Q * tf.one_hot(self.a_ph, self.n_action), 1)
        best_v = tf.reduce_max(self.target_Q, 1)
        """if self.C['double']:
            self.ns_Q = self._build_net(self.ns_ph, 'online-cnn','online-dense')
            choice = tf.argmax(self.ns_Q, 1)
            best_v = tf.reduce_sum(self.target_Q * tf.one_hot(choice, self.n_action), 1)
        else:
            best_v = tf.reduce_max(self.target_Q, 1)"""


        with tf.variable_scope('loss'):
            target = tf.clip_by_value(self.r_ph, -1, 1) + (1.-self.d_ph)*self.C['discount_factor']*tf.stop_gradient(best_v)
            loss = tf.losses.huber_loss(target, pred, reduction = tf.losses.Reduction.MEAN)

        return loss
      
    def _build_train(self):
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
	    #optimizer = tf.train.RMSPropOptimizer(self.lr_ph, epsilon = 1e-2)
        
        if self.C['grad_clip'] is None:
            self.optimize_op = optimizer.minimize(self.loss)
        else:
            grads = optimizer.compute_gradients(self.loss, var_list=self.s_Q_global_params1+self.s_Q_global_params2) # possible
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (tf.clip_by_norm(grad, self.C['grad_clip']), var)
            self.optimize_op = optimizer.apply_gradients(grads)

        ops_init = []
        for o,t in zip(self.wt_cnn, self.s_Q_global_params1):
            ops_init.append(t.assign(o))

        self.init_online_op = tf.group(*ops_init)

        ops = []
        for o,t in zip(self.s_Q_global_params1, self.target_Q_global_params1):
            ops.append(t.assign(o))
            
        for o,t in zip(self.s_Q_global_params2, self.target_Q_global_params2):
            ops.append(t.assign(o))
            
        self.update_target_op = tf.group(*ops) #operation for updating target network


        
    def _make(self):
        self._make_ph()
        self.s_Q, self.o1, self.o2,self.o3 = self._build_net(self.s_ph, 'online-cnn', 'online-dense')
        self.target_Q, self.a1, self.a2,self.a3 = self._build_net(self.ns_ph, 'target-cnn', 'target-dense')
        self.s_Q_global_params1 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online-cnn')]
        self.s_Q_global_params2 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online-dense')]
        self.target_Q_global_params1 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-cnn')]
        self.target_Q_global_params2 = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-dense')]
        self.loss = self._build_graph() 
        self._build_train()

    def update_target_network(self):
        tf.get_default_session().run(self.update_target_op)

    def action(self, s):
        return np.argmax(tf.get_default_session().run(self.s_Q, feed_dict={self.s_ph:[s]})[0])
        
    def train(self, buffer, learning_rate):
        sb, ab, rb, db, nsb = buffer.sample(self.C['batch_size'])
        tf.get_default_session().run(self.optimize_op, feed_dict ={self.s_ph: sb, self.a_ph: ab, self.r_ph: rb, self.d_ph:db, self.ns_ph: nsb, self.lr_ph: learning_rate})
        # return Q

    def get_outputs(self, state):
        o1, o2, o3 = tf.get_default_session().run([self.o1, self.o2,self.o3], feed_dict = {self.s_ph: [state]})
        return [o1, o2, o3] 
      
    def get_weights(self):
        w1, w2 = tf.get_default_session().run([self.target_Q_global_params1, self.target_Q_global_params2])
        return [w1, w2]
          
    def load_weights(self):
        tf.get_default_session().run(self.init_online_op)
    
    def save(self, directory):
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), directory+'/model.ckpt')
        
    def load(self, directory):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), directory+'/model.ckpt')