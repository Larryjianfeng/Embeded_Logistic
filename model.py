#encoding=utf-8
import numpy as np 
import tensorflow as tf 
import pickle
import random 
model_dir = '/home/yanjianfeng/kaggle/data/model_dir/'


people_dic, group_dic, dic = pickle.load(open('/home/yanjianfeng/kaggle/data/data.dump', 'r'))
def create_train_op(loss):
    train_op = tf.contrib.layers.optimize_loss(loss = loss, 
        global_step = tf.contrib.framework.get_global_step(), 
        learning_rate = 0.1, 
        clip_gradients = 10.0, 
        optimizer = "Adam")
    return train_op 

def create_input():
    random_id = random.randint(0, len(dic['outcome'])-2049)
    keys = dic.keys() 
    data = {}
    for k in keys:
        data[k] = dic[k][random_id: random_id+2048]
    return data


# 主体部分还是最好不要放在函数里面，不太容易提取出某个特定的值
# 或者直接把主体部分放在tf.Session里面比较容， 大概就是这么一个模式；


global_step = tf.Variable(0, name = 'global_step', trainable=False)

people_id = tf.placeholder("int64", [None])
group = tf.placeholder('int64', [None])
time = tf.placeholder('int64', [None])
peofea = tf.placeholder('int64', [None, 262])
rowfea = tf.placeholder('int64', [None, 174])
outcome = tf.placeholder("int64", [None])

name_embed = tf.get_variable('names', shape = [189120, 10])
group_embed = tf.get_variable('groups', shape = [35000, 10])
name_ = tf.nn.embedding_lookup(name_embed, people_id)
group_ = tf.nn.embedding_lookup(group_embed, group)

name_w = tf.get_variable('name_w', shape = [10, 2])
group_w = tf.get_variable('group_w', shape = [10, 5])

name_outcome = tf.matmul(name_, name_w)
group_outcome = tf.matmul(group_, group_w)

w_1 = tf.get_variable('w_1', shape = [262, 10])
w_2 = tf.get_variable('w_2', shape = [174, 10])
w_3 = tf.get_variable('w_3', shape = [1])

peofea_outcome = tf.matmul(tf.to_float(peofea), w_1)
rowfea_outcome = tf.matmul(tf.to_float(rowfea), w_2)

time_outcome = tf.mul(tf.to_float(time), w_3)
time_outcome = tf.expand_dims(time_outcome, -1)

name_outcome = tf.sigmoid(name_outcome)
group_outcome = tf.sigmoid(group_outcome)
peofea_outcome = tf.sigmoid(peofea_outcome)
rowfea_outcome = tf.sigmoid(rowfea_outcome)
time_outcome = tf.sigmoid(time_outcome)

x = tf.concat(1, [name_outcome, group_outcome, peofea_outcome, rowfea_outcome, time_outcome])

w_f = tf.get_variable('w_f', shape = [28, 28])
b = tf.get_variable('b', shape = [1])
w_f_2 = tf.get_variable('w_f_2', shape = [28, 1])

pred = tf.sigmoid(tf.matmul(x, w_f)) + b 
pred = tf.matmul(pred, w_f_2)

y = tf.expand_dims(tf.to_float(outcome), -1)

prob = tf.sigmoid(pred)
prob = tf.to_float(tf.greater(prob, 0.5))
c = tf.reduce_mean(tf.to_float(tf.equal(prob, y)))

loss = tf.nn.sigmoid_cross_entropy_with_logits(pred, y)
loss = tf.reduce_mean(loss)
train_op = create_train_op(loss)



# 这里的顺序很重要，要是在最前面用saver，则会save到最开始的情况？
saver = tf.train.Saver()
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print 'the model being restored is '
        print ckpt.model_checkpoint_path 
        saver.restore(sess, ckpt.model_checkpoint_path)
        print 'sucesssfully restored the session'

    count = global_step.eval()

    for i in range(0, 10000):
        data = create_input()
        l, _ , c_ = sess.run([loss, train_op, c], feed_dict = {people_id: data['people_id'],
            group: data['group'],
            time: data['time'],
            peofea: data['people_features'],
            rowfea: data['row_features'],
            outcome: data['outcome']})
        print 'the loss\t' + str(l) + '\t\tthe count\t' + str(c_)
        global_step.assign(count).eval()
        saver.save(sess, model_dir + 'model.ckpt', global_step = global_step)
        count += 1 

