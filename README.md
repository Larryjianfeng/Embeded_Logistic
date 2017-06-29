--- 灵感 --- 
因为最近一直在做rnn based NLP，其中无论是什么cell，lstm，
 GRU或者cnn都是基于单词的embedding表示；单词的embdding就是把每个单词表示成一个向量，
 然后通过bp训练这些向量的值，这种想法很奇妙，于是我尝试性的把这种思想用在logistic regression上面；

--- 问题 --- 
对于logistic regression的话，很多向量都是categorial，如果碰到有1000个category怎么做？转换成1000*1的one-hot向量吗？
方法：用embedding， 每个category给一个10维的向量，然后再用传统的回归或者神经网络的方法；

--- 实验 --- 
1： 数据一览；
数据来自kaggle， 是redhat那个项目，感兴趣的自己去看看；
2：方法；
标题是逻辑回归，但是本质上还是神经网络做分类；但是这个问题传统上都是用逻辑回归解决的，因为包含了很多categorial的数据，然后label是0和1，要求做分类；
运行一个logistic regression是很简单的；但是这里的问题是数据里面有个group变量和一个people向量，group大概有3k+种类，people大概有180K+种类，显然转换成dummy变量再做逻辑回归的话不合适；这里我主要是参考word embedding的思想，在tensorflow里面建立两个个词典，一个people词典一个group词典，然后训练的时候分别去查这个词典返回两个10维的实数向量，这两个实数向量就分别是people和group的特征；之后再随便弄了一点full connected的层和一些激活函数，效果不错，很快收敛到90%以上了；
3：效果；
这个数据的话，我刚开始只是想用来实验在tf.Session（）的情况下怎么样batch读取tfrecords数据的，因为tfrecords数据读取的话不需要把整个数据load进去内存；之前一直用estimator的方法读tfrecords，但是用session之后似乎没有很好的解决方法；
效果还不错，主要是感觉对于多种类的问题都可以用embedding的方法来做了以后；
