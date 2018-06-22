import tensorflow as tf
#Load dataset from tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/temp/data/",one_hot=True)
#It is a DNN so I have added 3 hidden layers 
layer_1=500
layer_2=500
layer_3=500
n_classes=10
batch_size=100
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')
#Create a neural network model
def nn(data):
    hl1={'weights':tf.Variable(tf.random_normal([784,layer_1])),
         'biases':tf.Variable(tf.random_normal([layer_1]))}
    hl2={'weights':tf.Variable(tf.random_normal([layer_1,layer_2])),
         'biases':tf.Variable(tf.random_normal([layer_2]))}
    hl3={'weights':tf.Variable(tf.random_normal([layer_2,layer_3])),
         'biases':tf.Variable(tf.random_normal([layer_3]))}
    output={'weights':tf.Variable(tf.random_normal([layer_3,n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}
    l1=tf.add(tf.matmul(data,hl1['weights']),hl1['biases'])
    l1=tf.nn.relu(l1)
    l2=tf.add(tf.matmul(l1,hl2['weights']),hl2['biases'])
    l2=tf.nn.relu(l2)
    l3=tf.add(tf.matmul(l2,hl3['weights']),hl3['biases'])
    l3=tf.nn.relu(l3)
    output_l=tf.matmul(l3,output['weights'])+output['biases']
    return output_l
    
    
                                                
        
def train(x):
    pred=nn(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    hm_epoch=4
    with tf.Session() as ses:
        ses.run(tf.global_variables_initializer())
        for epoch in range( hm_epoch):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                           epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                           _,c=ses.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                           epoch_loss+=c
            print('epoch',epoch,"completed out of :",hm_epoch,"loss:",epoch_loss)
        correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,"float"))
        print("acc:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
train(x)
