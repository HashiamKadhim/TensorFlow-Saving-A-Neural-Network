import tensorflow as tf
import numpy as np
import pickle


'''
Sample code to demonstrate how to save your neural network model for training at a later time.

Data is loaded from a pckle file for this example. 
'''

f = open('data_for_NN_starters_mins.pckl', 'rb')
train_x, train_y, test_x, test_y, inpoutsize = pickle.load(f)
f.close()

#number of features for the input. In this case its 4. 
input_size=inpoutsize-1

test_y=[[xxx] for xxx in test_y]

#number of nodes for each hidden layer. This is no where near the optimal structure for this problem. It is just for illustration purposes. 
n_nodes_hl1=4
n_nodes_hl2=4

#number of classes 
n_classes=1
batch_size=100

#height x width
x=tf.placeholder('float',[None,input_size])
y=tf.placeholder('float', [None, 1])



def neural_network_model(data):

    #defining hidden layers
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([input_size,n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}


    # (input_data*weights)+biases
    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1=tf.nn.relu(l1)

    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2=tf.nn.relu(l2)

    output=tf.add(tf.matmul(l2,output_layer['weights']), output_layer['biases'])


    return output, hidden_1_layer['weights'], hidden_1_layer['biases'],hidden_2_layer['weights'], hidden_2_layer['biases'],output_layer['weights'], output_layer['biases']


def train_neural_network(x,hm_epochs=0,restore=False,run=False, save=False, checkpoint_file=None, regularization_constant=0.05):
    
    with tf.Session() as sess:
        
        prediction, hlw1,hlb1,hlw2,hlb2,olw,olb=neural_network_model(x)
        num_train_sets=len(train_x)

        #Different Cost Functions to choose from.
        #mean square error
        cost=tf.reduce_mean( tf.square(tf.sub(prediction,y)))

        #mean error
        cost2=tf.reduce_mean( tf.abs(tf.sub(prediction,y)))

        #mean square error with L^1 Regularization on weights and biasis
        cost3=tf.reduce_mean( tf.square(tf.sub(prediction,y)))+(regularization_constant/num_train_sets)*(tf.reduce_mean(tf.abs(hlw1))+ tf.reduce_mean( tf.abs(hlb1))+
            tf.reduce_mean(tf.abs(hlw2))+tf.reduce_mean(tf.abs(hlb2))+
            tf.reduce_mean(tf.abs(olw))+tf.reduce_mean(tf.abs(olb)))

        #mean square error with L^2 Regularization on weights and biasis
        cost4=tf.reduce_mean( tf.square(tf.sub(prediction,y)))+(regularization_constant/num_train_sets)*(tf.nn.l2_loss(hlw1)+ tf.nn.l2_loss( hlb1)+
            tf.nn.l2_loss(hlw2)+tf.nn.l2_loss (hlb2)+
            tf.nn.l2_loss(olw)+tf.nn.l2_loss(olb))

        #cross entropy
        cost5=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y)) 

                              
        #Here we define the optimizer. Default learning_rate=0.001
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost4)

        #initializing saver
        saver=tf.train.Saver()

        #Restoring or running a new model
        if restore:
            print("Loading variables from '%s'." % checkpoint_file)
            saver.restore(sess, checkpoint_file)
            print 'restored'

        else:
            sess.run(tf.initialize_all_variables())

        #training
        if run:
            epoch=1
            
            while epoch <=hm_epochs:
                epoch_loss=0
                epoch_loss2=0   
                i=0
                batch_size=100
                while i<len(train_x):
                    batch_size=int(batch_size)
                    start=i
                    end=i+batch_size
                    batch_x=np.array(train_x[start:end])
                    batch_y=np.array([[xxx] for xxx in train_y[start:end]])
                    _,c,c2=sess.run([optimizer,cost,cost2],feed_dict={x:batch_x, y:batch_y})
                    epoch_loss+=c
                    epoch_loss2+=c2
                    batch_size=int(batch_size)
                    i=i+batch_size

                correct= tf.abs(tf.sub(prediction,y))

                accuracy = tf.reduce_mean(tf.cast(correct,'float'))
                print 'Epoch:', epoch, 'completed out of:', hm_epochs, 'loss:', epoch_loss, 'Train Mean Error: ', accuracy.eval({x:train_x,y:[[xxx] for xxx in train_y]}),'Test Mean Error: ', accuracy.eval({x:test_x,y:test_y})      
                epoch+=1

        if save:
            print("Saving variables to '%s'." % checkpoint_file)
            saver.save(sess,checkpoint_file)


        
    


train_neural_network(x,hm_epochs=5000,restore=0,run=True, save=True,checkpoint_file='Minutes_PGs.ckpt')






    


    
