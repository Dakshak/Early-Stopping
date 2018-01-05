import numpy as np
from utilities import *
from layers import *

class two_layered_NN(object):
    '''
    The two_layered_nn contains all the logic required to implement a two layered neural network.
    '''
    def __init__(self,data,**kwargs):
        '''
        Create an instance to initialize all variables
        '''
        self.X_train = data['train']
        self.Y_train = data['tr_targets']
        self.X_val = data['val']
        self.Y_val = data['tr_val']
        
        #self.learning_rate = kwargs.pop('learnin_rate', 0.1)
        self.batch_size = kwargs.pop('batch_size', 20)
        self.num_iterations = kwargs.pop('num_iterations',1000)
        self.regularization_type = kwargs.pop('regularization_type','None')
        self.hidden_layer_size = kwargs.pop('hidden_layer_size',200)
        self.reg = kwargs.pop('reg',0.1)
        self.probability = kwargs.pop('probability',0.5)
        self.classes = kwargs.pop('classes',10)
        self.param_history = kwargs.pop('param_history',0)
        self.decay_rate = kwargs.pop('decay_rate',0.95)
        self.optimization = kwargs.pop('optimization','sgd')
        self.parameters = {}
        self.parameters['w1'] = 0.3 * np.random.randn(np.int(self.X_train.shape[1]), self.hidden_layer_size)
        self.parameters['b1'] = np.zeros(self.hidden_layer_size)
        self.parameters['w2'] = 0.3 * np.random.randn(self.hidden_layer_size, self.classes)
        self.parameters['b2'] = np.zeros(self.classes)
        self.config={}
        self.param = {}
        self.config['w1']={}
        self.config['b1']={}
        self.config['w2']={}
        self.config['b2']={}
        self.param['count']=0
        self.param['last']=1000000
        
    
    def forward_pass(self,X,mode):
        '''
        Function to implement forward pass
        '''
        hidden_ip,cache_ip = forward_step(X,self.parameters['w1'],self.parameters['b1'])
        hidden_op,cache_relu = ReLu_forward(hidden_ip)
        if self.regularization_type == 'L2 with dropout' or self.regularization_type == 'dropout':
            hidden_op,cache_dp = dropout_forward(hidden_op,self.probability,mode)
        else:
            cache_dp = None
        out,cache_out = forward_step(hidden_op,self.parameters['w2'],self.parameters['b2'])
        scores = softmax(out)
        return(scores,cache_out,cache_relu,cache_dp,cache_ip)
    
    def backward_pass(self,probs,y_batch,cache_out,cache_relu,cache_dp,cache_ip):
        '''
        Function to implement backward pass
        '''
        if self.regularization_type == 'L2':
            _,W2,_ = cache_out
            _,W1,_ = cache_ip
            loss_,d_out = loss_with_regularization(probs,y_batch,self.reg,W2,W1) 
            dw2,db2,d_ho=backward_step_with_regularization(d_out,cache_out,self.reg,input_layer = False)
            d_hi = ReLu_backward(d_ho,cache_relu)
            dw1,db1,_ = backward_step_with_regularization(d_hi,cache_ip,self.reg,input_layer = True)
        elif self.regularization_type == 'L2 with dropout':
            _,W2,_ = cache_out
            _,W1,_ = cache_ip
            loss_,d_out = loss_with_regularization(probs,y_batch,self.reg,W2,W1) 
            dw2,db2,d_ho=backward_step_with_regularization(d_out,cache_out,self.reg,input_layer = False)
            d_ho = dropout_backward(d_ho,cache_dp)
            
            d_hi = ReLu_backward(d_ho,cache_relu)
            dw1,db1,_ = backward_step_with_regularization(d_hi,cache_ip,self.reg,input_layer = True)
        elif self.regularization_type == 'dropout':
            loss_,d_out = loss(probs,y_batch) 
            dw2,db2,d_ho=backward_step(d_out,cache_out,input_layer = False)
            d_ho = dropout_backward(d_ho,cache_dp)
            d_hi = ReLu_backward(d_ho,cache_relu)
            dw1,db1,_ = backward_step(d_hi,cache_ip,input_layer = True)
        else:
            loss_,d_out = loss(probs,y_batch) 
            dw2,db2,d_ho=backward_step(d_out,cache_out,input_layer = False)
            d_hi = ReLu_backward(d_ho,cache_relu)
            dw1,db1,_ = backward_step(d_hi,cache_ip,input_layer = True)
        
        return(loss_,dw2,db2,dw1,db1)
    
    def predict(self,X_batch,mode):
        '''
        Function to predict the label for give input
        '''
        probs,_,_,_,_ = self.forward_pass(X_batch,mode)
        y_pred = np.argmax(probs, axis = 1)
        return (y_pred)
    
    def train(self):
        '''
        Function to train the two layered neural network
        '''
        grads={}
        loss_,grads['w1'],grads['b1'],grads['w2'],grads['b2']=None,None,None,None,None
        loss_history=[]
        train_acc_history=[]
        val_acc_history=[]
        
        for it in range(self.num_iterations):
            idx = np.random.choice(np.int(self.X_train.shape[0]), np.int(self.batch_size), replace=True)
            X_batch = self.X_train[idx]
            y_batch = self.Y_train[idx]
            # The following steps implement the forward and backward pass
            
            probs,cache_out,cache_relu,cache_dp,cache_ip = self.forward_pass(X_batch,'train')
            loss_,grads['w2'],grads['b2'],grads['w1'],grads['b1'] =  self.backward_pass(probs,y_batch,cache_out,cache_relu,cache_dp,cache_ip)
            loss_history.append(loss_)
            #Now update the weights and biases in paramaters
            
            for wt,wt_val in self.parameters.items():
                self.parameters[wt],self.config[wt] = update(wt_val,grads[wt],self.optimization,self.config[wt])
                
            train_acc = (self.predict(X_batch,'train') == y_batch).mean()
            val_acc = (self.predict(self.X_val,'train') == self.Y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            probs_val,cache_out_val,cache_relu_val,cache_dp_val,cache_ip_val = self.forward_pass(self.X_val,'train')
            loss_val,_ =  loss(probs_val,self.Y_val) 
            #if self.regularization_type == 'early_stop':
            
            if it % 10 == 0:
                print ('iteration '+str(it) + ' / '+ str(self.num_iterations) +' :loss ' + str(loss_))
                print('training accuracy: '+ str(train_acc) + ' and validation accuracy: '+ str(val_acc) + '| validation loss: '+ str(loss_val))
                self.param = early_stop(loss_val,self.param)
                
                ## Early stopping implementation
                if self.param['count']>=3: # 3 is the patience limit
                    print('Overfitting point reached')
                    break
        return (self.parameters,loss_history,train_acc_history,val_acc_history)  
        
   