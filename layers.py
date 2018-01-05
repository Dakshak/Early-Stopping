import numpy as np

def forward_step(x,w,b):
    """
    TO DO: Compute the forward pass of the neural network
    Points Allocated: 5
    
    Inputs:
    x: Numpy array of shape (N,d) where N is number of samples 
    and d is the dimension of input
    w: numpy array of shape (d x H) where H is size of hidden layer
    b: It is the bias matrix of shape (H,)
    
    Outputs:
    out: output of shape (N x H)
    cache: values used to calculate the output (x,w,b)
    """
    out = None
    cache = (x,w,b)
    ### Type your code here ###
    
    out = x.dot(w)+b
    
    #### End of your code ####
    
    return (out,cache)


def backward_step(d_out,cache,input_layer = False):
    """
    TO DO: Compute the backward pass of the neural network
    Points Allocated: 15
    
    Inputs:
    d_out: calculated derivatives 
    cache: (x,w,b) values of corresponding d_out values
    input_layer: TRUE/FALSE
    
    Outptus:
    dx: gradients with respect to x
    dw: gradients with respect to w
    db: gradients with respect to b 
    """
    x,w,b = cache
    dx,dw,db = None, None, None
    
    # Note that dx is not valid for input layer. If input_layer is true then dx will just return None
    
    ### Type your code here ###
    if not input_layer:
        dx = d_out.dot(w.T)  
    dw = x.T.dot(d_out)
    db = np.sum(d_out, axis=0)
    
    #### End of your code ####
    
    return (dw,db,dx)

def ReLu_forward(x):
    """
    TO DO: Compute the ReLu activation for forward pass
    Points Allocated: 5
    
    Inputs:
    x: numpy array of any shape
    
    Outputs:
    out : should be same shape as input
    cache: values used to calculate the output (out)
    """
    cache = x
    out = None
    ### Type your code here ###
    
    out = np.maximum(0, x)
    
    #### End of your code ####  
    return (out,cache)

def ReLu_backward(d_out,cache):
    """
    TO DO: Compute the backward pass for ReLu 
    Points Allocated: 15
    
    Inputs: 
    d_out: derivatives of any shape
    cache: has x corresponding to d_out
    
    Outputs:
    dx: gradients with respect to x
    """
    x = cache
    dx = None
    ### Type your code here ###
    
    dx = d_out
    dx[x < 0] = 0
    
    #### End of your code #### 
    return (dx)

def softmax(x):
    """
    TO DO: Compute the softmax loss and gradient for the neural network
    Points Allocated: 25
    
    Inputs:
    x: numpy array of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    out: softmax output of shape 
    loss: loss of forward pass (scalar value)
    dx: gradient of loss with respect to x
    """
    out = None
    ### Type your code here ###
    
    shift_scores = x - np.max(x, axis = 1).reshape(-1,1)
    out = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
    
    #### End of your code ####     
    return (out)

def loss(x,y):
    """
    TO DO: Compute the softmax loss and gradient for the neural network
    Points Allocated: 25
    
    Inputs:
    x: Matrix of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    loss: loss of forward pass (scalar value)
    dx: gradient of loss with respect to x
    """
    loss,de = None,None
    num_train = x.shape[0]
    loss = -np.sum(np.log(x[range(num_train), list(y)]))
    loss /= num_train 
    de= x.copy()
    de[range(num_train), list(y)] += -1 
    de /= num_train
    return (loss,de)

def backward_step_with_regularization(d_out,cache,reg,input_layer = False):
    """
    TO DO: Compute the backward pass of the neural network
    Points Allocated: 15
    
    Inputs:
    d_out: calculated derivatives 
    cache: (x,w,b) values of corresponding d_out values
    input_layer: TRUE/FALSE
    reg: regularization constant
    
    Outptus:
    dx: gradients with respect to x
    dw: gradients with respect to w
    db: gradients with respect to b 
    """
    x,w,b = cache
    dx,dw,db = None, None, None
    
    # Note that dx is not valid for input layer. If input_layer is true then dx will just return None
    
    ### Type your code here ###
    if not input_layer:
        dx = d_out.dot(w.T)  
    dw = x.T.dot(d_out) + reg*w
    db = np.sum(d_out, axis=0)
    
    #### End of your code ####
    
    return (dw,db,dx)

def loss_with_regularization(x,y,reg,W2,W1):
    """
    TO DO: Compute the softmax loss and gradient with regularization for the neural network 
    Points Allocated: 25
    
    Inputs:
    x: Matrix of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    reg: regularization constant
    W1: input layer weights
    W2: output layer weights
    
    Outputs:
    loss: loss of forward pass (scalar value)
    dx: gradient of loss with respect to x
    """
    loss,de = None,None
    num_train = x.shape[0]
    loss = -np.sum(np.log(x[range(num_train), list(y)]))
    loss /= num_train 
    loss +=  reg * 0.5* (np.sum(W1 * W1)+np.sum(W2 * W2))
    de= x.copy()
    de[range(num_train), list(y)] += -1 
    de /= num_train
    return (loss,de)

def dropout_forward(x,probability,mode):
    '''
    Forward step for dropout
    Inputs: 
    x: numpy array of shape (N,D)
    probability: drop probability of a neuron
    mode: train/test as dropout happens only during train
    
    Outputs:
    out: numpy array of same shape as x
    cache: Values to be used during backward pass
    '''
    filter_ = None
    if mode=='train':
        [N,D] = x.shape
        #filter_ = np.random.binomial(1, probability, size=[N,D])
        filter_ = (np.random.rand(N,D) < (1-probability))/(1-probability)
        out = x*filter_
    elif mode=='test':
        out = x
    else:
        raise ValueError("mode must be 'test' or 'train'")
    cache = (probability, filter_,mode)
    out = out.astype(x.dtype, copy=False)
    return(out,cache)

def dropout_backward(grad,cache):
    '''
    Backward step for dropout
    
    Inputs:
    grad: numpy array of shape (N,D)
    cache: Values which are calculated during dropout forward step
    
    Outputs:
    dx: numpy array of shape (N,D)
    '''
    _, filter_, mode = cache
    if mode == 'train':
        dx = grad*filter_
    elif mode == 'test':
        dx = grad
    else:
        raise ValueError("mode must be 'test' or 'train'")
    return(dx)

def update(w,dw,type_,config):
    '''
    Function to update parameters
    Inputs:
    w - initial weight
    dw - gradient for weight w
    Outputs:
    w - updated weight
    config - values used to update the w
    '''
    if type_ == 'sgd':
        if not bool(config): 
            print('Optimization using SGD')
            config.setdefault('learning_rate', 0.1)
        w += -config['learning_rate']* dw
    elif type_ == 'SGD_with_momentum':
        '''
        - learning_rate: Scalar learning rate.
        - momentum: Scalar between 0 and 1 giving the momentum value.
        - velocity: A numpy array of the same shape as w and dw used to store a moving
        average of the gradients.
        '''
        if not bool(config): 
            print('Optimization using SGD and Momentum')
            config.setdefault('learning_rate', 1e-2)
            config.setdefault('momentum', 0.9)
        v = config.get('velocity', np.zeros_like(w)) # Initial Velocity
        
        ### Type your code here ###
        v = config['momentum']* v - config['learning_rate']*dw
        w +=v
        #### End of your code ####
        config['velocity'] = v
    elif type_ == 'rmsprop':
        '''
        - learning_rate: Scalar learning rate.
        - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
        gradient r.
        - delta: Small scalar used for smoothing to avoid dividing by zero.
        - r: Moving average of second moments of gradients.
        '''
        if not bool(config): 
            print('Optimization using RMS prop')
            config.setdefault('learning_rate', 1e-2)
            config.setdefault('decay_rate', 0.99)
            config.setdefault('delta', 1e-8)
            config.setdefault('r', np.zeros_like(w))
        ### Type your code here ###
        config['r'] = config['decay_rate']*config['r']+(1-config['decay_rate'])*(dw*dw)
        w += -config['learning_rate']* dw / (np.sqrt(config['r'])+config['delta'])
        #### End of your code ####
    elif type_ == 'adam':
        '''
        - learning_rate: Scalar learning rate.
        - beta1: Decay rate for moving average of first moment of gradient.
        - beta2: Decay rate for moving average of second moment of gradient.
        - delta: Small scalar used for smoothing to avoid dividing by zero.
        - first_moment: Moving average of gradient.
        - second_moment: Moving average of squared gradient.
        - t: Iteration number.
        '''
        if not bool(config): 
            print('Optimization using ADAM')
            config.setdefault('learning_rate', 1e-2)
            config.setdefault('beta1', 0.9)
            config.setdefault('beta2', 0.999)
            config.setdefault('delta', 1e-8)
            config.setdefault('first_moment', np.zeros_like(w))
            config.setdefault('second_moment', np.zeros_like(w))
            config.setdefault('t', 0)
        ### Type your code here ###
        config['t']+=1 
        config['first_moment'] = config['beta1']*config['first_moment'] + (1- config['beta1'])*dw
        config['second_moment'] = config['beta2']*config['second_moment'] + (1- config['beta2'])*(dw**2)   
        first_moment_b = config['first_moment']/(1-config['beta1']**config['t'])
        second_moment_b = config['second_moment']/(1-config['beta2']**config['t'])
        w = w -config['learning_rate']* first_moment_b / (np.sqrt(second_moment_b) + config['delta'])
        #### End of your code ####
    else:
        raise ValueError("Undefined optimization type")
    return(w,config)

def early_stop(loss,param):
    '''
    Function to calculate the stopping criteria. It increments the count by 1 if current loss is greater than 1 and updates the previous loss with current loss
    Inputs:
    loss: current loss of validation data
    param:
        last: previous loss
        count: count to keep track number of times the loss is greater than previous loss
    '''
    ### Type your code here###
    print('Last Loss'+str(param['last'])+'; Loss Count' +str(param['count']))
    if loss >= param['last']:
        param['count']+=1
    else:
        param['count'] = 0
    param['last']=loss
    ### End of your code ###
    return(param)
    
    
