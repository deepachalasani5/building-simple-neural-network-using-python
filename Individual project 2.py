#!/usr/bin/env python
# coding: utf-8

# In[91]:


from random import random
import math
import csv


# In[92]:


class neuron():
    def __init__(self,av,no_of_weights,index_neuron):
        self.av=av
        self.no_of_weights=no_of_weights
        self.index_neuron=index_neuron
        self.weights=[]
        self.delta_weight=[]
        self.grad_val=0
        for i in range(no_of_weights):
            n=0
            self.delta_weight.append(n)
        for i in range(no_of_weights):
            n=random()
            self.weights.append(n)
    def multiply_weights(self,previous_layer):
        result=0
        for i in range(len(previous_layer)):
                result=result + ((previous_layer[i].av) * (previous_layer[i].weights[self.index_neuron]))
        self.av=1/(1+(math.exp(-result)))


# In[93]:


input_layer=[neuron(0,2,0),neuron(0,2,1),neuron(1,2,2)]
hidden_layer=[neuron(0,2,0),neuron(0,2,1),neuron(1,2,2)]
output_layer=[neuron(0,0,0),neuron(0,0,1)]


# In[94]:


for i in range(len(hidden_layer)):
    hidden_layer[i].multiply_weights(input_layer)
    print(hidden_layer[i].av)


# In[95]:


def feedforword_process(inputs):
    outputs=[]
    for i in range(len(input_layer)-1):
        input_layer[i].av=inputs[i]
    for i in range(len(hidden_layer)-1):
        hidden_layer[i].multiply_weights(input_layer)
    for i in range(len(output_layer)):
        output_layer[i].multiply_weights(hidden_layer)
        outputs.append(output_layer[i].av) 
    return outputs


# In[96]:


inputs=[0.5,0.5]
feedforword_process(inputs)


# In[97]:


lamb=0.5
lr=0.5
mt=0.5
def backprop(outputs):
    error=[0,0]
    #To get the delta weights of input layer and hidden layer , we are supposed to find out gradient value of output layer and hidden layer 
    #In order to find out gradient Value for output layer , we need to find out the error of outputs
    for i in range(len(output_layer)):
        error[i]=outputs[i]-output_layer[i].av
        
    # now , using the error of the outputs, we can find the gradient value of the output layer  
    for i in range(len(output_layer)):
        output_layer[i].grad_var=lamb*output_layer[i].av*(1-output_layer[i].av)*error[i]
        
    #By using the gradient value of the output layer , we can derive the gradient value of the hidden layer
    for i in range(len(hidden_layer)):
        for j in range(len(output_layer)):
            hidden_layer[i].grad_var=lamb*hidden_layer[i].av*(1-hidden_layer[i].av)*(output_layer[j].grad_var*hidden_layer[i].weights[j])
    
    #with the help of gradient value of the output layer ,we can obtain the delta weight of hidden layer  
    for i in range(len(hidden_layer)):
        for j in range(len(output_layer)):
            hidden_layer[i].delta_weight[j]=lr*output_layer[j].grad_var*hidden_layer[i].av+(mt*hidden_layer[i].delta_weight[j])
    
    #using the gradient value of the hidden layer, delta weight of input layer can be easily achieved 
    for i in range(len(input_layer)):
        for j in range(len(hidden_layer[i].delta_weight)):
            input_layer[i].delta_weight[j]=lr*hidden_layer[j].grad_var*input_layer[i].av+(mt*input_layer[i].delta_weight[j])
    
    #updating the input layer weights using the delta weights of the input layer 
    for i in range(len(input_layer)):
        for j in range(len(input_layer[i].weights)):
            input_layer[i].weights[j]=input_layer[i].delta_weight[j]+input_layer[i].weights[j]
            print(input_layer[i].weights[j])
    #updating the hidden layer weights using the delta weights of the hidden layer
    for i in range(len(hidden_layer)):
        for j in range(len(hidden_layer[i].weights)):
            hidden_layer[i].weights[j]=hidden_layer[i].delta_weight[j]+hidden_layer[i].weights[j]
            #print(hidden_layer[i].weights[j])
    


# In[98]:


outputs=[1,1]
backprop(outputs)


# In[87]:


def training_function():
    epoch = 20
    for i in range(epoch):
        with open('d_train.csv') as csv_f:
            read = csv.reader(csv_f,quoting=csv.QUOTE_NONNUMERIC)
            next(read)
            inputs = []
            outputs = []
            for everyrow in read:
                inputs.append(everyrow[1:3])
                outputs.append(everyrow[3:5])
            for i in inputs:
                feedforword_process(i)
            for j in outputs:
                backprop(j)  


# In[88]:


def training_error():
    error = [0,0]
    result1 = 0
    with open('d_train.csv','r') as csv_f:
                read = csv.reader(csv_f,quoting=csv.QUOTE_NONNUMERIC)
                next(read)
                rows = (read)
                cluster= 0
                inputs = []
                outputs = []
                count = 0 
                for i in rows:
                    count = count+1
                    inputs=(i[1:3])
                    outputs = (i[3:5])
                    feedforword_process(inputs)
                    for j in range(len(outputs)):
                        error[j]=((outputs[j]-output_layer[j].av)**2)
                        result1 =(error[0]+error[1])/2
                    cluster=cluster+result1
                cluster = cluster/count
                rootmean = math.sqrt(cluster)
                print(rootmean)


# In[89]:


training_error()


# In[90]:


def validation_error():
    error = [0,0]
    result2 = 0
    with open('d_test.csv','r') as csv_f:
                read = csv.reader(csv_f,quoting=csv.QUOTE_NONNUMERIC)
                next(read)
                rows = (read)
                cluster1 = 0
                inputs = []
                outputs = []
                count = 0 
                for i in rows:
                    count = count+1
                    inputs=(i[1:3])
                    outputs = (i[3:5])
                    feedforword_process(inputs)
                    for j in range(len(outputs)):
                        error[j]=((outputs[j]-output_layer[j].av)**2)
                        cluster2 =(error[0]+error[1])/2
                    cluster1=cluster1+result2
                cluster1 = cluster1/count
                rootmean = math.sqrt(cluster1)
                print(rootmean)


# In[78]:


validation_error()


# In[ ]:




