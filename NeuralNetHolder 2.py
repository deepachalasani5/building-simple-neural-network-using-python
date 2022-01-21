import math
import random
import re
class neuron:
    def __init__(self,no_of_weights):
        self.av = 0.0
        self.weights = []
        self.no_of_weights = no_of_weights
        x = 0
        while x < no_of_weights:
            self.weights.append(random.uniform(-1,1))
            x+=1
            
    def act_fn_hidden(self,lamda,input): #create activation function for the neuron(sigmoid fn)
        self.av = 1/(1+ math.exp(-lamda*input))
        #print('activated value ='+str(self.av))
        return self.av
    def act_fn_in(self,input): #create activation function for the neuron(sigmoid fn)
        self.av = input
        #print('activated value ='+str(self.av))
        return self.av
        
    def vectors(self,input): #multiply input with the weight
        vector = self.av * self.weights[input]
        #print(vector)
        return vector
    
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.input_node= []
        self.hidden_node=[]
        self.output_node=[]
        #create list for each layer
        self.input1 = neuron(2)
        self.input2 = neuron(2)
        self.hidden1=neuron(2)
        self.hidden2=neuron(2)
        self.hidden_bias=neuron(2)
        self.output1=neuron(2)
        self.output2=neuron(2)
        self.input_node.append(self.input1)
        self.input_node.append(self.input2)
        self.hidden_node.append(self.hidden1)
        self.hidden_node.append(self.hidden2)
        self.hidden_node.append(self.hidden_bias)
        self.output_node.append(self.output1)
        self.output_node.append(self.output2)
        #put weights from offline training here
        self.input1.weights.append(6.421051926642344)
        self.input1.weights.append(-32.31223334950358)
        self.input2.weights.append(-0.040069)
        self.input2.weights.append(16.044)
        self.hidden1.weights.append(-0.94944)
        self.hidden1.weights.append(24.68)
        self.hidden2.weights.append(-2.77199)
        self.hidden2.weights.append(-2.277336)
        self.hidden_bias.weights.append(-0.3045289)
        self.hidden_bias.weights.append(-22.2820864)

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        print(input_row)
        a=input_row[0]
        a= float(a.replace('\U00002013', '-'))
        b=input_row[1]
        b=float(b.replace('\U00002013', '-'))
        input_norm=[0,0]
        #normalize inputs
        input_norm[0]= (a+363.209)/(569.9919+363.209)
        input_norm[1]= (b-63.34053)/(755.7372-65.34053)
        #start a prediction (Feed forward from offline training)
        lamda=0.8
        i=0
        v1=0.0
        v2=0.0
        while i< 2:
                self.input_node[i].act_fn_in(input_norm[i])
                v1+= self.input_node[i].vectors(0)
                v2+= self.input_node[i].vectors(1)
                i+=1
        i=0
        input_vector=[v1,v2]
        v3=0.0 #vector to output node 1
        v4=0.0 #vector to output node 2
        while i< 2: #calculate vector 3 and 4
            self. hidden_node[i].act_fn_hidden(lamda,input_vector[i])
            v3+=self.hidden_node[i].vectors(0)
            v4+=self.hidden_node[i].vectors(1)
            i+=1
        i=0
        self.hidden_node[2].act_fn_in(1.0) #set activation value for bias hidden node
        v3 += self.hidden_node[2].vectors(0) #combined to vectors which will go to output node
        v4 +=self.hidden_node[2].vectors(1)
        #go through output node
        self.output_node[0].act_fn_hidden(lamda,v3)
        self.output_node[1].act_fn_hidden(lamda,v4)
        vel_x= self.output_node[0].av*(7.6+5.77882)-5.77882
        vel_y= self.output_node[1].av*(7.114944+5.78864)-5.78864 
        return (vel_x, vel_y)
        
                                

            

