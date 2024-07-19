"Name       : Roi Halali && Dor Kershberg "
"Titel      : pytorch lerning             "
"Sub Titel  : LSTM model                  "

#%%
# Libraries:
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import glob
import random
# from spider_diagram import *
import pickle
from librosa import display

#%%
#Functions:

def plot_graph  (   arr1,titel1,x_title1,y_title1,
                    arr2,titel2,x_title2,y_title2,
                    arr3,titel3,x_title3,y_title3,
                    arr4,titel4,x_title4,y_title4,
                    path
                ):
    arr1=np.array(arr1.detach())
    arr2=np.array(arr2.detach())
    #arr3=np.array(arr3.detach())
    #arr4=np.array(arr4.detach())

    with plt.style.context('dark_background'):
        plt.figure(figsize=(20, 15))
        plt.suptitle( 'cost function MSE-error, one LSTM layer', fontsize=25)

        plt.subplot(2, 2, 1)
        plt.plot(arr1, color='c', linewidth=.5, label=titel1)
        plt.title(titel1)
        plt.xlabel(x_title1)
        plt.ylabel(y_title1)

        plt.subplot(2, 2, 2)
        plt.plot(arr2, color='c', linewidth=.5, label=titel2)
        plt.title(titel2)
        plt.xlabel(x_title2)
        plt.ylabel(y_title2)

        plt.subplot(2, 2, 3)
        plt.plot(arr3, color='c', linewidth=.5, label=titel1)
        plt.title(titel3)
        plt.xlabel(x_title3)
        plt.ylabel(y_title3)

        plt.subplot(2, 2, 4)
        plt.plot(arr4, color='c', linewidth=.5, label=titel2)
        plt.title(titel4)
        plt.xlabel(x_title4)
        plt.ylabel(y_title4)
    plt.savefig(path)
    plt.show()


def Data_set(data_dir, save_dir,data_len):
    dataBase = []
    temp_data_base = []
    check_phoneme_name = []
    i = 0

    for phoneme_dir in glob.iglob(data_dir + '**', recursive=True):
        i += 1
        if phoneme_dir[-3:] == 'npy' and i % dataBase_div == 0:
            parent_dir = os.path.dirname(phoneme_dir)
            phoneme_name = parent_dir[len(data_dir):]

            if check_phoneme_name != phoneme_name and phoneme_name != "aa" and phoneme_name in vowels_list:
                check_phoneme_name = phoneme_name
                if phoneme_name == "ow":
                    dataBase += temp_data_base[:int(data_len * 1.3)] + temp_data_base[-int(data_len * 1.3):]
                else:
                    dataBase += temp_data_base[:data_len] + temp_data_base[-data_len:]
                temp_data_base = []

            if phoneme_name in vowels_list:
                phoneme_val = eval(phoneme_name)
                data = (np.load(phoneme_dir, allow_pickle=True), phoneme_val)
                temp_data_base.append(data)

            # if data[0].size != 320 :
            #     print(phoneme_dir)
            #     os.remove(phoneme_dir)


    with open(save_dir, 'wb') as f:
        pickle.dump(dataBase, f)

def get_batches(database, batch_size):
    x=DataLoader(dataset=database, batch_size=batch_size, shuffle=True, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, prefetch_factor=2,
               persistent_workers=False)

    return x

def array_reduse(arr,row, col):
    #get 2D array in convert to array in shape (row,col)
    new_arr=[]

    arr_cols=arr.shape[1]
    arr_rows=arr.shape[0]

    #from each colom choose only 2:
    n = random.randint(0, arr_cols-1)

    for i in range (arr_rows):
        for j in range(row):
            pass

def decision(result,decision_values,target,batch_size):

    #decision_vector:
    decision=[]
    for b in range(batch_size):
        decision_vector = []
        for i in range (len(result[0])) :
            if result[b,i]>decision_values[i] :
                decision_vector.append(1)
            else :
                decision_vector.append(0)
        decision.append(torch.Tensor(decision_vector))
    decision=torch.stack(decision)

    #apdate decision_values :
    for b in range(batch_size):
        for i in range (len(result[0])) :
            if target[b,i]!=decision[b,i] :
                decision_values[i]=0.5   #(result[b,i]-decision_values[i])/2
    acuur,feature_accur=accurecy(target, decision, batch_size)
    return   acuur,feature_accur,decision_values,decision

def accurecy(target,result,batch_size):
    # get tow "hot vectors" and compute the accurecy
    accur=0
    feature_accur=np.zeros(len(target[0]))
    feature_accur_div=np.array([0,0,0,0,0])
    for b in range(batch_size):
        for i in range(len(target[0])):
            if result[b,i] == target[b,i]:
                accur+=1
                feature_accur[i]+=1
    acuur=accur/(len(target[0])*batch_size)
    feature_accur=feature_accur/batch_size
    return acuur, feature_accur

def Targ(target,batch_size):
    target_CEL=np.zeros(batch_size)
    for b in range(batch_size):
        for i in range(len(target[0])):
            if target[b, i] == 1:
                target_CEL[b]=i
                return torch.from_numpy(target_CEL).type(torch.LongTensor)


def train_func(net, train_dataBase, eval_dataBase, epochs, n_hidden, n_steps, lr, batch_size, eval_every, decision_value,
               print_train_data,print_eval_data,num_classes
               ):

    train_h = net.init_hidden()             #initialize hc
    decision_values=[decision_value,decision_value,decision_value,decision_value]

    # loss functions:
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()
    # criterion= torch.nn.CosineEmbeddingLoss()
    # criterion=torch.nn.NLLLoss()


    counter = 0
    max_accuracy = 0
    loss_tot = []
    avrage_loss=0
    accurecy_train_tot=[]
    avg_accuracy_train=0
    avg_feacher_accuracy_train=np.zeros(num_classes)

    accuracy_eval_tot = []
    eval_loss_tot = []
    for e in range(epochs):
        for data in get_batches(train_dataBase, batch_size):
            net.train()

            #optimizers:
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            #opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
            #opt = torch.optim.Adamax(net.parameters(), lr=lr)

            # Input,Target
            inputs, targets = data[0], data[1]
            if inputs.shape[0]!=batch_size :
                print("error "+str(counter))
                continue
            inputs = inputs.view(batch_size, n_steps, input_dim)
            inputs= inputs.type(torch.DoubleTensor)
            counter += 1

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            train_h = train_h.data #(each.data for each in train_h)

            net.zero_grad()

            output, train_h = net.forward(inputs, train_h)

            loss = criterion(output, targets.double())
            avrage_loss+=loss

            acurecy_train,fetcher_acurecy_train,decision_values, decision1 = decision(output,decision_values,targets,batch_size)

            avg_accuracy_train+=acurecy_train
            avg_feacher_accuracy_train+=fetcher_acurecy_train
            loss.backward()

            opt.step()

            if counter % print_train_data == 0:
                avrage_loss=avrage_loss/print_train_data
                loss_tot.append(avrage_loss)

                avg_accuracy_train=avg_accuracy_train/print_train_data
                avg_feacher_accuracy_train=avg_feacher_accuracy_train/print_train_data
                accurecy_train_tot.append(acurecy_train)

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Train Loss: {:.6f}".format(avrage_loss),
                      "Avrage Accuracy: {:.4f}".format(avg_accuracy_train),
                      "Avrage Feature Accuracy:" +str( avg_feacher_accuracy_train)
                      )
                avg_accuracy_train=0
                avg_feacher_accuracy_train=np.zeros(num_classes)
                avrage_loss=0


            if counter % eval_every == 0:
                net.eval()
                with torch.no_grad():
                    avg_eval_loss=0
                    avg_accuracy_eval = 0
                    avg_feacher_accuracy_eval = np.zeros(num_classes)
                    len_eval_dataBase = 0
                    eval_h = net.init_hidden()

                    for data in get_batches(eval_dataBase, batch_size):
                        len_eval_dataBase += 1
                        # evaluation Input,Target
                        inputs, targets = data[0], data[1]
                        if inputs.shape[0] != batch_size:
                            continue
                        inputs = inputs.view(batch_size, n_steps, input_dim)
                        inputs = inputs.type(torch.DoubleTensor)

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        eval_h = eval_h.data #tuple([each.data for each in eval_h])

                        output, eval_h = net.forward(inputs, eval_h)

                        eval_loss = criterion(output, targets)
                        avg_eval_loss+=eval_loss

                        acurecy_eval,fetcher_acurecy_eval, decision_values,decision2 = decision(output, decision_values, targets, batch_size)

                        avg_accuracy_eval += acurecy_eval
                        avg_feacher_accuracy_eval+=fetcher_acurecy_eval

                        if len_eval_dataBase % print_eval_data == 0:
                            eval_loss_tot.append( avg_eval_loss /print_eval_data)
                            print("Epoch: {}/{}...".format(e + 1, epochs),
                                  "Step: {}...".format(len_eval_dataBase),
                                  "Eval Loss: {:.4f}".format(eval_loss),
                                  "Accuracy: {:.4f}".format(acurecy_eval),
                                  )
                            avg_eval_loss=0

                    avg_accuracy_eval = avg_accuracy_eval / len_eval_dataBase
                    avg_feacher_accuracy_eval=avg_feacher_accuracy_eval / len_eval_dataBase
                    accuracy_eval_tot.append(avg_accuracy_eval)
                    print("Average Accuracy: {:.4f}".format(avg_accuracy_eval))
                    print("Average Feature Accuracy:"+str(np.array(avg_feacher_accuracy_eval,dtype="float16")))
                    print("decision values:"+ str(np.array(decision_values,dtype="float16")))
                    print(lr)
                    avg_feacher_accuracy_eval=np.zeros(5)
                    if avg_accuracy_eval > max_accuracy:
                        max_accuracy = avg_accuracy_eval
                        torch.save(net.state_dict(), model_dir)
                        torch.save(train_h,hc_dir)
                        lr -= 0.00005
                        lr_count=0
                    else:
                        lr -= 0.00005
                        lr_count+=1
                        if lr_count == 5 :
                            lr+=0.001

    plot_graph(
                torch.stack(loss_tot), "Training Loss","avrage loss for evry " +str(print_train_data)+" steps","Loss",
                torch.stack(eval_loss_tot), "Evaluation Loss", "avrage loss for evry "+str(print_eval_data)+" steps","Loss",
                np.array(accurecy_train_tot), "Training accurecy ", "avrage accurecy for evry " + str(print_train_data) + " steps", "Accurecy",
                np.array(accuracy_eval_tot), "Evaluation accurecy ","avrege accurecy for evry evaluation loop", "Accurecy",
                "train plot.png"
               )

def test_func(net,test_dataBase,input_dim, n_steps,batch_size, eval_every,decision_value,test_h,epochs,
              print_test_data):

    # Loss functions:
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    test_h =test_h.data  #tuple([each.data for each in test_h])

    counter = 0
    avrage_loss=0
    loss_tot=[]
    avg_accuracy_test=0
    avg_fetcher_acurecy_test=np.zeros(5)
    accurecy_test_tot=[]
    decision_values=[decision_value,decision_value,decision_value,decision_value,decision_value]

    with torch.no_grad():
        for e in range(epochs):
            for data in get_batches(test_dataBase,batch_size):

                # Input,Target
                inputs, targets = data[0], data[1]
                if inputs.shape[0] != batch_size:
                    print("error " + str(counter))
                    continue
                inputs = inputs.view(batch_size, n_steps, input_dim)
                counter += 1

                net.train()
                net.zero_grad()

                output, _ = net.forward(inputs, test_h)

                loss = criterion(output, targets.double())
                avrage_loss += loss

                acurecy_test, fetcher_acurecy_test, decision_values, decision1 = decision(output, decision_values, targets,batch_size)
                avg_accuracy_test += acurecy_test
                avg_fetcher_acurecy_test+=fetcher_acurecy_test

                if counter % print_test_data == 0:
                    avrage_loss=avrage_loss/print_test_data
                    loss_tot.append(avrage_loss)

                    avg_accuracy_test=avg_accuracy_test/print_test_data
                    accurecy_test_tot.append(acurecy_test)

                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Test Loss: {:.6f}".format(avrage_loss),
                          "Avrage Accuracy: {:.4f}".format(avg_accuracy_test))
                    print("Average Feature Accuracy Test:" + str(np.array(avg_fetcher_acurecy_test, dtype="float16")))
                    avrage_loss=0
                    avg_accuracy_test=0
                    avg_fetcher_acurecy_test=np.zeros(5)


    # plot_graph(
    #     torch.stack(loss_tot), "Test Loss", "avrage loss for evry " + str(print_test_data) + " steps", "Loss",
    #     torch.zeros(100), " ", " ", " ",
    #     np.array(accurecy_test_tot), "Test accurecy ","avrage accurecy for evry " + str(print_train_data) + " steps","Accurecy",
    #     torch.zeros(100), " ", " ", " ",
    #     "test plot.png"
    # )

def predict(array, n_steps, phoneme):

    #Load the best model
    loaded_model = RNN().double()
    loaded_model.load_state_dict(torch.load(model_dir))
    loaded_model.eval()

    #Load the best weights
    test_h=torch.load(hc_dir)

    speaker_features=[]
    for data in get_batches(array, batch_size):

        if data.shape[0] != batch_size:
            print("error")
            continue
        with torch.no_grad():
            data = data.view(batch_size, n_steps, input_dim)
    
            #plot data:
            sr=16000
            frame_size = int(np.floor(16 * (10 ** -3) / (1 / sr)))
            hoplen = int(frame_size / 2)
    
            # plt.title("data from pred")
            # plt.title('Spectogram')
            # display.specshow(data[0].numpy().T, sr=sr, x_axis='time', y_axis='linear', hop_length=hoplen, win_length=frame_size,fmax=int(5500))
            # plt.colorbar(format="%+2.f")
            # plt.show()
    
            prediction, _ = loaded_model(data, test_h)
            
            decision_values = [decision_value, decision_value, decision_value, decision_value]
            target = []
            for i in range(batch_size):
                target.append(np.array(phoneme, dtype=int))
            target=np.array(target)
            acuur,feature_accur,decision_values,decisions=decision(prediction,decision_values,target,batch_size)
            # print("Predictions Accuracy: {:.4f}".format(acuur))
            # print("Features Accuracy:" + str(np.array(feature_accur, dtype="float16")))
            for b in range(batch_size):
                speaker_features.append(np.array(decisions[b,:]))
                
    return acuur

    # Nj=np.array(len(speaker_features))
    # speaker_features=np.array(speaker_features)
    # ejp=[]

    # num_features=np.zeros((13,1))
    # for i in range (len(num_features)):
    #     num_features[i]=sum(speaker_features[:,i])

    # for i in range(len(num_features)):
    #       ejp.append(num_features[i] / Nj)

    # ejp = np.array(ejp)
    # spyder_diagram(ejp)

# def accuracy_predict(array,features):


#%%
# Dirs & Mode:

#data dirs:
train_data_dir = 'data/train/'
test_data_dir  = 'data/test/'
val_data_dir   = 'data/evaluation/'
pred_data_dir  = 'data/prediction/'


model_dir="model.pth"
hc_dir="hc"

#%%
# Data set

dataBase_div=1 # divide the data base by dataBase_div
#Hebrew Phonemes
b   = np.array([0,1,0,0,0,1,0,0,0,1,0,0,0],dtype=int)
d   = np.array([0,1,1,0,0,1,1,0,0,1,0,0,0],dtype=int)
f   = np.array([0,1,0,0,0,1,0,0,0,0,1,0,1],dtype=int)
g   = np.array([0,1,0,1,0,0,0,0,0,1,1,0,0],dtype=int)
k   = np.array([0,1,1,1,0,0,0,0,0,0,0,0,0],dtype=int)
# kcl = np.array([0,1,1,1,0,0,0,0,0,0,0,0,0],dtype=int)
l   = np.array([0,1,1,0,0,1,1,0,0,1,1,0,0],dtype=int)
m   = np.array([0,1,0,0,0,1,0,0,0,1,0,1,0],dtype=int)
n   = np.array([0,1,0,0,0,1,0,0,0,1,0,1,0],dtype=int)
p   = np.array([0,1,1,0,0,1,0,0,0,0,0,0,0],dtype=int)
q   = np.array([0,1,0,1,1,0,0,0,0,0,0,0,0],dtype=int)
r   = np.array([1,1,0,0,0,0,1,0,0,1,1,0,0],dtype=int)
s   = np.array([0,1,0,0,0,1,1,0,0,0,1,0,1],dtype=int)
sh  = np.array([0,1,1,0,0,0,1,0,0,0,1,0,1],dtype=int)
t   = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
#tcl = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
#dcl = np.array([0,1,0,0,0,1,1,0,0,0,0,0,0],dtype=int)
th  = np.array([0,1,0,0,0,1,1,0,0,1,1,0,0],dtype=int)
v   = np.array([0,1,0,0,0,1,0,0,0,1,1,0,1],dtype=int)
z   = np.array([0,1,0,0,0,1,1,0,0,1,1,0,1],dtype=int)# 'חסר כ' סופית,ח' וע

#special
ch  = np.array([0,1,0,1,0,0,1,0,0,0,0,0,1],dtype=int)
zh  = np.array([0,1,1,0,0,0,1,0,0,1,1,0,1],dtype=int)

#vowels
# aa  = np.array([1,0,0,1,1,0,0,0,1,1,1,0,0],dtype=int)#'אאאא
# ae  = np.array([1,0,0,0,1,0,0,0,1,1,1,0,0],dtype=int)#'אה
# eh  = np.array([0,0,0,0,0,0,0,0,0,1,1,0,0],dtype=int)#'אה
# iy  = np.array([0,0,1,0,0,0,0,0,1,1,1,0,0],dtype=int)#'אי
# uw  = np.array([1,0,1,1,0,0,0,1,1,1,1,0,0],dtype=int)#'אוו
# ow  = np.array([1,0,0,1,0,0,0,1,1,1,1,0,0],dtype=int)#'או

aa  = np.array([1,0,1,0],dtype=int)#'אאאא
eh  = np.array([0,0,0,0],dtype=int)#'אה
iy  = np.array([0,1,0,0],dtype=int)#'אי
uw  = np.array([1,1,1,1],dtype=int)#'אוו
ow  = np.array([1,0,1,1],dtype=int)#'או

non =np.array([0,0,0,0,0],dtype=int)

vowels_list=["aa","eh","iy","ow","uw"]
#
# #train data Base:
# # Data_set(train_data_dir,'data/train_dataBase.pickle',5000)
# with open('data/train_dataBase.pickle', 'rb') as f:
#        train_dataBase = pickle.load(f)
# train_dataBase=train_dataBase[::dataBase_div]
#
# #evaluation data Base:
# # Data_set(val_data_dir,'data/eval_dataBase.pickle',400)
# with open('data/eval_dataBase.pickle', 'rb') as f:
#       eval_dataBase = pickle.load(f)
# eval_dataBase=eval_dataBase[::dataBase_div]
#
# #test data Base:
# # Data_set(test_data_dir,'data/test_dataBase.pickle',1000)
# with open('data/test_dataBase.pickle', 'rb') as f:
#     test_dataBase = pickle.load(f)
# test_dataBase=test_dataBase[::dataBase_div]

#%%
#Code Parameters

decision_value=0.5         # if the returned value from the model is higher then decision_value the value is 1

#Model Hyper parametesr:
batch_size      = 1
num_epochs      = 5
learning_rate   = 0.001
num_classes     = 4         #number of classes for detection
#eval_every      =np.floor((len(train_dataBase)/batch_size)*0.1).astype(np.int64)   #evaluate the data evrey eval_every steps

#Model Layers:
n_layers        = 1          #Number of recurrent
bd              = False      #bidirectional RNN
batch_first     = True
biasFC          = True       #add a bias to FC layers

# Enter RNN layer:
# 3 long RNN layer - get STFT
input_dim       = 80        #RNN input dim for each cell
time_steps      = 4         #time steps
n_hidden        = 80


# FC output layers :
#passing all the RNN layer cells outputs to FC. output shape = num classes
# shape - (batch_size,80,4)-> (80*4*batch_size,1)
in_noudes1=80*4
if bd:
    in_noudes1=80*2*4
out_noudes1=50


in_noudes2=50
out_noudes2=num_classes

# FC output layers for hc:
#the output hc needs to be in size (1,batch_size,5) for entering the next forword step
in_noudes2h=5
out_noudes2h=10

# #Printing:
# print_train_data=np.floor((len(train_dataBase)/batch_size)*0.05).astype(np.int64)
# print_test_data=np.floor((len(test_dataBase)/batch_size)*0.05).astype(np.int64)
# print_eval_data=np.floor((len(eval_dataBase)/batch_size)*0.2).astype(np.int64)

#%%
# Model Architecture:
class RNN(nn.Module):

    def __init__(self,

                 n_layers=n_layers,batch_first=batch_first,biasFC=biasFC,batch_size=batch_size,bd=bd, #general parameters

                 input_dim=input_dim ,n_hidden=n_hidden,                        # enter RNN layer

                 in_noudes1 = in_noudes1, out_noudes1= out_noudes1,             # FC layers

                 in_noudes2=in_noudes2, out_noudes2=out_noudes2,                # FC output layers

                 in_noudes2h=in_noudes2h,out_noudes2h=out_noudes2h              # FC output layers for hc

                 ):

        super().__init__()
        self.batch_size=batch_size
        self.n_layers=n_layers
        self.input_dim=input_dim
        self.n_hidden=n_hidden
        self.bd=bd

        #rnn layer
        self.rnn=nn.RNN( input_dim ,n_hidden, batch_first=batch_first,bidirectional=bd )

        # fully-connected output layer
        self.FC_1 = nn.Linear(in_features=in_noudes1, out_features=out_noudes1, bias=biasFC)
        self.FC_out = nn.Linear(in_features=in_noudes2, out_features=out_noudes2, bias=biasFC)
        self.FC_outh = nn.Linear(in_features=in_noudes2h, out_features=out_noudes2h, bias=biasFC)

        #activation layer
        self.softmax=torch.nn.Softmax()
        self.sigmoid=nn.Sigmoid()
        self.relu=torch.nn.ReLU()
        self.dropout=nn.Dropout(p=0.7)
        # Initialize the weights
        self.init_weights()

    def forward(self, x,hc):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hc`. '''

        x1, h1 = self.rnn(x, hc)  # 4 time steps
        x1 = x1.contiguous().view(self.batch_size,80*4,1)
        x2=self.FC(x1)
        output=self.FC_output(x2)

        # Return output and the hidden state (h, c)
        return output,  h1

    def init_weights(self):
        ''' Initialize weights for fully connected layer '''

        # self.FC_12h.bias.data.fill_(0)
        # self.FC_12h.weight.data.uniform_(-1, 1)

        self.FC_out.bias.data.fill_(0)
        self.FC_out.weight.data.uniform_(-1, 1)

        self.FC_outh.bias.data.fill_(0)
        self.FC_outh.weight.data.uniform_(-1, 1)

    def init_hidden(self) :
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x input_dim x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if self.bd:
            mult=2
        else:
            mult=1
        return weight.new(mult*self.n_layers,self.batch_size, self.n_hidden).zero_().double()#weight.new(mult*self.n_layers,self.batch_size, self.n_hidden).zero_().double()

    def FC(self,x):
        output=[]
        for b in range(batch_size):
            temp = self.FC_1(self.dropout( x[b,:,:].reshape(1, in_noudes1)) )

            # outputs:
            # output.append(self.relu(temp))
            # output.append(self.softmax(temp))
            output.append(self.sigmoid(temp))
        output=torch.stack(output).view(batch_size, -1)
        return output

    def FC_output(self,x):
        output=[]
        for b in range(batch_size):
            temp = self.FC_out(x[b,:])

            # outputs:
            #output.append(self.relu(temp))
            #output.append(self.softmax(temp))
            output.append(self.sigmoid(temp))
        output=torch.stack(output).view(batch_size, -1)
        return output


#%%
# Initialize the network
net = RNN(
             n_layers=n_layers,batch_first=batch_first,biasFC=biasFC,batch_size=batch_size,bd=bd, #general parameters

             input_dim=input_dim ,n_hidden=n_hidden,                        # enter LSTM layer

             in_noudes1 = in_noudes1, out_noudes1= out_noudes1,             # FC output layers

             in_noudes2=in_noudes2, out_noudes2=out_noudes2,                # FC output layers

             in_noudes2h=in_noudes2h,out_noudes2h=out_noudes2h              # FC output layers for hc
          )
net = net.double()

# # Train
# train_func(net, train_dataBase,eval_dataBase, epochs=num_epochs, n_hidden=n_hidden, n_steps=time_steps, lr=learning_rate,batch_size=batch_size,eval_every=eval_every,decision_value=decision_value,
#                   print_train_data=print_train_data,print_eval_data=print_eval_data,num_classes=num_classes)

# Test
# loaded_model = RNN().double()
# loaded_model.load_state_dict(torch.load(model_dir))
# loaded_model.eval()
#
# test_h=torch.load(hc_dir)
# test_func(loaded_model, test_dataBase, input_dim=input_dim, n_steps=time_steps,batch_size=batch_size,eval_every=10,decision_value=decision_value,test_h=test_h,epochs=5,
#             print_test_data=print_test_data)

# test=[]
# test_features=[]
# for i in range(len(test_dataBase)):
#     test.append(test_dataBase[i][0])
#     test_features.append(test_dataBase[i][1])
# test=test[0:len(test)-(len(test)-80)]

# train=[]
# for i in range(len(train_dataBase)):
#     train.append(train_dataBase[i][0])
# train=train[0:len(train)-(len(train)%10)]

def new_rec_pred(pred_data_dir, phoneme):
    
    Roi_rec=[]
    
    #Prediction on new recordings
    for phoneme_file in glob.iglob(pred_data_dir + '**', recursive=True):
        if phoneme_file[-3:] == 'npy':          #check npy file's form lest three letters
            pred_data = np.load(phoneme_file, allow_pickle=True)
            Roi_rec.append(pred_data)
    Roi_rec = Roi_rec[0:len(Roi_rec)-(len(Roi_rec) % 10)]
    
    pred = predict(Roi_rec, n_steps = time_steps, phoneme = phoneme)
    
    return pred















