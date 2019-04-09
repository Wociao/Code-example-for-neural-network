import numpy as np
from urllib import request
import gzip
import math
import pickle


def grad_softmax_crossentropy(X, y):
    m = y.shape[0]
    ones_for_answers = np.zeros_like(X)
    ones_for_answers[np.arange(len(X)), y] = 1

    p = np.exp(X) / np.exp(X).sum(axis=-1, keepdims=True)
    return (- ones_for_answers + p) / m

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)

        training_images, training_labels, testing_images, testing_labels = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
        # Normalize the images
        training_images.astype('float32')
        testing_images.astype('float32')
        training_images = training_images / 255
        testing_images = testing_images / 255
        return training_images, training_labels, testing_images, testing_labels


TRimg,TRlab,TSimg,TSlab=load()
print(len(TRimg),len(TRlab),len(TSimg),len(TSlab))
print(len(TRimg[0]),len(TRlab),len(TSimg[0]),len(TSlab))

#hyper parameters
TotalTrainingData=len(TRimg)
BatchSize=128   #batchsize
epochs=10   #epochs times
IterationInOneEpoches=math.ceil(TotalTrainingData/BatchSize)
print(IterationInOneEpoches,BatchSize)

#network parameters
HidSize=[200,50]
ClassNO=10

#input layer
INPLayer=TRimg[0].copy()
print(len(INPLayer))
# W, b between inputlayer and hidden layer 1
WA=0.01*np.random.randn(len(INPLayer),HidSize[0])
BA=np.zeros((1,HidSize[0]))
WB=0.01*np.random.randn(HidSize[0],HidSize[1])
BB=np.zeros((1,HidSize[1]))
WC=0.01*np.random.randn(HidSize[1],ClassNO)
BC=np.zeros((1,ClassNO))

LearningRate=0.01
reg=1e-3
BatchSize=128
INPLayer_B_Label=TRlab[0:BatchSize].copy()
INPLayer_B=TRimg[0:BatchSize].copy()
BatchSize=len(INPLayer_B)
for i in range(0,epochs):

    #input image & label
    for j in range(0,IterationInOneEpoches-1):

        #input image & label for one batch
        INPLayer_B_Label=TRlab[BatchSize*j:BatchSize*(j+1)].copy()
        INPLayer_B=TRimg[BatchSize*j:BatchSize*(j+1)].copy()
        #FW:
        #print(len(INPLayer_B))
        #input -> hidden 1 relu
        HA=np.maximum(0,np.dot(INPLayer_B,WA)+BA)
        #print(HA.shape)
        # hidden 1 -> hidden 2 relu
        HB=np.maximum(0,np.dot(HA,WB)+BB)
        #print(HB.shape)
        # hidden 2 -> output
        OutPut=np.dot(HB,WC)+BC
        #print(OutPut.shape)
        #softmax
        Result=np.exp(OutPut)
        probs=Result/np.sum(Result,axis=1,keepdims=True)
        #print(probs)
        #print(np.sum(probs))
        #print(TRlab.shape)
        #compute the loss
        #print(len(INPLayer_B_Label))
        #print(INPLayer_B_Label)

        #predicted_class = np.argmax(OutPut, axis=1)
        #print('Training accuracy: ', np.mean(predicted_class == INPLayer_B_Label))

        #calculate loss
        Corect_logprobs=-np.log(probs[range(BatchSize),INPLayer_B_Label])
        Data_loss=np.sum(Corect_logprobs)/BatchSize
        reg_loss=0.5*reg*np.sum(WA*WA)+0.5*reg*np.sum(WB*WB)+0.5*reg*np.sum(WC*WC)
        LOSS=Data_loss+reg_loss
        if j==1:
            print("loss is : ",LOSS,"Dataloss",Data_loss)
        #
        #BW:

        Dscores=probs.copy()
        Dscores[range(BatchSize),INPLayer_B_Label]-=1
        Dscores/=BatchSize

        Dscores=grad_softmax_crossentropy(probs,INPLayer_B_Label)


        DWC=np.dot(HB.T,Dscores)
        DBC=np.sum(Dscores,axis=0,keepdims=True)
        DhiddenB=np.dot(Dscores,WC.T)
        DhiddenB[HB <=0]=0

        DWB=np.dot(HA.T,DhiddenB)
        DBB=np.sum(DhiddenB,axis=0,keepdims=True)
        DhiddenA=np.dot(HB,WB.T)
        DhiddenA[HA <=0]=0

        DWA=np.dot(INPLayer_B.T,DhiddenA)
        DBA=np.sum(DhiddenA,axis=0,keepdims=True)


        DWA += reg*WA
        DWB += reg * WB
        DWC += reg * WC

        WA += -LearningRate*DWA
        BA += -LearningRate*DBA
        WB += -LearningRate*DWB
        BB += -LearningRate*DBB
        WC += -LearningRate*DWC
        BC += -LearningRate*DBC
    print(i)

# evaluate training set accuracy
INPLayer_C=TSimg.copy()
INPLayer_C_Label=TSlab.copy()
HA=np.maximum(0,np.dot(INPLayer_C,WA)+BA)
#print(HA.shape)
HB=np.maximum(0,np.dot(HA,WB)+BB)
#print(HB.shape)
OutPut=np.dot(HB,WC)+BC
#print(OutPut.shape)
Result=np.exp(OutPut)
probs=Result/np.sum(Result,axis=1,keepdims=True)


predicted_class = np.argmax(OutPut, axis=1)
print( 'Testing accuracy: ',np.mean(predicted_class == INPLayer_C_Label))

#for hidden layer