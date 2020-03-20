from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from keras.callbacks import EarlyStopping
from keras import layers, models
from keras import optimizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier

#searching the optimal hyper-parameter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import ast, csv

#from jetimage import jet_images
#jetimages=jet_images()

images=np.load('jetimagesnnsmol.npy')
labels=np.load('jetlabelsnnsmol.npy')

datajets=[]
with open('JetData.csv', mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	line_count = 0
	for row in csv_reader:
		#if line_count>=1:
		datajets.append(ast.literal_eval(ast.literal_eval(row[',datajets'])[1]))
		line_count+=1

ids=[d[0] for d in datajets] # if d[7]>0]
data=[d[1:-1] for d in datajets] #s if d[7]>0])

weights=[d[-1] for d in datajets] #s if d[7]>0] #remove negative weights?
data=[d[:len(d)-1] for d in data]
for i in range(len(data)):
	data[i].append(weights[i])
data=np.array(data)

print(data[0])

plt.figure()
#plt.imshow(np.array([k[5:15] for k in images[3][5:15]]))
plt.imshow(images[3])
#plt.show()

images=images/100
#images=np.array([[[[j] for j in k] for k in h] for h in images])
#labels=np.array([je[0] for je in jetimages])

trainToTestRatio=0.8
validFrac=0.1

nps=np.random.RandomState(12302)  #Set random seed for reproducibility
nClasses = 2   #binary output

theShape = images.shape[0]
classShuffle = nps.permutation(theShape)
classTrainLimit = int(theShape*trainToTestRatio)	  
classValidLimit = int(theShape*(trainToTestRatio+validFrac))
	  
#setup the various datasets for multiclass training as NUMPY arrays
classI        = images
classY        = labels
classL=data

#shuffle datasets
classI = [classI[i] for i in classShuffle]
classY = [classY[i] for i in classShuffle]
classL = [classL[i] for i in classShuffle]

#split datasets
classTrainY, classValidY, classTestY = np.split( classY,  [classTrainLimit, classValidLimit] )
classTrainI, classValidI, classTestI     = np.split( classI,  [classTrainLimit, classValidLimit] )
classTrainL, classValidL, classTestL     = np.split( classL,  [classTrainLimit, classValidLimit] )

class_names = ['gluon', 'quark']

"""print(classTrainI[0], classTrainY[0])
plt.figure()
plt.imshow(classTrainI[0])
plt.colorbar()
plt.grid(False)
plt.show()"""



plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(classTrainI[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[classTrainY[i]])
plt.savefig('jetimageset.png')


from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler


"""
#save the scalar for use later
scalerFile = "XTrainScalar.save"
joblib.dump(scaler, scalerFile)"""

# convert integers to dummy variables (keras needs one hot encoding)
y_train_onehot = classTrainY #np.utils.to_categorical(classTrainY, num_classes=2)
y_test_onehot  = classTestY #np_utils.to_categorical(classTestY, num_classes=2)
y_valid_onehot = classValidY #np_utils.to_categorical(classValidY, num_classes=2)

#from keras.layers.core import Input
from keras.models import Model


effhy=[[],[]]
ap=[0.1] #np.arange(0.05, 0.275, 0.025) #range(1,100, 5) #
for kl in ap:
	k=0.13
	# this is your image input definition. You have to specify a shape. 
	image_input = layers.Input(shape=(20,20,3))
	# Some more data input with 10 features (eg.)
	other_data_input = layers.Input(shape=(7,))    
	
	conv1 = layers.Conv2D(32, (1, 1))(image_input)
	conv1=layers.LeakyReLU(alpha=k)(conv1)

	conv2 = layers.Conv2D(32, (1, 1))(conv1)
	conv2=layers.LeakyReLU(alpha=k)(conv2)

	#conv3 = layers.Conv2D(32, (1, 1))(conv2)
	#conv3=layers.LeakyReLU(alpha=k)(conv3)

	first_part_output = layers.Flatten()(conv2)
	
	# Merge the output of the convNet with added features by concatenation
	merged_model = layers.concatenate([first_part_output, other_data_input])
	
	dnn=layers.Dense(64)(merged_model)
	dnn=layers.LeakyReLU(alpha=k)(dnn)

	# Predict on the output (say you want a binary classification)
	predictions = layers.Dense(1, activation ='sigmoid')(dnn)

	# Now create the model
	model = Model(inputs=[image_input, other_data_input], outputs=predictions)

	# see your model 
	model.summary()
	
	callbacks = []
	callbacks.append(EarlyStopping(monitor='val_binary_crossentropy', patience=50))

	optimizer=optimizers.RMSprop(lr=kl) #learning_rate,decay=decay)
	optimizer='Nadam'
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

	inpus=[classTrainI, classTrainL]
	valids=[classValidI, classValidL]

	history = model.fit(inpus, y_train_onehot, epochs=4, shuffle=True, callbacks=callbacks, validation_data=(valids, y_valid_onehot))
	test_loss, test_acc = model.evaluate([classTestI, classTestL],  classTestY, verbose=2)
	print(test_loss, test_acc)

	predProbClass  = model.predict([classTestI, classTestL])
	probsquark=predProbClass[:,0] 
	probsquark=probsquark/float(max(probsquark))
	probsquark=[(l-0.5)/0.5 for l in probsquark]
	effhy[0].append(roc_auc_score(classTestY, probsquark))

	predProbClass  = model.predict([classTrainI, classTrainL])
	probsquark=predProbClass[:,0] 
	probsquark=probsquark/float(max(probsquark))
	probsquark=[(l-0.5)/0.5 for l in probsquark]
	effhy[1].append(roc_auc_score(classTrainY, probsquark))


"""
plt.figure()
plt.plot(ap, effhy[0], label='testing efficiency')
plt.plot(ap, effhy[1], label='training efficiency')
plt.legend()
plt.ylabel('Testing efficiency')
plt.xlabel('Learning rate')
plt.savefig('nnlr.png')


plt.figure()
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')"""





def deep_learning_create(conv_layers, dense_layers, dunits, cunits, alpha, tuplec, learning_rate, decay):
    model = models.Sequential()

    #bulid the layers
    for i in range(1,conv_layers+1):
        if i==1:
           model.add(layers.Conv2D(cunits, tuplec, input_shape=(20, 20, 3)))
	   model.add(layers.LeakyReLU(alpha=alpha))
	   #model.add(layers.MaxPooling2D((2, 2)))
        else:
           model.add(layers.Conv2D(cunits, tuplec))
	   model.add(layers.LeakyReLU(alpha=alpha))
	   #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    for i in range(1,dense_layers+1):
        model.add(layers.Dense(dunits))
        model.add(layers.LeakyReLU(alpha=0.15))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer=optimizers.RMSprop(lr=learning_rate, decay=decay)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model

# fix random seed for reproducibility

np.random.seed(123)
import random 
random.seed(123)
pgdl={'dense_layers':[1,3,5,7], 'conv_layers':[1,3,5,7],
                          'dunits':[16,32,64], 'cunits':[16,32,64], 'alpha':[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35], 'tuplec':[(1,1), (3,3), (5,5)],
                          'learning_rate':[0.03, 0.05, 0.07, 0.1, 0.013, 0.15],
                          'decay':[1e-6,1e-4,1e-2],
                          'epochs':[5, 10, 15, 20, 25, 30, 35, 40]}

effs={}
import copy
for i in []: #range(1):
    temp=copy.copy(pgdl)
    temp.update({n: random.sample(temp[n], 1)[0] for n in temp.keys()})
    we=deep_learning_create(temp['conv_layers'], temp['dense_layers'], temp['cunits'], temp['dunits'], temp['alpha'], temp['tuplec'], temp['learning_rate'], temp['decay'])
    we.fit(classTrainI, y_train_onehot, epochs=temp['epochs'], shuffle=True, callbacks=callbacks, validation_data=(classValidI, y_valid_onehot))
    predProbClass  = we.predict(classTestI)
    probsquark=predProbClass[:,0] 
    probsquark=probsquark/float(max(probsquark))
    probsquark=[(l-0.5)/0.5 for l in probsquark]
    effs[roc_auc_score(classTestY, probsquark)]=temp
print(effs)

#print(max(effs.keys()), effs[max(effs.keys())], ' are best hyperparameters')



predProbClass  = model.predict([classTestI, classTestL])
predProbClass1  = model.predict([classTrainI, classTrainL])

Rocs=[]
probs=[]
for n, predProbClass in enumerate([predProbClass,predProbClass1]):
	if n==1:
		
		classTestY=classTrainY
		predProbClass=predProbClass1
			
	classPredY = np.array([0 if p<0.5 else 1 for p in predProbClass])

	probsquark=predProbClass[:,0] 
	print(probsquark)
	probsquark=probsquark/float(max(probsquark))
	plt.figure((n+2)*30)
	probsquark=[(l-0.5)/0.5 for l in probsquark]

	probsquark0=[probsquark[i] for i in range(len(probsquark)) if classTestY[i]==0]
	probsquark1=[probsquark[i] for i in range(len(probsquark)) if classTestY[i]==1]
			
	plt.hist(probsquark0, bins=20, label='Gluons', range=(0,1), color='r', alpha=0.4, normed=True)
	plt.hist(probsquark1, bins=20, label='Quarks', range=(0,1), color='b', alpha=0.4, normed=True)
	plt.legend()
	plt.xlabel('BDT soft classification')
	plt.ylabel('Number of counts')
	probs.append([probsquark0, probsquark1])
	

	if n==0:
		plt.title('Test data')
		plt.savefig('testhistnn.png')
	if n==1:
		plt.title('Train data')			
		plt.savefig('trainhistnn.png')

	#Calculate bin effs for BDT
	print('accuracy %s %s \n' %(accuracy_score(classTestY,classPredY), trainToTestRatio))

	#Fill correct and incorrect dicts
	correctDict   = {0:[],1:[]}
	incorrectDict = {0:[],1:[]}

	for true, guess in zip(classTestY, classPredY):
	  if true==guess:
	    correctDict[true].append(1)
	  else:
	    incorrectDict[true].append(1)

	correctList   = []
	incorrectList = []

	#sum the weights in the dict for each cat
	for iCat in range(len(correctDict.keys())):
	  correctList.append(sum(correctDict[iCat]))
	  incorrectList.append(sum(incorrectDict[iCat]))

	#convert to numpy for pyplot
	correctArray   = np.asarray(correctList)
	incorrectArray = np.asarray(incorrectList)

	print('\nCorrect Weights')
	print(correctArray)
	print('\nIncorrect Weights')
	print(incorrectArray)
	
	effArrayBDT = np.array([float(i)/float((sum(correctArray)+sum(incorrectArray))) for i in correctArray])
	
	print('\nEffs are:')
	print(effArrayBDT)
	print('Average BDT accuracy is: %f'%effArrayBDT.mean())

	#plot ROC curve and calulate efficiencies
	fp, tp, ts=roc_curve(classTestY, probsquark)
		
	print('0.6 efficiency %s \n' % roc_auc_score(classTestY, probsquark)) #predProbClass[:,1]))
	Rocs.append([fp,tp])
	
from liklihoodproductanal import eefs
plt.figure(400)
fig, (ax) = plt.subplots(1, 1, sharex=True)

for n, a in enumerate(Rocs):
        fp,tp=a[0],a[1]
        if n==0:
                labs='Testing dataset CNN'
		plt.plot(fp,tp, 'b', label=labs)
		#plt.errorbar(fp, tp, [0.04 for k in fp])
		ax.fill_between(fp, 0, tp, facecolor='blue', alpha=0.3)#hatch="\ " ) 
        else:
                labs='Training dataset CNN'
		plt.plot(fp,tp, 'b', label=labs, linestyle='--')

for g, k  in enumerate([eefs[1]]):
	efficiencyg=k[0]
	efficiencyq=k[1]
	la='3 value LPD'
	m='m'
	if g==1:
		la='7 value LPD'
		m='r'
	plt.plot(efficiencyg, efficiencyq, color=m, label=la)

plt.plot([0,1], [0,1], 'k', label='blind guess')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid()
plt.legend(loc='lower right')
plt.savefig('nnROCcurve.png')
#plt.show()



