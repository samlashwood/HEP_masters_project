#Usual imports for XGB
import xgboost as xg
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
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
import ast
import csv
from liklihoodproductanal import eefs #efficiencyg, efficiencyq

def hyperparamsrand(classTrainI, label_encoded_y):
	print('working')
	modelclass = XGBClassifier(random_state=12345)
	param_grid = dict(n_estimators=range(100, 1000, 100), max_depth=range(1,6), min_child_weight=range(400,1000,100), learning_rate=np.arange(0,1,0.1) )
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
	counter=0
	print('here we go')
	while counter<100:       #all([s[0] for s in list(param_grid.values())])==True and
		if sum([len(a) for a in param_grid.values()])<=4:
			print('brokened')
			break
		nits=10
		if counter==0:
			nits=30
		elif counter==1:
			nits=20
		grid_search = RandomizedSearchCV(modelclass, param_grid, n_iter=nits, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
		# encode string class values as integers
		#label_encoded_y = LabelEncoder().fit_transform(classTrainY)
		grid_result = grid_search.fit(classTrainI, label_encoded_y)
		best=grid_result.best_params_
		keys=best.keys()
		newparams={}
		print('best', best)
		for i, key in enumerate(keys):
			if len(param_grid[key])>1:
				length=(param_grid[key][1]-param_grid[key][0])

				increment=(length)//10
				if length==1:
					increment=1
				#print(key, type(key))
				if key=='learning_rate'	or length<1:
					increment=(length)/10
					if increment>0.001:
						if best[key]-5*increment>0:
							a=best[key]-increment*5
						else:
							a=best[key]-(increment*int(best[key]/increment))
						newparams[key]=list(np.arange(a, best[key]+5*increment, increment))
						#print(newparams[key], key, increment, 'yay')
					else:
						newparams[key]=[best[key]]
				if increment==0:
					increment=1
				if increment!=length and length>=1:
					if best[key]-5*increment<=0:
						newparams[key]=range(best[key]-(increment*(best[key]//increment)), best[key]+((best[key]//increment)*increment), increment)
					else:	
						newparams[key]=range(best[key]-5*increment, best[key]+5*increment, increment) 
					#print(newparams[key], key, length, 'yay')
				elif increment==length:
					newparams[key]=[best[key]]
			else:	
				newparams[key]=[best[key]]
		print(newparams)
		
		if newparams==param_grid:
			break	
		else:
			param_grid=newparams
		counter+=1
	return best


datajets=[]
with open('JetData.csv', mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	line_count = 0
	for row in csv_reader:
		#if line_count>=1:
		datajets.append(ast.literal_eval(ast.literal_eval(row[',datajets'])[1]))
		line_count+=1

#global options
equaliseWeights = False
crossValidate = False
end=-1 	# -1 for all variables

#calculate jet data and format correctly- construct data for training- of the form data_array; labels, liklihood labels, weights, weights'
#the data array should be a list of arrays- each array corresponds to a jet variable (frag funct, multiplicity etc.)- one for each jet.
#The first element in this array is the list of jet IDs from pgdid (g/q), the second is the ID according to likelihood, the third is the array of weights for the jets (a jet variable in the root file, the forth can be ignored atm
#using the same convention for identity as the discriminator- (0 for gluons, 1 for quarks)

ids=[d[0] for d in datajets] # if d[7]>0]
data=[d[1:end] for d in datajets] #s if d[7]>0])

weights=[d[-1] for d in datajets] #s if d[7]>0] #remove negative weights?
data=[d[:len(d)-1] for d in data]
for i in range(len(data)):
	data[i].append(weights[i])

labelsetc=[ids] #,ids,weights]
dataends=np.size(data,0)  #find the length of data array, used to split total array later


#format list (modified code)
t1=data
t2=labelsetc
trainTotal=t1
for k in t2:
	trainTotal.append(k)
trainTotal=np.array(trainTotal)

#set up features
allVars = np.array(['multiplicity','width','frag func', 'charge', 'charged multiplicity', 'hardness', 'pull'])
if end!=-1:	
	allVars=allVars[:end-1]

#Note: validation set comes from train (cross validation)

accuraciestt=[[],[]]  #accuracies from different training ratios

for place, trainToTestRatio in enumerate([0.9]): #np.arange(0.05,1,0.05)):   #[0.6]:	

	hypereffs=[[],[]] 

	#shuffle dataset and shape correctly

	nps=np.random.RandomState(12302)  #Set random seed for reproducibility
	nClasses = 2   #binary output

	theShape = trainTotal[0:dataends].shape[0]
	classShuffle = nps.permutation(theShape)
	classTrainLimit = int(theShape*trainToTestRatio)	  
	  
	#setup the various datasets for multiclass training as NUMPY arrays
	classI        = trainTotal[0:dataends]
	classY        = trainTotal[dataends]

	#shuffle datasets
	classI = [classI[i] for i in classShuffle]
	classY = [classY[i] for i in classShuffle]

	#split datasets
	classTrainY, classTestY           = np.split( classY,  [classTrainLimit] )
	classTrainI=np.array(classI[:classTrainLimit])
	classTestI=np.array(classI[classTrainLimit:])
		
	#formatting test and train dataset as DMatrices
	trainingMC = xg.DMatrix(classTrainI, label=classTrainY, feature_names=allVars)	  
	testingMC  = xg.DMatrix(classTestI,  label=classTestY, feature_names=allVars)

	#trainParams={}
	#trainParams = {'n_estimators': 800, 'learning_rate': 0.153, 'max_depth': 30, 'min_child_weight': 8}  #very OT
	#trainParams={'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 700}  #this great
	#trainParams={'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 450}  #this ok

	trainParams={'learning_rate':0.1, 'max_depth':4, 'min_child_weight':350}
	trainParams['objective'] = 'multi:softprob'
	trainParams['num_class'] = nClasses
	trainParams['nthread'] = 1
	trainParams['eval_metric'] = 'mlogloss'

	crossValidate=True
	if crossValidate:
	    cvResult = xg.cv(trainParams, trainingMC, num_boost_round=50, 
			   nfold = 4, early_stopping_rounds = 30, stratified = True, verbose_eval=True , seed = 12345)
	    trainParams['n_estimators'] = cvResult.shape[0]

	print('nests', trainParams['n_estimators'])
	
	#train the model
	nClassesModel = xg.train(trainParams, trainingMC)
	
	#print('logloss %s %s \n' % (log_loss(classTestY, nClassesModel.predict(testingMC)), trainToTestRatio))

	# predict classes
	predProbClass = nClassesModel.predict(testingMC).reshape(classTestY.shape[0],nClasses)
	predProbClass1 = nClassesModel.predict(trainingMC).reshape(classTrainY.shape[0], nClasses)
	
	#examine the accuracies for test and train sets to mitigate overtraining
	Rocs=[]
	probs=[]
	for n, predProbClass in enumerate([predProbClass,predProbClass1]):
		if n==1:
			
			classTestY=classTrainY
			
			testingMC=trainingMC
			predProbClass=predProbClass1
				
		classPredY = np.argmax(predProbClass, axis=1) 

		probsquark=predProbClass[:,1] #/(predProbClass[:,1]+predProbClass[:,0])
		print(min(probsquark), max(probsquark))
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
			plt.savefig('testhist.png')
		if n==1:
			plt.title('Train data')			
			plt.savefig('trainhist.png')

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

		accuraciestt[n].append(effArrayBDT.mean())

		if place==0:

			xg.plot_importance(nClassesModel,grid=True, importance_type='gain', show_values=False)
			plt.savefig('feature_importances_final.png')

			#plot ROC curve and calulate efficiencies
			fp, tp, ts=roc_curve(classTestY, probsquark)
			
			print('0.6 efficiency %s \n' % roc_auc_score(classTestY, probsquark)) #predProbClass[:,1]))
			Rocs.append([fp,tp])
			hypereffs[n].append(roc_auc_score(classTestY, predProbClass[:,1]))

			
			if n==0:
				label_encoded_y = LabelEncoder().fit_transform(classTrainY)    # encode string class values as integers
				#print('hyperparameters')
				#print('rand', hyperparamsrand(classTrainI, label_encoded_y))
				
				
				model=xg.XGBClassifier(random_state=12345)
				model.fit(classTrainI, classTrainY)
				model.feature_names=allVars
				d=model.get_booster()				
				listtree=d.get_dump()

				for k,tr in enumerate(listtree):
					temp=tr
					for m, var in enumerate(allVars):
						string='f' + str(m)
						temp=temp.replace(string, var)	
					listtree[k]=temp	

				"""for jk in listtree[1:5]:
					print(jk)"""			

import pickle 
pickle.dump(nClassesModel, open("qgdiscrimination.pickle.dat", "wb"))	
nClassesModel.save_model('quarkgluon.model')	
			
"""
#plot graph of varied max tree depth hyperparameter
plt.figure(30)
plt.plot(range(1,1001,10),hypereffs[0], label='testing data')
plt.plot(range(1,1001,10),hypereffs[1], label='training data')
plt.xlabel('Minimum child weight of trees')
plt.ylabel('Discrimination efficency')
plt.legend()
plt.savefig('minchild600all.png')


"""       

 
plt.figure(400)
fig, (ax) = plt.subplots(1, 1, sharex=True)



for n, a in enumerate(Rocs):
        fp,tp=a[0],a[1]
        if n==0:
                labs='Testing dataset BDT'
		plt.plot(fp,tp, 'b', label=labs)
		#plt.errorbar(fp, tp, [0.04 for k in fp])
		ax.fill_between(fp, 0, tp, facecolor='blue', alpha=0.3)#hatch="\ " ) 
        else:
                labs='Training dataset BDT'
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

plt.plot([0,1],[0,1], 'k', label='blind guess')  
ax.fill_between(efficiencyg, 0, efficiencyq, facecolor='red', alpha=0.2) 
fp, tp=Rocs[0][0], Rocs[0][1]
No_tps = len(tp)
import scipy as sp
points = sp.arange(int(round(No_tps/20)), int(round(9*No_tps/10)), int(round(No_tps/20))).tolist()
errors=np.array([0, 0.02, 0.012, 0.014, 0.012, 0.01, 0.01, 0.01, 0.015, 0.015, 0.01, 0.02, 0.015, 0.012, 0.012, 0.01, 0.013, 0.01, 0.018, 0.018,0])
xpoints=np.array([0, 0.017]+[fp[g] for g in points]+[1])
ypoints=np.array([0, 0.1]+[tp[g] for g in points]+[1])

ax.set(xlim=(0,1), ylim=(0,1))    
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

plt.fill_between(xpoints, ypoints+errors, ypoints-errors, facecolor='green', alpha=0.4, label='error')
plt.grid()
plt.legend(loc='lower right')
plt.savefig('BDT_roccurve.png')
#print('ROCS', Rocs)

"""
#plot overtraining graph

plt.figure(200)
plt.xlabel('Ratio of data used to train')
plt.ylabel('Accuracy of BDT')
#plt.title('Graph to study overtraining')
plt.plot(np.arange(0.01,1,0.01), accuraciestt[0], label='accuracies for testing')
plt.plot(np.arange(0.01,1,0.01), accuraciestt[1], label='accuracies for training')
plt.legend()
plt.savefig('Accuracy of BDT')"""

from scipy.stats import ks_2samp
a, Gluon_KS = ks_2samp(probs[0][0],probs[1][0])
a, Quark_KS = ks_2samp(probs[0][1],probs[1][1])
print(Gluon_KS, Quark_KS)

print('acc', accuraciestt)
plt.show()
