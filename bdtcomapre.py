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
#from liklihoodproductanal import eefs #efficiencyg, efficiencyq

events=[]
with open('EventData.csv', mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	line_count = 0
	for row in csv_reader:
		#if line_count>=1:
		events.append(ast.literal_eval(row['Eventdata']))

		line_count+=1

#global options
equaliseWeights = False
crossValidate = False
end=-1 	# -1 for all variables
loaded_model = pickle.load(open("qgdiscrimination.pickle.dat", "rb"))

allVars = np.array(['multiplicity','width','frag func', 'charge', 'charged multiplicity', 'hardness', 'pull'])
wpeffs=[[],[],[]]
workingpoints=np.arange(0.1, 1, 0.01)
for wp in workingpoints:
	workingpoint=wp
	print(wp)
	nClasses=2
	vbfprobsoft=[]
	jets=[]
	jetsids=[]
	truths=[]
	gghs=[]
	outputsbdt=[]
	jetid=[]
	for totaldata in events:
		#truejetlabels, jetdatas=[],[]
		#vbflabel, trueeventlabel=1,1

		if totaldata[3][1]>totaldata[3][2] or totaldata[3][0]>totaldata[3][2]:  #ignore background
			truejetlabels=np.array([k[1] for k in totaldata[:3] if k])
			jetdatas=[k[0] for k in totaldata[:3] if k]
			vbflabel=totaldata[3][:2].index(max(totaldata[3]))
			vbfprobs=totaldata[3][1]
			trueeventlabel=totaldata[4]
			if jetdatas:
				vbfMC  = xg.DMatrix(np.array(jetdatas),  label=truejetlabels, feature_names=allVars)
				predProbClass = loaded_model.predict(vbfMC).reshape(truejetlabels.shape[0], nClasses)
				predProbClass=(predProbClass-0.39228997)/(0.7765569-0.39228997)

				outputsbdt+=predProbClass[:,1].tolist()
				jetid+=truejetlabels.tolist()
				predClass=[0 if i<workingpoint else 1 for i in predProbClass[:,1]]
				jets.append(predClass)
				jetsids.append([vbflabel, trueeventlabel])
				vbfprobsoft.append(vbfprobs)
				truths.append(trueeventlabel)
				gghs.append(totaldata[3][0])
	#plt.figure()
	#plt.hist([outputsbdt[k] for k in range(len(outputsbdt)) if jetid[k]==0], alpha=0.3)				
	#plt.hist([outputsbdt[k] for k in range(len(outputsbdt)) if jetid[k]==1], alpha=0.3)
	#plt.show()
	def crude_guess(jets, vbf, ggh):
		quarks=len([k for k in jets if k==1])
		gluons=len([k for k in jets if k==0])
		if jets:	
			if jets[0]==1 and quarks>=2 and vbf>0.25 and ggh<0.35:		
				return 1
			if jets[0]==0 and gluons>2 and vbf<0.2 and ggh>0.6:		
				return 0
		return vbf

	crudes=[crude_guess(jets[i], vbfprobsoft[i], gghs[i]) for i in range(len(jets))]

	crudes1=[1 if crudes[i]>gghs[i] else 0 for i in range(len(crudes))]
	print(jets)
	print(len([k for k in crudes1 if k==1]))
	print(len([k for k in truths if k==1]))

	fp, tp, ts=roc_curve(truths, vbfprobsoft)
	fp1, tp1, ts1=roc_curve(truths, crudes)
	print(roc_auc_score(truths, vbfprobsoft))
	print(roc_auc_score(truths, crudes))
	wpeffs[0].append(roc_auc_score(truths, vbfprobsoft))
	wpeffs[1].append(roc_auc_score(truths, crudes))

	"""plt.figure() #plot histograms of outputs
	plt.hist([vbfprobsoft[i] for i in range(len(truths)) if truths[i]==1])
	plt.hist([vbfprobsoft[i] for i in range(len(truths)) if truths[i]==0])

	plt.figure()
	plt.hist([crudes[i] for i in range(len(truths)) if truths[i]==1])
	plt.hist([crudes[i] for i in range(len(truths)) if truths[i]==0])"""

	#plt.figure()
	#plt.plot(fp, tp, color='r', label='Current 3 class')
	#plt.plot(fp1, tp1, color='b', label='Simple cuts')
	
	#code for totals is us-current method. eg rw is are the events we got right (r) and current got wrong (w)
	truth=[l[1] for l in jetsids]
	rr=0
	rw=0
	wr=0
	ww=0	
	wr1, rw1=[],[]
	for i in range(len(jetsids)):
		if crudes1[i]==truths[i] and jetsids[i][0]==truths[i]:
			rr+=1
		if crudes1[i]==truths[i] and jetsids[i][0]!=truths[i]:
			rw+=1
			rw1.append(crudes[i])
		if crudes1[i]!=truths[i] and jetsids[i][0]==truths[i]:
			wr+=1
			wr1.append(crudes[i])
		if crudes1[i]!=truths[i] and jetsids[i][0]!=truths[i]:
			ww+=1

	print(len(crudes1))	
	print('FOR CRUDE: the good news is %s, the bad news is %s, out of %s' %(rw, wr, ww+rr+wr+rw))
	print(ww)
	#wr/rw to be used?
	print('right ggh ', len([k for k in rw1 if k==0]))
	print(len([k for k in rw1 if k==1]))
	print('wrong ggh ', len([k for k in wr1 if k==0]))
	print(len([k for k in wr1 if k==1]))

	
	#pad out jet identities for BDT- -1 is no jet
	for i in range(len(jets)):
		jets[i]+=[-1]*(3-len(jets[i]))
		jets[i]+=[jetsids[i][0]]


	#jets=[jets[i].append(jetsids[i][0]) for i in range(len(jets))]

	nps=np.random.RandomState(12302)  #Set random seed for reproducibility
	jets = nps.permutation(np.array(jets))

	nps=np.random.RandomState(12302)
	truth = nps.permutation(np.array(truth))

	nps=np.random.RandomState(12302)
	jetsids = nps.permutation(np.array(jetsids))
	
	#establish weights and truth ids etc.
	truth=[l[1] for l in jetsids]
	vbftag=vbfprobsoft #[l[0] for l in jetsids]
	weights=[len([i for i in truths if i==1])/len([i for i in truths if i==0]) if k==0 else 1 for k in truth]
	
	
	#split the data for training and testing
	truth0, truth1= np.split(np.array(truth), [int(0.8*len(truth))])
	jets0, jets1= np.split(np.array(jets), [int(0.8*len(truth))])
	vbf0, vbf1= np.split(np.array(vbftag), [int(0.8*len(truth))])
	w0, w1=np.split(np.array(weights), [int(0.8*len(truth))])


	trainingMC1 = xg.DMatrix(jets0, label=truth0, weight=w0)	 
	testingMC1 = xg.DMatrix(jets1, label=truth1, weight=w1)	

	trainParams={'learning_rate':0.1, 'max_depth':6, 'min_child_weight':50}
	trainParams['objective'] = 'multi:softprob'
	trainParams['num_class'] = nClasses
	trainParams['nthread'] = 1
	trainParams['eval_metric'] = 'mlogloss'

	crossValidate=True
	if crossValidate:
	    cvResult = xg.cv(trainParams, trainingMC1, num_boost_round=10, 
			   nfold = 4, early_stopping_rounds = 30, stratified = True, verbose_eval=True , seed = 12345)
	    trainParams['n_estimators'] = cvResult.shape[0]


	#train the model
	vbfmodel = xg.train(trainParams, trainingMC1)
		
	# predict classes
	predProbClass = vbfmodel.predict(testingMC1).reshape(truth1.shape[0], nClasses)
	classPredY = np.argmax(predProbClass, axis=1)
	print(classPredY) 	#make sure theyre not all 1!
	#code for totals is us-current method. eg rw is are the events we got right (r) and current got wrong (w)
	rr=0
	rw=0
	wr=0
	ww=0
	wr1, rw1=[],[]
	
	for i, k in enumerate(classPredY):
		if k==truth1[i] and vbf1[i]==truth1[i]:
			rr+=1
		if k==truth1[i] and vbf1[i]!=truth1[i]:
			rw+=1
			rw1.append(k)
		if k!=truth1[i] and vbf1[i]==truth1[i]:
			wr+=1
			wr1.append(k)
		if k!=truth1[i] and vbf1[i]!=truth1[i]:
			ww+=1

	print('FOR BDT: the good news is %s, the bad news is %s, out of %s'  %(rw, wr, ww+rr+wr+rw))

	print('right ggh ', len([k for k in rw1 if k==0]))
	print('right vbf ', len([k for k in rw1 if k==1]))
	print('wrong ggh ', len([k for k in wr1 if k==0]))
	print('wrong vbf ', len([k for k in wr1 if k==1]))

	print(roc_auc_score(truth1, predProbClass[:,1]))
	fp2, tp2, ts2=roc_curve(truth1, predProbClass[:,1])
	#plt.plot(fp2, tp2, color='g', label='BDT method')
	wpeffs[2].append(roc_auc_score(truth1, predProbClass[:,1]))

	#.xlabel('False positive rate')	
	#plt.ylabel('True positive rate')
	#plt.legend(loc='lower right')
	#plt.savefig('vbfroc.png')
	#plt.show()	
		

#plotting the granular comparison of AUCs for different working points
plt.figure()
plt.plot(workingpoints, wpeffs[0], color='r', label='Current method')
plt.plot(workingpoints, wpeffs[1], color='b', label='Simple cuts')
plt.plot(workingpoints, wpeffs[2], color='g', label='BDT method')
plt.xlabel('Working point for quark/gluon discrimination')
plt.ylabel('Area under ROC curve')
plt.legend()
plt.savefig('wpcomp1.png')
plt.show()
