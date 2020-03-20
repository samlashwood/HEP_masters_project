#from MatchAndAnalyse_290120 import jet_variables
import ROOT as R
import pandas as pd
import numpy as np
import pickle
import matplotlib
#matplotlib.use('Agg')
import xgboost
import matplotlib.pyplot as plt
import re

import ast
import csv
data=[]
histbinsboth=[]
with open('JetData.csv', mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	line_count = 0
	for row in csv_reader:
		#if line_count>=1:
		data.append(ast.literal_eval(ast.literal_eval(row[',datajets'])[1]))
		line_count+=1
"""
with open('Histbins.csv', mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	line_count = 0
	for row in csv_reader:
		if line_count==0:
			histbinsboth.append(ast.literal_eval(row['histbins']))
		line_count+=1"""

weights=[d[-1] for d in data] #s if d[7]>0] #remove negative weights?
data=[d[:len(d)-2] for d in data]
for i in range(len(data)):
	data[i].append(weights[i])

binno=50

qmat=np.histogram([d[1] for d in data if d[0]==1], bins=(max([d[5] for d in data if d[0]==0])+1), range=(-0.5, (max([d[5] for d in data if d[0]==0])+0.5)), density=True)
gmat=np.histogram([d[1] for d in data if d[0]==0], bins=(max([d[5] for d in data if d[0]==0])+1), range=(-0.5, (max([d[5] for d in data if d[0]==0])+0.5)), density=True)
qwid=np.histogram([d[2] for d in data if d[0]==1], bins=binno, density=True)
gwid=np.histogram([d[2] for d in data if d[0]==0], bins=binno, density=True)
qff=np.histogram([d[3] for d in data if d[0]==1], bins=binno, density=True)
gff=np.histogram([d[3] for d in data if d[0]==0], bins=binno, density=True)
qcha=np.histogram([d[4] for d in data if d[0]==1], bins=binno, density=True)
gcha=np.histogram([d[4] for d in data if d[0]==0], bins=binno, density=True)
qcc=np.histogram([d[5] for d in data if d[0]==1], bins=(max([d[5] for d in data if d[0]==0])+1), range=(-0.5, (max([d[5] for d in data if d[0]==0])+0.5)), density=True)
gcc=np.histogram([d[5] for d in data if d[0]==0], bins=(max([d[5] for d in data if d[0]==0])+1), range=(-0.5, (max([d[5] for d in data if d[0]==0])+0.5)), density=True)
qhar=np.histogram([d[6] for d in data if d[0]==1], bins=binno, density=True)
ghar=np.histogram([d[6] for d in data if d[0]==0], bins=binno, density=True)
qpul=np.histogram([d[7] for d in data if d[0]==1], bins=binno, range=(0,0.04), density=True)
gpul=np.histogram([d[7] for d in data if d[0]==0], bins=binno, range=(0,0.04), density=True)


#plt.hist([d[5] for d in data if d[0]==1], bins=(max([d[5] for d in data if d[0]==0])+1), range=(-0.5, (max([d[5] for d in data if d[0]==0])+0.5)), normed=True)

def averagebins(index, hist):
#-------------function to return the average of filled histgram bins in case a bin is left empty by lack of data
	i=0
	while i==0: #continue until filled bin found
		if type(index)==int:
			index=[index]
		if len(index)>=len(hist):  #edge case of empty histogram
			break
		average=0
		for k in index: 
			average+=hist[k]
		average=average/len(index)   #calculate average of bins considered thus far
		i=average
		if index[0]!=0:  #edge case of edges of histogram
			index.insert(0, index[0]-1)
		if index[-1]!=len(hist)-1:
			index.append(index[-1]+1)
	return i*(len(index)-1)


def findhist(value, edges, heights):    
#----------function find PDF value using histograms---------------- 
	edges=edges[:-1]
	width=0
	i=1
	while width==0:
		width=float(edges[-i]-edges[-(i+1)])  #width of bin assuming equal widths
		i+=1
	for k1 in range(len(edges)):		
		if value>=(edges[k1]) and value<=(edges[k1]+width): 
			if heights[k1]!=0:
				return np.abs(heights[k1]*width)   #calculate area of bin containing value (probability)
			else:
				a=averagebins(k1,heights)    #if required bin empty, get averaged value
				return np.abs(a*width)		
		else:
			pass
	return 0.5


liks=[]
for numvar in [0,1]:

	liklihoodq=[[],[]]
	for j in range(len(data)):
		i=0
		i1=2
		#find probabilities for quark PDF
		m=findhist(data[j][1], qmat[1], qmat[0])
		w=findhist(data[j][2], qwid[1], qwid[0])
		f=findhist(data[j][3], qff[1], qff[0])
		c=findhist(data[j][4], qcha[1], qcha[0])
		cc=findhist(data[j][5], qcc[1], qcc[0])
		h=findhist(data[j][6], qhar[1], qhar[0])
		p=findhist(data[j][7], qpul[1], qpul[0])

		#find probabilites for gluon PDF
		m1=findhist(data[j][1], gmat[1], gmat[0])
		w1=findhist(data[j][2], qwid[1], gwid[0])
		f1=findhist(data[j][3], gff[1], gff[0])
		c1=findhist(data[j][4], gcha[1], gcha[0])
		cc1=findhist(data[j][5], gcc[1], gcc[0])
		h1=findhist(data[j][6], ghar[1], ghar[0])
		p1=findhist(data[j][7], gpul[1], gpul[0])

		product=(f/(f+f1))*(w/(w+w1))*(m/(m+m1))

		if numvar==1:
			
			product=(f/(f+f1))*(w/(w+w1))*(m/(m+m1))*(c/(c+c1))*(cc/(cc+cc1))*(p/(p+p1))*(h/(h+h1))   #calculate product so that minimising happens for gluons
		
		liklihoodq[data[j][0]].append(product)	

	j=[max(k) for k in liklihoodq]  #find factor to scale likelihood products by 
	liklihoodq=[[i/max(j) for i in k] for k in liklihoodq]
	liklihoodq=[g[:-1] for g in liklihoodq]
	liks.append(liklihoodq)


eefs=[]
for ew, liklihood in enumerate(liks):
	if ew==0:
		plt.figure(100)     #plot liklihood products
		rang=(0,1)
	if ew==1:
		plt.figure(101)     #plot liklihood products
		rang=(0,1)
	plt.ylabel("Number of entries")
	plt.xlabel('Liklihood discriminant product')
	range_i = (0,1)
	

	bins_ql = np.histogram(liklihood[1], range = rang,  bins= binno)
	bins_ql_norml = np.histogram(liklihood[1], range = rang,  bins= binno, density=True)
	bins_q_norml =  plt.hist(liklihood[1], bins= binno, range = rang, color = 'b', alpha = 0.6, normed= True)  
	norm_const_ql =  bins_q_norml[0][0]/ bins_ql[0][0]
	bincentres_ql = 0.5*(bins_ql[1][1:]+bins_ql[1][:-1])
	m = plt.errorbar(bincentres_ql,bins_q_norml[0],yerr=np.array(bins_ql[0])**0.5*norm_const_ql ,fmt=',',capsize=3, color = 'b')

	bins_gl = np.histogram(liklihood[0], range = rang, bins=binno)
	bins_gl_norml = np.histogram(liklihood[0], range = rang, bins=binno, density=True)
	bins_g_norml =  plt.hist(liklihood[0], bins=binno, range = rang, color = 'r', alpha = 0.4, normed = True) 
	norm_const_gl =  bins_g_norml[0][3]/ bins_gl[0][3]
	bincentres_gl = 0.5*(bins_gl[1][1:]+bins_gl[1][:-1])
	n = plt.errorbar(bincentres_gl,bins_g_norml[0],yerr=np.array(bins_gl[0])**0.5*norm_const_gl ,fmt=',',capsize=3, color = 'r')
			
	plt.legend((m,n),("Quark", "Gluon"))
	plt.title("plot of liklihood product")
	plt.xlim(left=0)
	plt.ylim(bottom=0)

	if ew==0:
		plt.savefig('likelihoodhist3.png')
	if ew==1:
		plt.savefig('likelihoodhist6.png')

	#work out efficiencies for ROC curves
	efficiencyq=[]
	efficiencyg=[]
	totq=0
	totg=0

	bq=bins_ql_norml[0][::-1] 
	bg=bins_gl_norml[0][::-1]

	for k in np.arange(binno):            #calculate efficiencies using separation power at different bins- summing areas progressively
		totq+=(bq[k]/float(binno))
		efficiencyq.append(totq)
		totg+=(bg[k]/float(binno))
		efficiencyg.append(totg)

	plt.figure(102)	
	if ew==0:	
		labs='3 variable LPD'
	if ew==1:
		labs='7 variable LPD'

	plt.plot(efficiencyg, efficiencyq, label=labs)
	
	plt.xlabel("False positive rate")
	plt.ylabel("True positive rate")
	eefs.append([efficiencyg, efficiencyq])
	print("efficiency is ", np.trapz(efficiencyq, efficiencyg))  #efficiency is area under curve
	

plt.plot([0, 1],[0, 1], 'r', label='blind guess')
plt.title("Efficiency plot")
plt.legend(loc='lower right')
plt.savefig('likelihoodroc6.png')
#plt.show()
	

