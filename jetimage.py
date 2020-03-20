import ROOT as R
import pandas as pd
import numpy as np
import pickle
import xgboost
#import matplotlib.pyplot as plt
import re
from Matching_060320 import JetMatcher
#import seaborn as sns
from pandas import DataFrame

file1 = R.TFile.Open("/vols/build/cms/jwd18/public/jetConstitNtuples/GenFlavoursJan2020/ggH_210K.root") #open 1st file
R.gDirectory.pwd() #work in the open file
R.gDirectory.cd("vbfTagDumper/trees")  #change directory, just as you would in BASH
tree1=R.gDirectory.Get("vbfh_13TeV_GeneralDipho") #get the tree, from the directory

#basically, the gDirectory is your current working directory, assmued to be the root file when you open it, so when you open the tree, you usually automatcially pass it gDirectory. In this case you just have to move a bit too.

#repeat for other tree :)
file2 = R.TFile.Open("/vols/build/cms/jwd18/public/jetConstitNtuples/GenFlavoursJan2020/VBF_150K.root")
R.gDirectory.pwd()
R.gDirectory.cd("vbfTagDumper/trees")
tree2=R.gDirectory.Get("vbfh_13TeV_GeneralDipho")

file3 = R.TFile.Open("/vols/build/cms/jwd18/public/jetConstitNtuples/GenFlavoursJan2020/ggH_2017_jetConstituentFiles.root")
R.gDirectory.pwd()
R.gDirectory.cd("vbfTagDumper/trees")
tree3=R.gDirectory.Get("vbfh_13TeV_GeneralDipho")

def jet_images(): #Showvar = true for plotting the jet variables, histvar = true for returing histograms, false for BDTs
    
    data = JetMatcher(False, 0.2) #For Matching"

    multiplicities_t=[[],[]] #1st is ggH second is for VBF events
    widths_t=[[],[]]
    fragfunc_t=[[],[]]
    charges_t=[[],[]]
    chargedmult_t=[[],[]]
    Rs_t = [[],[]]
    Pulls_t = [[],[]]

    multiplicities_qg=[[],[]]
    widths_qg=[[],[]]
    fragfunc_qg=[[],[]]
    charges_qg=[[],[]]
    chargedmult_qg=[[],[]]
    Rs_qg = [[],[]]
    Pulls_qg = [[],[]]
    weights_qg=[[],[]]
    images_qg=[[],[]]
    pdgID = []
    pdgID_jet = []


    Quark_no = 0
    Gluon_no = 0

    Total = 0

    counter = 0
    trees = [tree1, tree2, tree3]
    for Dfile in data:
        FileMatched = 0
        
        Gluon_no_per = 0
        Quark_no_per = 0

        tree_i = data.index(Dfile)
        branches=trees[tree_i].GetListOfBranches()
        #----------------------------FILE LEVEL-----------------------------

        Jet_Charges = [[],[],[]]
        Jet_Chargedmults = [[],[],[]]
        Jet_Fragfuncs = [[],[],[]]
        Jet_Multiplicities = [[],[],[]]
        Widths = [[],[],[]]
        Jet_Energies = [[],[],[]]
        Rs = [[],[],[]]
        Pulls = [[],[],[]]

	images_q=[[],[],[]]
        Jet_Charges_q = [[],[],[]]
        Jet_Chargedmults_q = [[],[],[]]
        Jet_Fragfuncs_q = [[],[],[]]
        Jet_Multiplicities_q = [[],[],[]]
        Widths_q = [[],[],[]]
        Jet_Energies_q = [[],[],[]]
        Rs_q = [[],[],[]]
        Pulls_q = [[],[],[]]
        weights_q=[[],[],[]]

        Jet_Charges_g = [[],[],[]]
        Jet_Chargedmults_g = [[],[],[]]
        Jet_Fragfuncs_g = [[],[],[]]
        Jet_Multiplicities_g = [[],[],[]]
        Widths_g = [[],[],[]]
        Jet_Energies_g = [[],[],[]]
        Rs_g = [[],[],[]]
        Pulls_g = [[],[],[]]
        weights_g=[[],[],[]]
	images_g=[[],[],[]]

        for j in range(len(Dfile)): # Indiviual matched events
            Jets_R_Pt_Pdg = [[],[],[]] #A list of all the Rs, Pt and IDs for every jet (L, SL, SSL)
            event = Dfile[j][3]
            trees[tree_i].GetEntry(event) #Accessing the data of the matched event
            MatchedJets = [Dfile[j][0], Dfile[j][1], Dfile[j][2]] #Matched jets for that event
            for jet in range(len(MatchedJets)): #For each individual matched jet
                #----------------------------------------Lead, sublead, subsublead level-----------------------------------------------
                if MatchedJets[jet] != []:
                    Total = Total +1
                    FileMatched = FileMatched + 1
                    matcher = Dfile[j][jet][4] #Matcher tells you which jet (L,SL or SSL) was matched within R to a genjet, represented by (2,1,0) respectively
                    genjetID =  MatchedJets[jet][3] #Identifying the Jet
                    jet_eta = getattr(trees[tree_i], MatchedJets[jet][0]) #The value of eta,phi,Pt for the Jets
                    jet_phi =  getattr(trees[tree_i], MatchedJets[jet][1])
                    jet_Pt =  getattr(trees[tree_i], MatchedJets[jet][2])

                    if 1 <= np.abs(genjetID) <= 4 or genjetID == 21: #Restricting the lead jet to quarks except b (mod1-4) and gluons(21)
                        Jets_R_Pt_Pdg[jet].append([np.array([jet_eta, jet_phi]), jet_Pt, genjetID]) #The jet variables and the genjetID it was matched to.
                        Jet_Charge = []
                        Jet_Chargedmult = []
                        Jet_Fragfunc = []
                        Jet_Multiplicity = []
                        Jet_Pt = []
                        Jet_weight=[]
                        Width = []
                        R = []
                        Pull = []
                        Jet_Energy = []
                        Jet_Eta_Phis = []
			jet_images=[]
                 
                        pattern_leads1 =[["lead_jet_constit._eta","lead_jet_constit._phi","lead_jet_constit._pt","lead_jet_constit._charge","weight"],[r"sublead_jet_constit._eta",r"sublead_jet_constit._phi",r"sublead_jet_constit._pt",r"sublead_jet_constit._charge", r"weight"],[r"subsublead_jet_constit._eta",r"subsublead_jet_constit._phi",r"subsublead_jet_constit._pt",r"subsublead_jet_constit._charge",r"weight"]]

                        pattern_leads2 =[["lead_jet_constit.._eta","lead_jet_constit.._phi","lead_jet_constit.._pt","lead_jet_constit.._charge","weight"],[r"sublead_jet_constit.._eta",r"sublead_jet_constit.._phi",r"sublead_jet_constit.._pt",r"sublead_jet_constit.._charge", r"weight"],[r"subsublead_jet_constit.._eta",r"subsublead_jet_constit.._phi",r"subsublead_jet_constit.._pt",r"subsublead_jet_constit.._charge",r"weight"]]
                        pattern_leads1.reverse()
                        pattern_leads2.reverse()
                        
                        pattern_leads = pattern_leads1
                        for k in range(len(pattern_leads[jet])):
                            constit_no = 0
                            pattern_leads = pattern_leads1
                            pattern_var = []    
                            for b in branches:
                                if constit_no > 8:
                                    pattern_leads = pattern_leads2
                                name=b.GetName()                                
                                if re.match(pattern_leads[matcher][k],name):
                                    constit_no = constit_no +1
                                    pattern_value = getattr(trees[tree_i],name)
                                    if pattern_value != -999.0:
                                        pattern_var.append(pattern_value)
                                        
                                        #getattr(tree_i,name))   #print that branch value in the event (could add to a list instead or something
                                        #print name
                            if len(pattern_var) != 0:
                                if k == 0:
                                    Jet_Eta_Phis.append(pattern_var)
                                    Jet_Multiplicity.append(getattr(trees[tree_i],"n_constits"))
                                if k == 1:
                                    Jet_Eta_Phis.append(pattern_var)
                                if k == 2:
                                    Jet_Pt.append(pattern_var)
                                if k == 3:
                                    Chargedmult = []
                                    Jet_Charge.append(pattern_var)
                                    for charge in pattern_var:
                                        if charge != 0:
                                            Chargedmult.append(charge)  
                                    #if len(Chargedmult) != 0:
                                    Jet_Chargedmult.append(len(Chargedmult))
                                if k == 4:
                                    Jet_weight.append(pattern_var)

                                    
                        if Jet_Pt != []:
			    
                            if sum(Jet_Pt[0]) != 0:
                                Jet_Fragfunc.append(sum((pt**2) for pt in Jet_Pt[0])**0.5/sum(Jet_Pt[0]))
                                M_mat = [[],[]]
                                Del_etas = []
                                for eta in Jet_Eta_Phis[0]:
                                    Del_etas.append(Jets_R_Pt_Pdg[jet][0][0][0] - eta)
                                Del_phis = []
                                for phi in Jet_Eta_Phis[1]:
                                    Del_phis.append(Jets_R_Pt_Pdg[jet][0][0][1] - phi)
				jet_images.append([Del_etas, Del_phis, Jet_Pt[0], Jet_Charge[0]])
                                M11 = sum(Del_etas[i]**2*Jet_Pt[0][i]**2 for i in range(len(Del_etas)))
                                M12 = sum(Jet_Pt[0][i]**2*Del_etas[i]*Del_phis[i] for i in range(len(Del_etas)))
                                M22 = sum(Del_phis[i]**2*Jet_Pt[0][i]**2 for i in range(len(Del_phis)))

                                M_mat[0].append(M11)
                                M_mat[0].append(M12)
                                M_mat[1].append(M12)
                                M_mat[1].append(M22)

                                eigens= sorted(np.linalg.eig(np.array(M_mat))[0])[0]
                                Width.append((eigens/sum((pt**2) for pt in Jet_Pt[0]))**0.5)

                                R.append(max(Jet_Pt[0])/sum(pt for pt in Jet_Pt[0]))
                                Pull.append(sum(Jet_Pt[0][i]*np.abs(np.array(Del_etas[i], Del_phis[i]))**2 for i in range(len(Del_etas)))/jet_Pt)

                                Widths[jet].append(Width[0])
                                Jet_Multiplicities[jet].append(Jet_Multiplicity[0])
                                if Jet_Chargedmult != []:
                                    Jet_Chargedmults[jet].append(Jet_Chargedmult[0])
                                Jet_Charges[jet].append(Jet_Charge[0])
                                Jet_Fragfuncs[jet].append(Jet_Fragfunc[0])
                                Rs[jet].append(R[0])
                                Pulls[jet].append(Pull[0])

                                if 1<= np.abs(genjetID)<=4:
                                    Quark_no= Quark_no+1
                                    Quark_no_per= Quark_no_per +1
                                    Widths_q[jet].append(Width[0])
                                    if Jet_Chargedmult != []:
                                        Jet_Chargedmults_q[jet].append(Jet_Chargedmult[0])
                                    Jet_Multiplicities_q[jet].append(Jet_Multiplicity[0])
                                    Jet_Charges_q[jet].append(Jet_Charge[0])
                                    Jet_Fragfuncs_q[jet].append(Jet_Fragfunc[0])
                                    Rs_q[jet].append(R[0])
                                    Pulls_q[jet].append(Pull[0])
                                    weights_q[jet].append(Jet_weight[0])
           			    images_q[jet].append(jet_images[0])

                                if np.abs(genjetID) == 21:
                                    Gluon_no = Gluon_no+1
                                    Gluon_no_per = Gluon_no_per+1
                                    Widths_g[jet].append(Width[0])
                                    Jet_Multiplicities_g[jet].append(Jet_Multiplicity[0])
                                    Jet_Charges_g[jet].append(Jet_Charge[0])
                                    if Jet_Chargedmult != []: #For when we only want to consider jets with at least 1 charged particle, works equally with no cps
                                        Jet_Chargedmults_g[jet].append(Jet_Chargedmult[0])  
                                    Jet_Fragfuncs_g[jet].append(Jet_Fragfunc[0])
                                    Rs_g[jet].append(R[0])
                                    Pulls_g[jet].append(Pull[0])
                                    weights_g[jet].append(Jet_weight[0])
			            images_g[jet].append(jet_images[0])

        if counter == 2:
            multiplicities_t[0].append(Jet_Multiplicities)
            widths_t[0].append(Widths)
            fragfunc_t[0].append(Jet_Fragfuncs)
            charges_t[0].append(Jet_Charges)
            chargedmult_t[0].append(Jet_Chargedmults)
            Rs_t[0].append(Rs)
            Pulls_t[0].append(Pulls)
        else:
            multiplicities_t[counter].append(Jet_Multiplicities)
            widths_t[counter].append(Widths)
            fragfunc_t[counter].append(Jet_Fragfuncs)
            charges_t[counter].append(Jet_Charges)
            chargedmult_t[counter].append(Jet_Chargedmults)
            Rs_t[counter].append(Rs)
            Pulls_t[counter].append(Pulls)

        
        multiplicities_qg[0].append(Jet_Multiplicities_q) #[[[#Lead jet multiplicities],[],[]] #File1 ,[[],[],[]] #File2 ,[[],[],[]]]
        widths_qg[0].append(Widths_q)
        fragfunc_qg[0].append(Jet_Fragfuncs_q)
        charges_qg[0].append(Jet_Charges_q)
        chargedmult_qg[0].append(Jet_Chargedmults_q)
        Rs_qg[0].append(Rs_q)
        Pulls_qg[0].append(Pulls_q)
        weights_qg[0].append(weights_q)
	images_qg[0].append(images_q)
        multiplicities_qg[1].append(Jet_Multiplicities_g)
        widths_qg[1].append(Widths_g)
        fragfunc_qg[1].append(Jet_Fragfuncs_g)
        charges_qg[1].append(Jet_Charges_g)
        chargedmult_qg[1].append(Jet_Chargedmults_g)
        Rs_qg[1].append(Rs_g)
        Pulls_qg[1].append(Pulls_g)
        weights_qg[1].append(weights_g)
	images_qg[1].append(images_g)

        print "File:", counter
        print "Matched events in File:",counter,"is", FileMatched
        counter +=1
        
        print "Total No. of quarks in file:",counter,"is:", Quark_no_per
        print "Total no. of gluons in file:",counter, "is:", Gluon_no_per

    print "Total No. of quarks:", Quark_no
    print "Total no. of gluons:", Gluon_no
    print "Total number of matched events:", Total
    
    varsnn=[]
    for i, j in enumerate(images_qg):
	typep=1
	if i==1:
		typep=0
	for k in j:
		for l in k:
			for m in l: 
	 			varsnn.append([typep]+m)
    labels=[]
    images=[]
    for q in range(len(varsnn)):
    	h=np.histogram2d(varsnn[q][1], varsnn[q][2], weights=varsnn[q][3], bins=20, range=[[-0.5,0.5],[-0.5,0.5]], density=True)
    	j=np.histogram2d(varsnn[q][1], varsnn[q][2], weights=np.absolute(varsnn[q][4]), bins=20, range=[[-0.5,0.5],[-0.5,0.5]], density=True)[0]
	pl=np.array([np.array([np.array([np.nan_to_num(h[0])[i][l], np.nan_to_num(j)[i][l], 0]) for l in range(len(j[0]))]) for i in range(len(j))])
	images.append(pl)
 	labels.append(varsnn[q][0])    
    return np.array(images), np.array(labels)

datas, labels=jet_images()

np.save('jetimagesnnsmol.npy', datas)
np.save('jetlabelsnnsmol.npy', labels)
