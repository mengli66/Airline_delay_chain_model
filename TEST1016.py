import os
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from numpy import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
#from sklearn.datasets import make_classification
#from sklearn import preprocessing
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

os.chdir('W:\\Downloads\\sample\\1016')

input_file = "1016_iter_arrange_py.csv"
# comma delimited is the default
df = pd.read_csv(input_file, header = 0)

# for space delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = " ")

# for tab delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = "\t")

# put the original column names in a python list
original_headers = list(df.columns.values)

# remove the non-numeric columns
df = df._get_numeric_data()

# put the numeric column names in a python list
numeric_headers = list(df.columns.values)

# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.as_matrix()

where_are_NaNs = isnan(numpy_array)

numpy_array[where_are_NaNs] = 0
           
min_max_scaler=preprocessing.MinMaxScaler()
           
l=np.shape(numpy_array)
p=l[0]
p1=15193
p2=6994
p3=3546
p4=1409
p5=662
p6=285
p7=139
p8=43
p9=19
#0,30-,,,1,30+
clf_AD0 = joblib.load('smote_ad_py1.pkl') 
clf_DD0 = joblib.load('smote_dd_py1.pkl') 
clf_AD1 = joblib.load('X30+_AD_smote_py.pkl') 
clf_DD1 = joblib.load('X30+_DD_smote_py.pkl') 


numpy_array1=numpy_array[p1:,:]
yad1=numpy_array1[:,21]
zad1=numpy_array1[:,22]
cltype=numpy_array1[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array1[loct0,:]
data0=data0[0,:,:]
data1=numpy_array1[loct1,:]
data1=data1[0,:,:]
xad1_0=data0[:,8:21]
zad1_0=clf_AD0.predict(xad1_0)

xad1_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
zad1_1=clf_AD1.predict(xad1_1)

zad1[loct0]=zad1_0
zad1[loct1]=zad1_1

#xad1=numpy_array[p1:,2:9]
#yad1=numpy_array[p1:,9]
#zad1=clf_AD.predict(xad1)

#2
t1=p1
t2=p2
numpy_array2=numpy_array[t2:t1,:]
yad2=numpy_array2[:,21]
ydd2=numpy_array2[:,20]
zad2=np.zeros((t1-t2))
cltype=numpy_array2[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array2[loct0,:]
data0=data0[0,:,:]
data1=numpy_array2[loct1,:]
data1=data1[0,:,:]

apt2=numpy_array2[:,2]
ladg2=numpy_array2[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in apt2:
    ladg2[i-1]=max(zad1[i-1]-max(apt2[i-1]/15,0)+2.75,0)
ladg2_0=ladg2[loct0]
ladg2_1=ladg2[loct1]
    
xdd2_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg2_0).T))
zdd2_0=clf_DD0.predict(xdd2_0)
xdd2_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg2_1).T))
zdd2_1=clf_DD1.predict(xdd2_1)
zdd2=np.zeros((t1-t2))
#ZERO PRE-AD?
zdd2[loct0]=zdd2_0
zdd2[loct1]=zdd2_1
#use dd?
xad2_0=data0[:,8:21]
xad2_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad2_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd2_0).T))
#xad2_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd2_1).T))
zad2_0=clf_AD0.predict(xad2_0)
zad2_1=clf_AD1.predict(xad2_1)
zad2[loct0]=zad2_0
zad2[loct1]=zad2_1

#3
t1=p2
t2=p3
numpy_array3=numpy_array[t2:t1,:]
yad3=numpy_array3[:,21]
ydd3=numpy_array3[:,20]
zad3=np.zeros((t1-t2))
cltype=numpy_array3[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array3[loct0,:]
data0=data0[0,:,:]
data1=numpy_array3[loct1,:]
data1=data1[0,:,:]

apt3=numpy_array3[:,2]
ladg3=numpy_array3[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in apt3:
    ladg3[i-1]=max(zad2[i-1]-max(apt3[i-1]/15,0)+2.75,0)
ladg3_0=ladg3[loct0]
ladg3_1=ladg3[loct1]

xdd3_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg3_0).T))
zdd3_0=clf_DD0.predict(xdd3_0)
xdd3_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg3_1).T))
zdd3_1=clf_DD1.predict(xdd3_1)
zdd3=np.zeros((t1-t2))

zdd3[loct0]=zdd3_0
zdd3[loct1]=zdd3_1
#use dd?
xad3_0=data0[:,8:21]
xad3_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad3_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd3_0).T))
#xad3_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd3_1).T))
zad3_0=clf_AD0.predict(xad3_0)
zad3_1=clf_AD1.predict(xad3_1)
zad3[loct0]=zad3_0
zad3[loct1]=zad3_1

#4
t1=p3
t2=p4
numpy_array4=numpy_array[t2:t1,:]
yad4=numpy_array4[:,21]
ydd4=numpy_array4[:,20]
zad4=np.zeros((t1-t2))
cltype=numpy_array4[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array4[loct0,:]
data0=data0[0,:,:]
data1=numpy_array4[loct1,:]
data1=data1[0,:,:]

apt4=numpy_array4[:,2]
ladg4=numpy_array4[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in apt4:
    ladg4[i-1]=max(zad3[i-1]-max(apt4[i-1]/15,0)+2.75,0)
ladg4_0=ladg4[loct0]
ladg4_1=ladg4[loct1]

xdd4_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg4_0).T))
zdd4_0=clf_DD0.predict(xdd4_0)
xdd4_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg4_1).T))
zdd4_1=clf_DD1.predict(xdd4_1)
zdd4=np.zeros((t1-t2))
#ZERO PRE-AD?
zdd4[loct0]=zdd4_0
zdd4[loct1]=zdd4_1
#use dd?
xad4_0=data0[:,8:21]
xad4_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad4_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd4_0).T))
#xad4_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd4_1).T))
zad4_0=clf_AD0.predict(xad4_0)
zad4_1=clf_AD1.predict(xad4_1)
zad4[loct0]=zad4_0
zad4[loct1]=zad4_1

#5
t1=p4
t2=p5
numpy_array5=numpy_array[t2:t1,:]
yad5=numpy_array5[:,21]
ydd5=numpy_array5[:,20]
zad5=np.zeros((t1-t2))
cltype=numpy_array5[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array5[loct0,:]
data0=data0[0,:,:]
data1=numpy_array5[loct1,:]
data1=data1[0,:,:]

apt5=numpy_array5[:,2]
ladg5=numpy_array5[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in apt5:
    ladg5[i-1]=max(zad4[i-1]-max(apt5[i-1]/15,0)+2.75,0)
ladg5_0=ladg5[loct0]
ladg5_1=ladg5[loct1]

xdd5_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg5_0).T))
zdd5_0=clf_DD0.predict(xdd5_0)
xdd5_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg5_1).T))
zdd5_1=clf_DD1.predict(xdd5_1)
zdd5=np.zeros((t1-t2))
#ZERO PRE-AD?
zdd5[loct0]=zdd5_0
zdd5[loct1]=zdd5_1
#use dd?
xad5_0=data0[:,8:21]
xad5_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad5_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd5_0).T))
#xad5_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd5_1).T))
zad5_0=clf_AD0.predict(xad5_0)
zad5_1=clf_AD1.predict(xad5_1)
zad5[loct0]=zad5_0
zad5[loct1]=zad5_1

#6
t1=p5
t2=p6
numpy_array6=numpy_array[t2:t1,:]
yad6=numpy_array6[:,21]
ydd6=numpy_array6[:,20]
zad6=np.zeros((t1-t2))
cltype=numpy_array6[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array6[loct0,:]
data0=data0[0,:,:]
data1=numpy_array6[loct1,:]
data1=data1[0,:,:]

apt6=numpy_array6[:,2]
ladg6=numpy_array6[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in ladg6:
    ladg6[i-1]=max(zad5[i-1]-max(apt6[i-1]/15,0)+2.75,0)
ladg6_0=ladg6[loct0]
ladg6_1=ladg6[loct1]

xdd6_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg6_0).T))
zdd6_0=clf_DD0.predict(xdd6_0)
xdd6_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg6_1).T))
zdd6_1=clf_DD1.predict(xdd6_1)
zdd6=np.zeros((t1-t2))
#ZERO PRE-AD?
zdd6[loct0]=zdd6_0
zdd6[loct1]=zdd6_1
#use dd?
xad6_0=data0[:,8:21]
xad6_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad6_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd6_0).T))
#xad6_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd6_1).T))
zad6_0=clf_AD0.predict(xad6_0)
zad6_1=clf_AD1.predict(xad6_1)
zad6[loct0]=zad6_0
zad6[loct1]=zad6_1

#7
t1=p6
t2=p7
numpy_array7=numpy_array[t2:t1,:]
yad7=numpy_array7[:,21]
ydd7=numpy_array7[:,20]
zad7=np.zeros((t1-t2))
cltype=numpy_array7[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array7[loct0,:]
data0=data0[0,:,:]
data1=numpy_array7[loct1,:]
data1=data1[0,:,:]

apt7=numpy_array7[:,2]
ladg7=numpy_array7[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in ladg7:
    ladg7[i-1]=max(zad6[i-1]-max(apt7[i-1]/15,0)+2.75,0)
ladg7_0=ladg7[loct0]
ladg7_1=ladg7[loct1]

xdd7_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg7_0).T))
zdd7_0=clf_DD0.predict(xdd7_0)
xdd7_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg7_1).T))
zdd7_1=clf_DD1.predict(xdd7_1)
zdd7=np.zeros((t1-t2))
#ZERO PRE-AD?
zdd7[loct0]=zdd7_0
zdd7[loct1]=zdd7_1
#use dd?
xad7_0=data0[:,8:21]
xad7_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad7_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd7_0).T))
#xad7_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd7_1).T))
zad7_0=clf_AD0.predict(xad7_0)
zad7_1=clf_AD1.predict(xad7_1)
zad7[loct0]=zad7_0
zad7[loct1]=zad7_1

#8
t1=p7
t2=p8
numpy_array8=numpy_array[t2:t1,:]
yad8=numpy_array8[:,21]
ydd8=numpy_array8[:,20]
zad8=np.zeros((t1-t2))
cltype=numpy_array8[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array8[loct0,:]
data0=data0[0,:,:]
data1=numpy_array8[loct1,:]
data1=data1[0,:,:]

apt8=numpy_array8[:,2]
ladg8=numpy_array8[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in ladg8:
    ladg8[i-1]=max(zad7[i-1]-max(apt8[i-1]/15,0)+2.75,0)
ladg8_0=ladg8[loct0]
ladg8_1=ladg8[loct1]

xdd8_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg8_0).T))
zdd8_0=clf_DD0.predict(xdd8_0)
xdd8_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg8_1).T))
zdd8_1=clf_DD1.predict(xdd8_1)
zdd8=np.zeros((t1-t2))
#ZERO PRE-AD?
zdd8[loct0]=zdd8_0
zdd8[loct1]=zdd8_1
#use dd?
xad8_0=data0[:,8:21]
xad8_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad8_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd8_0).T))
#xad8_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd8_1).T))
zad8_0=clf_AD0.predict(xad8_0)
zad8_1=clf_AD1.predict(xad8_1)
zad8[loct0]=zad8_0
zad8[loct1]=zad8_1

#9
t1=p8
t2=p9
numpy_array9=numpy_array[t2:t1,:]
yad9=numpy_array9[:,21]
ydd9=numpy_array9[:,20]
zad9=np.zeros((t1-t2))
cltype=numpy_array9[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array9[loct0,:]
data0=data0[0,:,:]
data1=numpy_array9[loct1,:]
data1=data1[0,:,:]

apt9=numpy_array9[:,2]
ladg9=numpy_array9[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in ladg9:
    ladg9[i-1]=max(zad8[i-1]-max(apt9[i-1]/15,0)+2.75,0)
ladg9_0=ladg9[loct0]
ladg9_1=ladg9[loct1]

xdd9_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg9_0).T))
zdd9_0=clf_DD0.predict(xdd9_0)
xdd9_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg9_1).T))
zdd9_1=clf_DD1.predict(xdd9_1)
zdd9=np.zeros((t1-t2))
#ZERO PRE-AD?
zdd9[loct0]=zdd9_0
zdd9[loct1]=zdd9_1
#use dd?
xad9_0=data0[:,8:21]
xad9_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad9_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd9_0).T))
#xad9_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd9_1).T))
zad9_0=clf_AD0.predict(xad9_0)
zad9_1=clf_AD1.predict(xad9_1)
zad9[loct0]=zad9_0
zad9[loct1]=zad9_1

#10
t1=p9
numpy_array10=numpy_array[:t1,:]
yad10=numpy_array10[:,21]
ydd10=numpy_array10[:,20]
zad10=np.zeros((t1-t2))
cltype=numpy_array10[:,0]
loct0=np.where(cltype==0)
loct1=np.where(cltype==1)
data0=numpy_array10[loct0,:]
data0=data0[0,:,:]
data1=numpy_array10[loct1,:]
data1=data1[0,:,:]

apt10=numpy_array10[:,2]
ladg10=numpy_array10[:,22]
bas0=data0[:,7:20]
#use comput ld?
for i in ladg10:
    ladg10[i-1]=max(zad9[i-1]-max(apt10[i-1]/15,0)+2.75,0)
#ladg10_0=ladg10[loct0]
ladg10_1=ladg10[loct1]

#xdd10_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg10_0).T))
#zdd10_0=clf_DD0.predict(xdd10_0)
xdd10_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg10_1).T))
zdd10_1=clf_DD1.predict(xdd10_1)
zdd10=np.zeros((19))
#ZERO PRE-AD?
#zdd10[loct0]=zdd10_0
zdd10=zdd10_1
#use dd?
#xad10_0=data0[:,8:21]
xad10_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
#xad10_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd10_0).T))
#xad10_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd10_1).T))
#zad10_0=clf_AD0.predict(xad10_0)
zad10_1=clf_AD1.predict(xad10_1)
#zad10[loct0]=zad10_0
zad10=zad10_1

zad1_10=np.vstack((np.atleast_2d(zad1).T,np.atleast_2d(zad2).T,np.atleast_2d(zad3).T,np.atleast_2d(zad4).T,np.atleast_2d(zad5).T,np.atleast_2d(zad6).T,np.atleast_2d(zad7).T,np.atleast_2d(zad8).T,np.atleast_2d(zad9).T,np.atleast_2d(zad10).T))
yad1_10=np.vstack((np.atleast_2d(yad1).T,np.atleast_2d(yad2).T,np.atleast_2d(yad3).T,np.atleast_2d(yad4).T,np.atleast_2d(yad5).T,np.atleast_2d(yad6).T,np.atleast_2d(yad7).T,np.atleast_2d(yad8).T,np.atleast_2d(yad9).T,np.atleast_2d(yad10).T))

dfy = pd.DataFrame(yad1_10)
dfz = pd.DataFrame(zad1_10)

##$
#t1=p3
#t2=p4
#numpy_array$=numpy_array[t2:t1,:]
#yad$=numpy_array$[:,21]
#ydd$=numpy_array$[:,20]
#zad$=np.zeros((t1-t2))
#cltype=numpy_array$[:,0]
#loct0=np.where(cltype==0)
#loct1=np.where(cltype==1)
#data0=numpy_array$[loct0,:]
#data0=data0[0,:,:]
#data1=numpy_array$[loct1,:]
#data1=data1[0,:,:]
#
#apt$=numpy_array$[:,2]
#ladg$=numpy_array$[:,22]
#bas0=data0[:,7:20]
##use comput ld?
#for i in apt$:
#    ladg$[i-1]=max(zad&&&&[i-1]-max(apt$[i-1]/15,0)+2.75,0)
#ladg$_0=ladg$[loct0]
#ladg$_1=ladg$[loct1]
#
#xdd$_0=np.hstack((np.atleast_2d(data0[:,3]).T,np.atleast_2d(data0[:,5]).T,bas0,np.atleast_2d(ladg$_0).T))
#zdd$_0=clf_DD0.predict(xdd$_0)
#xdd$_1=np.hstack((data1[:,4:10],np.atleast_2d(ladg$_1).T))
#zdd$_1=clf_DD1.predict(xdd$_1)
#zdd$=np.zeros((t1-t2))
##ZERO PRE-AD?
#zdd$[loct0]=zdd$_0
#zdd$[loct1]=zdd$_1
##use dd?
#xad$_0=data0[:,8:21]
#xad$_1=np.hstack((data1[:,4:10],np.atleast_2d(data1[:,20]).T))
##xad$_0=np.hstack((data0[:,8:20],np.atleast_2d(zdd$_0).T))
##xad$_1=np.hstack((data0[:,4:10],np.atleast_2d(zdd$_1).T))
#zad$_0=clf_AD0.predict(xad$_0)
#zad$_1=clf_AD1.predict(xad$_1)
#zad$[loct0]=zad$_0
#zad$[loct1]=zad$_1

##3
#t1=p2
#t2=p3
#apt3=numpy_array[t2:t1,1]
#ladg3=numpy_array[t2:t1,10]
#bas3=numpy_array[t2:t1,2:8]
##use comput ld?
#for i in apt3:
#    ladg3[i-1]=max(zad2[i-1]-max(apt3[i-1]/15,0)+2.75,0)
#
#xdd3=np.hstack((bas3,np.atleast_2d(ladg3).T))
#ydd3=numpy_array[t2:t1,8]
#zdd3=clf_DD.predict(xdd3)
##ZERO PRE-AD?
##zdd3=np.zeros((t1-t2))
##use dd?
#xad3=numpy_array[t2:t1,2:9]
##xad3=np.hstack((bas3,np.atleast_2d(zdd3).T))
#yad3=numpy_array[t2:t1,8]
#zad3=clf_AD.predict(xad3)
#
##4
#t1=p3
#t2=p4
#apt4=numpy_array[t2:t1,1]
#ladg4=numpy_array[t2:t1,10]
#bas4=numpy_array[t2:t1,2:8]
##use comput ld?
#for i in apt4:
#    ladg4[i-1]=max(zad3[i-1]-max(apt4[i-1]/15,0)+2.75,0)
#
#xdd4=np.hstack((bas4,np.atleast_2d(ladg4).T))
#ydd4=numpy_array[t2:t1,8]
#zdd4=clf_DD.predict(xdd4)
##ZERO PRE-AD?
##zdd4=np.zeros((t1-t2))
##use dd?
#xad4=numpy_array[t2:t1,2:9]
##xad4=np.hstack((bas4,np.atleast_2d(zdd4).T))
#yad4=numpy_array[t2:t1,8]
#zad4=clf_AD.predict(xad4)
#
##5
#t1=p4
#t2=p5
#apt5=numpy_array[t2:t1,1]
#ladg5=numpy_array[t2:t1,10]
#bas5=numpy_array[t2:t1,2:8]
##use comput ld?
#for i in apt5:
#    ladg5[i-1]=max(zad4[i-1]-max(apt5[i-1]/15,0)+2.75,0)
#
#xdd5=np.hstack((bas5,np.atleast_2d(ladg5).T))
#ydd5=numpy_array[t2:t1,8]
#zdd5=clf_DD.predict(xdd5)
##ZERO PRE-AD?
##zdd5=np.zeros((t1-t2))
##use dd?
#xad5=numpy_array[t2:t1,2:9]
##xad5=np.hstack((bas5,np.atleast_2d(zdd5).T))
#yad5=numpy_array[t2:t1,8]
#zad5=clf_AD.predict(xad5)
#
##6
#t1=p5
#t2=p6
#apt6=numpy_array[t2:t1,1]
#ladg6=numpy_array[t2:t1,10]
#bas6=numpy_array[t2:t1,2:8]
##use comput ld?
#for i in ladg6:
#    ladg6[i-1]=max(zad5[i-1]-max(apt6[i-1]/15,0)+2.75,0)
#
#xdd6=np.hstack((bas6,np.atleast_2d(ladg6).T))
#ydd6=numpy_array[t2:t1,8]
#zdd6=clf_DD.predict(xdd6)
##ZERO PRE-AD?
##zdd6=np.zeros((t1-t2))
##use dd?
#xad6=numpy_array[t2:t1,2:9]
##xad6=np.hstack((bas6,np.atleast_2d(zdd6).T))
#yad6=numpy_array[t2:t1,8]
#zad6=clf_AD.predict(xad6)
#
##7
#t1=p6
#t2=p7
#apt7=numpy_array[t2:t1,1]
#ladg7=numpy_array[t2:t1,10]
#bas7=numpy_array[t2:t1,2:8]
##use comput ld?
#for i in ladg7:
#    ladg7[i-1]=max(zad6[i-1]-max(apt7[i-1]/15,0)+2.75,0)
#
#xdd7=np.hstack((bas7,np.atleast_2d(ladg7).T))
#ydd7=numpy_array[t2:t1,8]
#zdd7=clf_DD.predict(xdd7)
##ZERO PRE-AD?
##zdd7=np.zeros((t1-t2))
##use dd?
#xad7=numpy_array[t2:t1,2:9]
##xad7=np.hstack((bas7,np.atleast_2d(zdd7).T))
#yad7=numpy_array[t2:t1,8]
#zad7=clf_AD.predict(xad7)
#
##8
#t1=p7
#t2=p8
#apt8=numpy_array[t2:t1,1]
#ladg8=numpy_array[t2:t1,10]
#bas8=numpy_array[t2:t1,2:8]
##use comput ld?
#for i in ladg8:
#    ladg8[i-1]=max(zad7[i-1]-max(apt8[i-1]/15,0)+2.75,0)
#
#xdd8=np.hstack((bas8,np.atleast_2d(ladg8).T))
#ydd8=numpy_array[t2:t1,8]
#zdd8=clf_DD.predict(xdd8)
##ZERO PRE-AD?
##zdd8=np.zeros((t1-t2))
##use dd?
#xad8=numpy_array[t2:t1,2:9]
##xad8=np.hstack((bas8,np.atleast_2d(zdd8).T))
#yad8=numpy_array[t2:t1,8]
#zad8=clf_AD.predict(xad8)
#
##9
#t1=p8
#t2=p9
#apt9=numpy_array[t2:t1,1]
#ladg9=numpy_array[t2:t1,10]
#bas9=numpy_array[t2:t1,2:8]
##use comput ld?
#for i in ladg9:
#    ladg9[i-1]=max(zad8[i-1]-max(apt9[i-1]/15,0)+2.75,0)
#
#xdd9=np.hstack((bas9,np.atleast_2d(ladg9).T))
#ydd9=numpy_array[t2:t1,8]
#zdd9=clf_DD.predict(xdd9)
##ZERO PRE-AD?
##zdd9=np.zeros((t1-t2))
##use dd?
#xad9=numpy_array[t2:t1,2:9]
##xad9=np.hstack((bas9,np.atleast_2d(zdd9).T))
#yad9=numpy_array[t2:t1,8]
#zad9=clf_AD.predict(xad9)
#
##10
#t1=p9
#
#apt10=numpy_array[:t1,1]
#ladg10=numpy_array[:t1,10]
#bas10=numpy_array[:t1,2:8]
##use comput ld?
#for i in ladg10:
#    ladg10[i-1]=max(zad9[i-1]-max(apt10[i-1]/15,0)+2.75,0)
#
#xdd10=np.hstack((bas10,np.atleast_2d(ladg10).T))
#ydd10=numpy_array[:t1,8]
#zdd10=clf_DD.predict(xdd10)
##ZERO PRE-AD?
##zdd10=np.zeros((t1))
##use dd?
##xad10=numpy_array[:t1,2:9]
#xad10=np.hstack((bas10,np.atleast_2d(zdd10).T))
#yad10=numpy_array[:t1,8]
#zad10=clf_AD.predict(xad10)
#

#xp=linspace(1,29,29)
#fig = plt.figure()
#plt.bar(xp,clf.feature_importances_,0.4,color="gree")
#plt.xlable("features")
#plt.ylable("importance")
#plt.title("importance")

#plt.show()
#plt.savefig("importance.jpg")
