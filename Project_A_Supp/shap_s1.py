import scipy.special
import numpy as np
import itertools
import random


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def func_model(model, x, y_predicted):
    X=np.expand_dims(x,2)
    y = model(X)
    return y[y_predicted]


def cal_shap(model, x,shapley_point, M, sp_size, MC, y_predicted):  
    #model for pretrained CNN model, x for input image
    #M for vector size (number of shapley values)
    #MC for monte carlo times, sp_size for super pixel size
    #shapley_point for which point we are calculating shapley value
    weights=np.zeros(M[0]*M[1],np.int)
    binom_M=np.zeros(M[0]*M[1],np.int)
    shapley_out = 0;
    for i in range(M[0]*M[1]):
        #X = np.zeros((MC,M+1))
        #X[:,-1] = 1
        #V = np.zeros((MC,M-1))
        binom_M[i]=(scipy.special.binom(M[0]*M[1]-1,i))
        if(binom_M[i] > MC):
            weights[i] = MC 
        else:
            weights[i] = binom_M[i]
        shape_v = list(x.shape)
        #print(shape_v)
        shape_v.insert(0,weights[i])
        #print(shape_v)
        V_plus  = np.zeros(tuple(shape_v))
        V_minus = np.zeros(tuple(shape_v))
        #print("V_minus shape", V_minus.shape)
        if(binom_M[i]<=MC):  #all
            j=0
            #print("i=",i,list(itertools.combinations(range(M-1),i)))
            for s in list(itertools.combinations(range(M[0]*M[1]-1),i)):   #eliminate i
                s1=list(s)
                s2=list()
                for i0 in range(len(s1)):
                    if(s1[i0] >= shapley_point[1]*M[1]+shapley_point[0]):
                        s1[i0] = s1[i0] + 1
                    sl_row=s1[i0]//M[1]
                    #print("s1[i0]",s1[i0])
                    #print("M[1]",M[1])
                    s1_col=s1[i0]%M[1]
                    s1_row=s1[i0]-s1_col
                    #print("s1_row",s1_row)
                    #print("s1_col",s1_col)
                    for u1 in range(sp_size[0]):
                        for u2 in range(sp_size[1]):
                            s2.append([u1+sp_size[0]*s1_row,u2+sp_size[1]*s1_col])
                s2=np.array(s2,np.int32)
                #print("s1",s1)
                #print("s2",s2)
                if(s2!=[]):
                    idx1=s2[:,0]
                    idx2=s2[:,1]
                    V_minus[j,idx1,idx2] = x[idx1,idx2]
                #print("V_minus",V_minus)
                #print("old s2",s2)
                for u1 in range(sp_size[0]):
                    for u2 in range(sp_size[1]):
                        if(s2.ndim==1):
                            s2=np.append(s2,[u1+shapley_point[0]*sp_size[0],u2+shapley_point[1]*sp_size[1]],axis=0)
                            s2=np.expand_dims(s2,axis=0)
                        else:
                            s2=np.append(s2,[[u1+shapley_point[0]*sp_size[0],u2+shapley_point[1]*sp_size[1]]],axis=0)
                        #if(s2.ndim==1):
                        #    s2=np.expand_dims(s2,axis=0)
                #print("new s2",s2)

                #print("s2",s2)
                idx1=s2[:,0]
                idx2=s2[:,1]
                V_plus[j,idx1,idx2] = x[idx1,idx2]
                #print("V_plus",V_plus)
                j = j+1

        else:
            for j in range(MC):
                s1 = list(random_combination(range(M[0]*M[1]-1),i))   #
                s2 = list()
                for i0 in range(len(s1)):
                    if(s1[i0] >= shapley_point[1]*M[1]+shapley_point[0]):
                        s1[i0] = s1[i0] + 1
                    s1_row=s1[i0]//M[1]
                    s1_col=s1[i0]%M[1]
                    for u1 in range(sp_size[0]):
                        for u2 in range(sp_size[1]):
                            s2.append([u1+sp_size[0]*s1_row,u2+sp_size[1]*s1_col])
                s2=np.array(s2)
                if(s2!=[]):
                    idx1=s2[:,0]
                    idx2=s2[:,1]
                    #print("s2:",s2)
                    V_minus[j,idx1,idx2] = x[idx1,idx2]
                #print("V_minus",V_minus)
                #print(sp_size)
                for u1 in range(sp_size[0]):
                    for u2 in range(sp_size[1]):
                        #print("old s2",s2)
                        if(s2.ndim==1):
                            s2=np.append(s2,[u1+shapley_point[0]*sp_size[0],u2+shapley_point[1]*sp_size[1]],axis=0)
                            s2=np.expand_dims(s2,axis=0)
                        else:
                            s2=np.append(s2,[[u1+shapley_point[0]*sp_size[0],u2+shapley_point[1]*sp_size[1]]],axis=0)
                        #print("new s2",s2)
                idx1=s2[:,0]
                idx2=s2[:,1]
                V_plus[j,idx1,idx2] = x[idx1,idx2]
                #print("V_plus",V_plus)


        #print("V_minus",V_minus)
        #print("V_plus",V_plus)
        #print(V_plus.shape)
        #y_plus  = func_model(model, V_plus,  y_predicted)
        #y_minus = func_model(model, V_minus, y_predicted)
        if(V_plus.ndim==3):
            V_plus = np.expand_dims(V_plus,3)
        y_tmp1 = model(V_plus)
        y_plus = y_tmp1[:,y_predicted]

        if(V_minus.ndim==3):
            V_minus = np.expand_dims(V_minus,3)
        y_tmp2 = model(V_minus)
        #print("y_tmp2",y_tmp2)
        y_minus= y_tmp2[:,y_predicted]

        shapley_out += sum(y_plus-y_minus)/weights[i]/(M[0]*M[1]-1);
        #print("y_plus:",y_plus)
        #print("y_minus:",y_minus)
        #print("shap_out",shapley_out)
        #print("V_minus",V_minus[0,:,0,0].T)
        #print(V_minus)
        #print("V_plus")
        #print("V_plus",V_plus[0,:,0,0].T)     
    #print(weights)
    return shapley_out

def shap_user_defined(x,model,sp_size=[2,1],MC=1000):
    size_x = list(x[0].shape)
    #print("x.shape",x.shape)   #(n,40,1)
    M = [int(a/b) for a, b in zip(size_x, sp_size)]
    shapley=np.zeros(size_x)
    #print(shapley)
    #print("M",M)  

    for m in range(x.shape[0]): # for 1 image 
        for i1 in range(M[0]):
            for i2 in range(M[1]):
                y_pre=model.predict(x[m:m+1])
                predicted_y=np.argmax(y_pre)
                #print(M)
                tmp1=cal_shap(model,x[m],[i1,i2],M,sp_size,MC,predicted_y)
                # xm:(40,1) [i1,i2] = [0,0] to [19,19]  
                #tmp1=cal_shap(model,x[m],i,M,sp_size,MC,predicted_y)
                for j1 in range(sp_size[0]):
                    for j2 in range(sp_size[1]):
                        shapley[i1*sp_size[0]+j1,i2*sp_size[1]+j2] = tmp1
    return shapley
    
#def f(x):
#    a = x[0,0] + x[0,1]*2 + x[1,0]*2 + x[1,1]*1;
#    return a
#
#M = [2,2]
#sp_size = 1 #2
#MC = 2
#x = [[1,2],[3,4]]
#phi = cal_shap(f, x,0, M,sp_size,MC)
#phi = point_shap(f, x,1, M,1,8)
#phi = point_shap(f, x,2, M,8)
#phi = point_shap(f, x,3, M,8)
#phi = point_shap(f, x,4, M,8)
#base_value = phi[-1]
#shap_values = phi[:-1]

#print("  reference =", reference)
#print("          x =", x)
#print("shap_values =", shap_values)
#print(" base_value =", base_value)
#print("   sum(phi) =", np.sum(phi))
#print("       f(x) =", f(x))
#print("random_combination",random_combination(range(8),2)) 
#print("random_combination",random_combination(range(20),2)) 

