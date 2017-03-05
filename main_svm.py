import DataGetter
import DataProcessor
import SVM
import numpy as np

#num_train_all = (1,2,5,10,15)
num_train_all = (1,2,5,10,15)
num_test = 5
n_trials = 10

n_act = 6;

(x_train,y_train,s_train) = DataGetter.get_data(is_train=True)
(x_test,y_test,s_test) = DataGetter.get_data(is_train=False)

#make 0-index
s_train -= 1
s_test -= 1
y_train -= 1
y_test -=1

x_all, y_all, s_all = DataProcessor.join_data(x_train,y_train,s_train,x_test,y_test,s_test)

for trial in range(0,n_trials):
    accs = np.zeros((n_act,len(num_train_all)))
    for nt in range(0,len(num_train_all)):
        num_train = num_train_all[nt]
        for i in range(0,n_act):
            
            idx = DataProcessor.filter_data(y_all,i) 
            x_use = x_all[idx,:]
            s_use = s_all[idx]
            y_use = y_all[idx]
            
            idx_train,idx_test = DataProcessor.split_data(s_use,num_train,num_test)
            
            x_use_train = x_use[idx_train,:]
            x_use_test = x_use[idx_test,:]
            
            s_use_train = s_use[idx_train]
            s_use_test = s_use[idx_test]
            
            y_use_train = y_use[idx_train]
            y_use_test = y_use[idx_test] 
            
            #s_use_train = DataGetter.reformat(s_use_train,np.unique(s_use_train).size)
            #s_use_test = DataGetter.reformat(s_use_test,np.unique(s_use_test).size)
                
            #acc = Neural.main(x_use_train,s_use_train,x_use_test,s_use_test)
            (acc,conf) = SVM.run_svm(x_use_train,s_use_train,x_use_test,s_use_test)
            
            print("here is acc for trial/nt/i: %d/%d/%d (%f)" %(trial,nt,i,acc))
            
            accs[i,nt] = acc
            
    np.save('res_rbf_svm/trial%d' %(trial),accs)