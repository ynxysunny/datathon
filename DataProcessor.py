import numpy as np

def join_data(x_train,y_train,s_train,x_test,y_test,s_test):
    x = np.concatenate((x_train,x_test),axis=0)
    y = np.concatenate((y_train,y_test),axis=0)   
    s = np.concatenate((s_train,s_test),axis=0)
    return(x,y,s)

def split_data(s,num_train,num_test):
    num_class = np.unique(s).size
    
    train_idx_all = np.zeros((0,1))
    test_idx_all = np.zeros((0,1))
    
    for i in range(0,num_class):
        idx = np.random.permutation(np.argwhere(s==i))
        
        train_idx_all = np.concatenate((train_idx_all,idx[0:num_train]),axis=0)
        test_idx_all = np.concatenate((test_idx_all,idx[num_train:num_train+num_test]),axis=0)

    train_idx_all = train_idx_all.astype(int)
    test_idx_all = test_idx_all.astype(int)
    
    return (train_idx_all[:,0],test_idx_all[:,0])