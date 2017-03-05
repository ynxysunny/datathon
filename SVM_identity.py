import DataGetter
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

ACTIVITY = 8
TRAIN_SIZE = 25

def join_data(x_train,y_train,s_train,x_test,y_test,s_test):
    x = np.concatenate((x_train,x_test),axis=0)
    y = np.concatenate((y_train,y_test),axis=0)   
    s = np.concatenate((s_train,s_test),axis=0)
    return(x,y,s)

def split_data(s,num_train,num_test):
    num_class = np.unique(s).size
    
    train_idx_all = np.zeros((0))
    test_idx_all = np.zeros((0))
    
    for i in range(1,num_class+1):
        idx = np.argwhere(s==i)
        idx = np.random.permutation(idx)[:,0]

        train_idx_all = np.concatenate((train_idx_all,idx[0:num_train]),axis=0)
        test_idx_all = np.concatenate((test_idx_all,idx[num_train:num_train+num_test]),axis=0)

    train_idx_all = train_idx_all.astype(int)
    test_idx_all = test_idx_all.astype(int)
    
    return (train_idx_all,test_idx_all)

def main():
    (x_train, y_train, s_train)=DataGetter.get_data(is_train=True)
    (x_test, y_test, s_test)=DataGetter.get_data(is_train=False)

    (x, y, s) = join_data(x_train,y_train,s_train,x_test,y_test,s_test)

    idx = np.argwhere(y == ACTIVITY)
    s = s[idx].astype(int)
    TEST_SIZE = s.shape[0]- TRAIN_SIZE

    (idx_train_ACTIVITY, idx_test_ACTIVITY) = split_data(s,TRAIN_SIZE,TEST_SIZE)

    id_train_ACTIVITY = s[idx_train_ACTIVITY][:,0]
    x_train_ACTIVITY=x[idx_train_ACTIVITY, :]
    
    print x_train_ACTIVITY.shape
    
    clf=svm.LinearSVC()
    clf.fit(x_train_ACTIVITY, id_train_ACTIVITY)
    dec_train=clf.decision_function(x_train_ACTIVITY)
    train_results=np.argmax(dec_train, axis =1) +1
    train_results.astype(float)
    confusion_matrix_train = confusion_matrix(train_results, id_train_ACTIVITY)

    x_test_ACTIVITY=x[idx_test_ACTIVITY, :]
    id_test_ACTIVITY = s[idx_test_ACTIVITY][:,0]
    dec_test=clf.decision_function(x_test_ACTIVITY)
    test_results=np.argmax(dec_test, axis =1) +1
    test_results.astype(float)
    confusion_matrix_test = confusion_matrix(test_results, id_test_ACTIVITY)
    accuracy = np.trace(confusion_matrix_test)/np.sum(confusion_matrix_test)

    print (confusion_matrix_train)
    print (confusion_matrix_test)
    print (accuracy)

if __name__ == '__main__':
    main()
