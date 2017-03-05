import DataGetter
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

def run_svm(x_train,y_train,x_test,y_test):
    clf = svm.NuSVC(kernel='rbf')
    #clf = svm.LinearSVC()
    
    clf.fit(x_train, y_train)    
    
    #dec_test = clf.decision_function(x_test)
    #test_results = np.argmax(dec_test, axis=1) +1
    
    test_results = clf.predict(x_test)
    test_results.astype(float)
    confusion_matrix_test = confusion_matrix(test_results, y_test)
    
    acc=float(confusion_matrix_test.trace())/float(np.sum(confusion_matrix_test))
    
    print np.sum(confusion_matrix_test)
    #print("test accuracy: %f" %acc)
    return (acc,confusion_matrix_test)

def main():
    
    (x_train, y_train, subjects_train)=DataGetter.get_data(is_train=True)
    (x_test, y_test, subjects_test)=DataGetter.get_data(is_train=False)

    clf = svm.LinearSVC()
    #clf = svm.NuSVC(kernel='rbf')
    
    clf.fit(x_train, y_train)

    #dec_test = clf.decision_function(x_test)
    #test_results = np.argmax(dec_test, axis=1) +1
    
    test_results = clf.predict(x_test)
    test_results.astype(float)
    confusion_matrix_test = confusion_matrix(test_results, y_test)

    #print (confusion_matrix_test)
    
    acc=float(confusion_matrix_test.trace())/float(np.sum(confusion_matrix_test))
    print("test accuracy for classifying activities with linear SVM: %f" %acc)

