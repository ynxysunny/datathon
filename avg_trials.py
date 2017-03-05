import numpy as np
import matplotlib.pyplot as plt
import SVM

def plot_res(folder,algo_name,num_trials=10):
    
    avg_accs = []
    for trial in range(0,num_trials):
        accs = np.load('%s/trial%d.npy' %(folder,trial))
        if trial == 0:
            avg_accs = accs
        else:
            avg_accs = avg_accs + accs
            
    avg_accs /= num_trials
    
    num_train_all = (1,2,5,10,15)
    plt.figure()
    
    labels = ('walking','walking_up','walking_down','sitting','standing','laying','all')
    print(len(labels))
    
    for j in range(0,7):
        plt.plot(num_train_all,avg_accs[j,:],label=labels[j])
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Observations')
    plt.title('User Learning Conditioned on Activity: %s' %(algo_name))
    plt.yticks([0.1*x for x in range(0,11)])
    plt.show()

#SVM.main()
plt.close("all")
plot_res('res_rbf_svm','RBF-SVM')
#plot_res('res_rbf_svm','Deep net',6)
plot_res('res_rfc','Random Forests')