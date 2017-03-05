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
    
    print avg_accs[0,:]
    
    num_train_all = (1,2,5,10,15)
    plt.figure()
    plt.plot(num_train_all, avg_accs[0,:])
    
    labels = ('walking','walking_up','walking_down','sitting','standing','laying')
    
    for i in range(0,6):
        plt.plot(num_train_all,avg_accs[i,:],label=labels[i])
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Observations')
    plt.title('User Learning Conditioned on Activity: %s' %(algo_name))
    plt.show()

SVM.main()
plt.close("all")
plot_res('res_rbf_svm','RBF-SVM')
plot_res('res_rbf_svm','Deep net',6)
plot_res('res_rfc','RFC')