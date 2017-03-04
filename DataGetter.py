import numpy as np

def get_train_data():
    x_train_file = 'datathonData/UCI_HAR_Dataset/train/X_train.txt'
    y_train_file = 'datathonData/UCI_HAR_Dataset/train/Y_train.txt'
    subject_train_file = 'datathonData/UCI_HAR_Dataset/train/subject_train.txt'
    
    x = np.loadtxt(x_train_file)
    y = np.loadtxt(y_train_file)
    subjects = np.loadtxt(subject_train_file)
    
    inertial_files = ['datathonData/UCI_HAR_Dataset/train/Inertial Signals/body_acc_x_train.txt', \
                      'datathonData/UCI_HAR_Dataset/train/Inertial Signals/body_acc_y_train.txt', \
                      'datathonData/UCI_HAR_Dataset/train/Inertial Signals/body_acc_z_train.txt', \
                      'datathonData/UCI_HAR_Dataset/train/Inertial Signals/body_gyro_x_train.txt', \
                      'datathonData/UCI_HAR_Dataset/train/Inertial Signals/body_gyro_y_train.txt', \
                      'datathonData/UCI_HAR_Dataset/train/Inertial Signals/body_gyro_z_train.txt', \
                      'datathonData/UCI_HAR_Dataset/train/Inertial Signals/total_acc_x_train.txt', \
                      'datathonData/UCI_HAR_Dataset/train/Inertial Signals/total_acc_y_train.txt', \
                      'datathonData/UCI_HAR_Dataset/train/Inertial Signals/total_acc_z_train.txt']
    inerts = np.zeros((7352,128,len(inertial_files)))
    print(inerts.shape)
    for i in range(0,len(inertial_files)):
        tmp= np.loadtxt(inertial_files[i])
        inerts[:,:,i]  = tmp
    
    return(x,y,subjects,inerts)