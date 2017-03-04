import numpy as np

def reformat(y,num_ent):

    res = np.zeros((y.size,num_ent),dtype=np.int)
    for i in range(0,y.size):
        res[i,int(y[i]-1)] = 1

    return res
    
def get_data(is_train=True):
    if is_train:
        file_type = 'train'
    else:
        file_type = 'test'
        
    x_file = 'datathonData/UCI HAR Dataset/%s/X_%s.txt' %(file_type,file_type)
    y_file = 'datathonData/UCI HAR Dataset/%s/Y_%s.txt' %(file_type,file_type)
    subject_file = 'datathonData/UCI HAR Dataset/%s/subject_%s.txt' %(file_type,file_type)
    
    x = np.loadtxt(x_file)

    y = np.loadtxt(y_file)
    subjects = np.loadtxt(subject_file)
    
    return (x,y,subjects)
    
#    inertial_files = ['datathonData/UCI HAR Dataset/%s/Inertial Signals/body_acc_x_%s.txt',\
#                      'datathonData/UCI HAR Dataset/%s/Inertial Signals/body_acc_y_%s.txt',\
#                      'datathonData/UCI HAR Dataset/%s/Inertial Signals/body_acc_z_%s.txt',\
#                      'datathonData/UCI HAR Dataset/%s/Inertial Signals/body_gyro_x_%s.txt',\
#                      'datathonData/UCI HAR Dataset/%s/Inertial Signals/body_gyro_y_%s.txt',\
#                      'datathonData/UCI HAR Dataset/%s/Inertial Signals/body_gyro_z_%s.txt',\
#                      'datathonData/UCI HAR Dataset/%s/Inertial Signals/total_acc_x_%s.txt',\
#                      'datathonData/UCI HAR Dataset/%s/Inertial Signals/total_acc_y_%s.txt',\
#                      'datathonData/UCI HAR Dataset/%s/Inertial Signals/total_acc_z_%s.txt']
#    inerts = np.zeros((7352,128,len(inertial_files)))
#
#    for i in range(0,len(inertial_files)):
#        tmp= np.loadtxt(inertial_files[i] %(file_type,file_type))
#        inerts[:,:,i]  = tmp
#    
#    return(x,y,subjects,inerts)