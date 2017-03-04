import numpy as np
import DataGetter
import tensorflow as tf

(x_train,y_train,subjects_train) = DataGetter.get_data(is_train=True)
(x_test,y_test,subjects_test) = DataGetter.get_data(is_train=False)