from sklearn.ensemble import RandomForestClassifier

def run_rfc(x_train,y_train,x_test,y_test):
    rfc = RandomForestClassifier(n_estimators = 10,max_depth = 10)
    rfc.fit(x_train,y_train)
    return rfc.score(x_test,y_test)