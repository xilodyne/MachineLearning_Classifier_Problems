def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
        
    ### your code goes here!

    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ###from naive_bayes import GaussianNB
    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### return the fit classifier
    return clf
