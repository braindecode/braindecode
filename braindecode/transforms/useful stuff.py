    y_pred = clf.predict(test_dataset)
    y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
    acc = accuracy_score(y_test, y_pred)
    return(model_accuracy)
