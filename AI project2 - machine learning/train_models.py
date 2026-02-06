from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_and_evaluate_models(train_features, test_features, train_labels, test_labels):
    # --- Logistic Regression ---
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(train_features, train_labels)
    logistic_predictions = logistic_regression_model = logistic_model.predict(test_features)
    
    # Calculate Metrics
    logistic_acc = accuracy_score(test_labels, logistic_predictions)
    logistic_f1 = f1_score(test_labels, logistic_predictions, average='weighted')
    
    # --- Decision Tree ---
    tree_model = DecisionTreeClassifier(max_depth=20) 
    tree_model.fit(train_features, train_labels)
    tree_predictions = tree_model.predict(test_features)
    
    # Calculate Metrics
    tree_acc = accuracy_score(test_labels, tree_predictions)
    tree_f1 = f1_score(test_labels, tree_predictions, average='weighted')
    
    # --- PRINTING TO COMPILER 
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(test_labels, logistic_predictions))
    
    print("\n" + "="*50)
    print("DECISION TREE CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(test_labels, tree_predictions))
    print("="*50 + "\n")
    
    return logistic_model, tree_model, logistic_acc, tree_acc, logistic_f1, tree_f1