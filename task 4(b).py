import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import RandomizedSearchCV
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt



def main():
    # Set random seed to ensure reproducible runs
    RSEED = 50

    df = pd.read_csv('data.csv', skiprows=3)

    bins = [7, 16, 24, 28, 32, 37, 100]
    label = [1, 2, 3, 4, 5, 6]
    df['binned'] = pd.cut(df['GA'], bins, labels=label, include_lowest=True)
    # print(df)
    del df['GA']
    del df['ID']

    new = df.groupby(['binned'])

    # ROC curve
    def evaluate_model(predictions, probs, train_predictions, train_probs):
        """Compare machine learning model to baseline performance.
        Computes statistics and shows ROC curve."""

        baseline = {}

        baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])
        baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])
        baseline['roc'] = 0.5

        results = {}

        results['recall'] = recall_score(y_test, predictions)
        results['precision'] = precision_score(y_test, predictions)
        results['roc'] = roc_auc_score(y_test, probs)

        train_results = {}
        train_results['recall'] = recall_score(y_train, train_predictions)
        train_results['precision'] = precision_score(y_train, train_predictions)
        train_results['roc'] = roc_auc_score(y_train, train_probs)

        for metric in ['recall', 'precision', 'roc']:
            print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric],2)} Train: {round(train_results[metric], 2)}')

        # Calculate false positive rates and true positive rates
        base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
        model_fpr, model_tpr, _ = roc_curve(y_test, probs)
        print('Model_fpr:', model_fpr)

        plt.figure(figsize=(8, 6))
        plt.rcParams['font.size'] = 16
        # Plot both curves
        plt.plot(base_fpr, base_tpr, color='b', label='baseline')
        plt.plot(model_fpr, model_tpr, color='r', label='model')
        plt.legend()
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate');
        plt.title('ROC Curves')
        plt.show()
        #plt.close()

    # Confusion Matrix
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Oranges):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.figure(figsize=(6, 6))
        #fig = plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size=15)
        plt.colorbar(aspect=4)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size=10)
        plt.yticks(tick_marks, classes, size=10)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        # Labeling the plot
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size=10)
        plt.xlabel('Predicted label', size=10)
        plt.show()
        #plt.close()

    for i,n in new:
        X = n.iloc[:, 1:-1]
        # print(X)
        Y = n.iloc[:, [0]]
        print(Y)

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=RSEED)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Hyperparameter grid
        param_grid = {
            'n_estimators': np.linspace(10, 200).astype(int),
            'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
            'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
            'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        }

        # Estimator for use in random search
        estimator = RandomForestClassifier(random_state=RSEED)

        # Create the random search model
        rs = RandomizedSearchCV(estimator, param_grid, n_jobs=-1,
                                scoring='roc_auc', cv=3,
                                n_iter=20, verbose=1, random_state=RSEED)

        # Fit
        rs.fit(X_train, y_train)

        rs.best_params_

        best_model = rs.best_estimator_

        train_rf_predictions = best_model.predict(X_train)
        train_rf_probs = best_model.predict_proba(X_train)[:, 1]

        rf_predictions = best_model.predict(X_test)
        rf_probs = best_model.predict_proba(X_test)[:, 1]

        n_nodes = []
        max_depths = []

        for ind_tree in best_model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')

        evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)

        cm = confusion_matrix(y_test, rf_predictions)
        plot_confusion_matrix(cm, classes=['LatePE = 1', 'LatePE = 0'],
                              title='Health Confusion Matrix')


if __name__ == "__main__":
    main()