from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

def scores_drawer(met_name, models_scores, final_scores, figsize=(10, 7)):
    """makes a plot with every model scores and ensemble voting score"""
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(models_scores))]
    plt.figure(figsize=figsize)
    plt.title(met_name)
    for (mod_name, scr), color in zip(models_scores.items(), colors):
        plt.plot(scr, label='{} : {}'.format(mod_name, np.mean(scr)),
                 color=color)
    plt.plot(final_scores, linewidth=3, label='Voting', color='black')
    plt.legend(loc='best')

def smooth(pred_list):
    """voting function"""
    result = np.array([np.array(pred) for pred in pred_list])
    result = np.sum(result, axis=0)
    return result / len(pred_list)

class voting_cv():
    # TODO saving results with param maps and comments
    def __init__(self, X, l_train, models, voting):
        """models: [('name', model instance), ...]
        voting: function expecting list of prediction as arg
        X: train data
        l_train: train labels
        """
        self.X = X
        self.models = models
        self.voting = voting
        self.l_train = l_train

    def cv(self, l_val=None,
            metrics=[('accuracy', accuracy_score)], cv=8,
            shuffle=True, random_state=10, draw=True, verbose=True):
        """l_val: labels for validation, if none: same as l_train
        metrics: metrics lists [('name', metric_callable), ...]
        cv: number of folds
        verbose: print score on each iter
        """

        if l_val is None:
            l_val = self.l_train

        kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        scores = {met_name:[] for met_name, _ in metrics}
        models_scores = {met_name:{mod_name:[] for mod_name, _ in self.models} for met_name, _ in metrics}

        split_i = 0
        for train_index, test_index in kf.split(self.X):
            # Dataset split
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train = self.l_train[train_index]
            y_test = l_val[test_index]

            # fit and predict
            vot, predictions = self.fit_predict(X_test, X_train, y_train)

            # Validation
            if verbose:
                print('split # {}'.format(split_i))
            for met_name, metric in metrics:
                for (mod_name, _), pred in zip(self.models, predictions):
                    models_scores[met_name][mod_name].append(
                                        metric(y_pred=pred, y_true=y_test))

            # Final Validation
                cur_metric = metric(y_pred=vot, y_true=y_test)
                scores[met_name].append(cur_metric)

                if verbose:
                    print('metric: {}, score = {}, models scores std = {}'\
                        .format(met_name, cur_metric,
                        np.std([scr[split_i] for _, scr in models_scores[met_name].items()])))

            split_i += 1

        if verbose:
            for met_name, _ in metrics:
                print(met_name)
                scr = scores[met_name]
                print(scr)
                print(np.mean(scr), np.std(scr))
                print()

        if draw:
            for met_name, _ in metrics:
                mod_names = [name for name, _ in self.models]
                scores_drawer(met_name, models_scores[met_name], scores[met_name], figsize=(10, 7))

        return scores

    def fit_predict(self, X_test, X_train=None, y_train=None):
        if X_train is None:
            X_train = self.X
        if y_train is None:
            y_train = self.l_train

        predictions = []
        for _, model in self.models:
            cur_model = model.fit(X_train, y_train)
            predictions.append(cur_model.predict(X_test))

        return self.voting(predictions), predictions

    def get_folds(self):
        pass

    def save_report(self, comments):
        pass
