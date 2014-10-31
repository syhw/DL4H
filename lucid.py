import numpy
import cPickle
from dnn import add_fit_and_score, DropoutNet, RegularizedNet, train_models

if __name__ == "__main__":
    add_fit_and_score(DropoutNet)
    add_fit_and_score(RegularizedNet)
    from sklearn.preprocessing import LabelEncoder
    import joblib
    ((X_train, y_train), (X_dev, y_dev), (X_test, y_test), lb) = joblib.load(
            "LUCID_words.joblib")
    nwords = len(lb.classes_)
    print "building the model..."
    train_models(X_train, y_train, X_test, y_test, X_train.shape[1],
                 nwords, x_dev=X_dev, y_dev=y_dev,
                 numpy_rng=numpy.random.RandomState(123),
                 svms=False, nb=False, deepnn=True, use_dropout=False, n_epochs=1000,
                 verbose=True, plot=True, name='_lucid_words_dnn_ReLUs_L2')
    train_models(X_train, y_train, X_test, y_test, X_train.shape[1],
                 nwords, x_dev=X_dev, y_dev=y_dev,
                 numpy_rng=numpy.random.RandomState(123),
                 svms=False, nb=False, deepnn=True, use_dropout=True, n_epochs=1000,
                 verbose=True, plot=True, name='_lucid_words_dnn_dropout_ReLUs')


