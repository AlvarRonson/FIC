from model import FeatureImportanceClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.data.data_stream import DataStream
from skmultiflow.lazy import KNNClassifier
from utils.data_preprocesing import read_kdd_data_multilable, read_data_arff, read_data_csv
import matplotlib as mpl

mpl.use('WX')

# 1.a Load and preprocessing data
"""
Data source: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
"""
#data, X, y = read_kdd_data_multilable('./data/kddcup.data_10_percent_corrected.csv')

# 1.b Load and preprocessing data
"""
Data source: https://github.com/alipsgh/data_streams
"""
#data, X, y = read_data_arff('stagger_w_50_n_0.1_103.arff')
#data, X, y = read_data_arff('./data/led_w_500_n_0.1_104.arff')

# 1.c Load and preprocessing data
"""
Data source: https://github.com/scikit-multiflow/streaming-datasets
"""
data, X, y = read_data_csv('streaming-datasets-master/elec.csv')
#data, X, y = read_data_csv('streaming-datasets-master/airlines.csv')
#data, X, y = read_data_csv('streaming-datasets-master/agr_a.csv')
#data, X, y = read_data_csv('streaming-datasets-master/covtype.csv')


# stream = DataStream(X[:50000], y[:50000])
stream = DataStream(X, y)
stream.prepare_for_use()

# 2a. Models initialization
nb = NaiveBayes()
ht = HoeffdingTreeClassifier()
aw = AccuracyWeightedEnsembleClassifier()
dw = DynamicWeightedMajorityClassifier()
ob = OnlineBoostingClassifier()
oz = OzaBaggingClassifier()
knn = KNNClassifier()

# 2b. Inicialization of FIC model for comparsion tests
fic = FeatureImportanceClassifier(base_estimators=[nb, ht, knn], period=200, threshold=0.2)


# 2c. Inicialization of FIC models for parameter testing
# fic1 = FeatureImportanceClassifier(base_estimators=[nb, ht, knn], period=100, threshold=0.2)
# fic2 = FeatureImportanceClassifier(base_estimators=[nb, ht, knn], period=500, threshold=0.4)
# fic3 = FeatureImportanceClassifier(base_estimators=[nb, ht, knn], period=1000, threshold=0.2)


# 3. Evalution settings
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=1000,
                                batch_size=1,
                                max_samples=1000000,
                                n_wait=1000,
                                metrics=['accuracy', 'f1', 'precision', 'recall', 'running_time', 'model_size'])
                                #metrics=['accuracy'])


# 4. Example of evaluation DDCW compared with other models

evaluator.evaluate(stream=stream, model=[nb, ht, fic], model_names=['NaiveBayes', 'HoeffdingTree', 'FeatureImportanceClassifier'])
#evaluator.evaluate(stream=stream, model=[aw, dw, fic], model_names=['AccurancyWeightedEnsemble', 'DynamicWeightedMajority','FeatureImportanceClassifier'])
#evaluator.evaluate(stream=stream, model=[ob, oz, fic], model_names=['OnlineBoosting', 'OzaBagging','DiversifiedDynamicClassWeighted'])
#evaluator.evaluate(stream=stream, model=[fic, aw, dw, ob, oz, nb], model_names=['FIC', 'AWE','DWM', 'OB', 'OZB', 'NB'])


# 5. Examples for comparing progress on FIC with different model parameters
# evaluator.evaluate(stream=stream, model=[fic1, fic2, fic2], model_names=['FIC-1', 'FIC-2','FI-3'])


# 6. Ger custom measurements
print(fic.custom_measurements)

import pickle
pickle.dump(evaluator, open("evaluatorobject", "wb"))

