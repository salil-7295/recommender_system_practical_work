from pathlib import Path
from surprise import Reader
from surprise.dataset import DatasetAutoFolds, Dataset
from surprise import accuracy



from surprise.trainset import Trainset
from  surprise.prediction_algorithms.algo_base import AlgoBase


def load_ratings_from_file() -> DatasetAutoFolds:
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    ratings = Dataset.load_from_file('ratings.csv', reader)
    return ratings


def get_data(from_surprise : bool = True) -> DatasetAutoFolds:
    data = load_ratings_from_surprise() if from_surprise else load_ratings_from_file()
    return data

def get_trained_model(model_class: AlgoBase, model_kwargs: dict, train_set: Trainset) -> AlgoBase:
    model = model_class(sim_options = model_kwargs)
    model.fit(train_set)
    return model

def get_trained_model_svd(model_class: AlgoBase, train_set: Trainset) -> AlgoBase:
    model = model_class()
    model.fit(train_set)
    return model

def get_trained_model_svdpp(model_class: AlgoBase, train_set: Trainset) -> AlgoBase:
    model = model_class()
    model.fit(train_set)
    return model

def get_trained_model_nmf(model_class: AlgoBase, train_set: Trainset) -> AlgoBase:
    model = model_class()
    model.fit(train_set)
    return model

def evaluate_model(model: AlgoBase, test_set: [(int, int, float)]) -> dict:
    predictions = model.test(test_set)
    metrics_dict = {}
    metrics_dict['RMSE'] = accuracy.rmse(predictions, verbose=False)
    metrics_dict['MAE'] = accuracy.rmse(predictions, verbose=False)
    return metrics_dict

def train_and_evalute_model_pipeline(model_class: AlgoBase, model_kwargs: dict = {},
                                     from_surprise: bool = False,
                                     test_size: float = 0.2) -> (AlgoBase, dict):
    data = get_data(from_surprise)
    train_set, test_set = train_test_split(data, test_size, random_state=42)
    print(str(model_class).split('.')[-1].strip('>')[:-1])
    
    if str(model_class).split('.')[-1].strip('>')[:-1] == 'SVD':
        model = get_trained_model_svd(model_class, train_set)
    elif str(model_class).split('.')[-1].strip('>')[:-1] == 'SVDpp':
        model = get_trained_model_svdpp(model_class, train_set)
    elif str(model_class).split('.')[-1].strip('>')[:-1] == 'NMF':
        model = get_trained_model_nmf(model_class, train_set)     
    else:
        model = get_trained_model(model_class, model_kwargs, train_set)

    metrics_dict = evaluate_model(model, test_set)
    return model, metrics_dict

