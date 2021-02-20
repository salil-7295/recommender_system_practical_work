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

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def recommendation_res(k: int, pred_res: list, movies: pd.DataFrame):
    res = []
    for item in pred_res:
        if item[4]['was_impossible'] == False:
            res.append([item[0], item[1], item[3]])
    res = pd.DataFrame(res, columns=['userId', 'movieId', 'pred_rating'])
    res = res.sort_values(by='pred_rating', ascending=False).reset_index(drop=True)
    res = res[:k]
    movies['movieId'] = movies['movieId'].astype('str')
    res = pd.merge(res[['movieId', 'pred_rating']], movies, how='left', on='movieId')

    return res[['movieId', 'title', 'genres', 'pred_rating']]


def get_user_recommendation(model_path: str, user_id: int, k: int, movie_path: str) -> pd.DataFrame:
    # load the model
    model = pickle.load(open(loaded_model, 'rb'))
    # generate the data for prediction
    test, df_movies = get_predict_data(user_id, movie_path)
    # make the predictions
    pred = model.test(test)
    # get recommendation result
    res = recommendation_res(k=k, pred_res=pred, movies=df_movies)
    return res

if __name__ == '__main__':
    data = get_data(from_surprise=False)
    res =get_user_recommendation(model_path='NMF.model', user_id=1,k=10,movie_path='../data/movielens/ml-small/movies.csv')
    print(res)