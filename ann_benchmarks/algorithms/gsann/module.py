from ..base.module import BaseANN


class GSann(BaseANN):
    def __init__(self, metric, args):
        self._metric = metric
        self._alg = args['alg']
        self.name = "gsann alg={}".format(self._alg)

    def fit(self, X):
        self._dim = X.shape[1]
        self._card = X.shape[0]
        print("dim={}, card={}".format(self._dim, self._card))
        print("done!")

    def set_query_arguments(self, s):
        self._s = s

    def query(self, q, k):
        return [i for i in range(k)]
