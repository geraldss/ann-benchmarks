import os
import subprocess

from ..base.module import BaseANN


class GSann(BaseANN):
    def __init__(self, metric, alg, r0, r1):
        self._metric = metric
        self._alg = alg
        self._r0 = r0
        self._r1 = r1
        self.name = "gsann dist={} alg={} r0={} r1={}".format(metric, alg, r0, r1)

    def fit(self, X):
        dim = X.shape[1]
        card = X.shape[0]
        print("dim={}, card={}".format(dim, card))

        execpath = "/home/gsann/gsann"
        execsize = os.path.getsize(execpath)
        print("execsize={}".format(execsize))
        datapath = "/tmp/datapipe"

        subprocess.Popen([
            execpath,
            "-dist", self._metric,
            "-alg", str(self._alg),
            "-r0", str(self._r0),
            "-r1", str(self._r1),
            "-dim", str(dim),
            "-card", str(card),
            "-data", datapath
            ])

        with open(datapath, "wb") as file:
            file.write(X.data)

        print("fit() done")

    def set_query_arguments(self, s):
        self._s = s

    def query(self, q, k):
        return [i for i in range(k)]
