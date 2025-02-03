import numpy
import os
import subprocess

from ..base.module import BaseANN


class GSann(BaseANN):
    def __init__(self, metric, alg, r0, r1):
        self._metric = metric
        self._alg = alg
        self._r0 = r0
        self._r1 = r1
        self._s = numpy.float32(0)
        self._qsig = None
        self._rsig = None

    def fit(self, X: numpy.array) -> None:
        card = X.shape[0]
        dim = X.shape[1]
        print(f"card={card}, dim={dim}")

        execpath = "/home/gsann/gsann"
        execsize = os.path.getsize(execpath)
        print(f"execsize={execsize}")

        data = "/home/gsann/data"
        queries = "/home/gsann/queries"
        results = "/home/gsann/results"

        X.astype("float32").tofile(data)

        subprocess.Popen([
            execpath,
            "-dist", self._metric,
            "-alg", str(self._alg),
            "-r0", str(self._r0),
            "-r1", str(self._r1),
            "-card", str(card),
            "-dim", str(dim),
            "-data", data,
            "-queries", queries,
            "-results", results,
            "-count", str(10),
            ],
                         stderr=subprocess.STDOUT)

        # block until data loaded
        with open(data + ".sig", "rb") as pipe:
            pipe.read()

        os.remove(data)

        self._queries = queries
        self._results = results
        self.res = None

        print("fit() done")

    def set_query_arguments(self, s):
        self._s = numpy.float32(s)

    def batch_query(self, X: numpy.array, n: int) -> None:
        X.astype("float32").tofile(self._queries)

        # signal queries written
        with open(self._queries + ".sig", "wb") as pipe:
            pipe.write(bytes([0]))

        # block until results
        with open(self._results + ".sig", "rb") as pipe:
            pipe.read()

        nq = X.shape[0]
        with open(self._results, "rb") as file:
            buf = file.read(nq*n*4)

        arr = numpy.frombuffer(buf, dtype=numpy.int32)
        arr.shape = (nq, n)
        self.res = arr

        os.remove(self._queries)

    def query(self, q: numpy.array, n: int) -> numpy.array:
        raise Exception("Use --batch")

        if not self._qsig:
            self._qsig = open(self._queries + ".sig", "wb")
            print("opened queries.sig")

        # signal queries written
        self._qsig.write(q.data)

        if not self._rsig:
            self._rsig = open(self._results + ".sig", "rb")
            print("opened results.sig")

        # block until results
        buf = self._rsig.read(n*4)
        arr = numpy.frombuffer(buf, dtype=numpy.int32)
        return arr

    def __str__(self):
        return f"gsann dist={self._metric} alg={self._alg} r0={self._r0} r1={self._r1}"
