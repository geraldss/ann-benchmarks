import numpy
import os
import subprocess

from ..base.module import BaseANN


class GSann(BaseANN):
    def __init__(self, metric, alg, r, p, s, sv, g, gv, cp, cs, v):
        self._metric = metric
        self._alg = alg
        self._r = r
        self._p = p
        self._s = s
        self._sv = sv
        self._g = g
        self._gv = gv
        self._cp = cp
        self._cs = cs
        self._v = v
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
            "-r", str(self._r),
            "-p", str(self._p),
            "-s", str(self._s),
            "-sv", str(self._sv),
            "-g", str(self._g),
            "-gv", str(self._gv),
            "-cp", str(self._cp),
            "-cs", str(self._cs),
            "-v", str(self._v),
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
        return f"gsann r={self._r} p={self._p} s={self._s} sv={self._sv} g={self._g} gv={self._gv} cp={self._cp} cs={self._cs} v={self._v}"
