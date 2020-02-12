import mmh3
import numpy as np


class BloomFilter(object):
    def __init__(self, inputArray, probability, cap=2):
        inputSize = len(inputArray)

        self.falsePos = probability
        self.size = self.getSize(inputSize, probability)
        self.hashCount = self.getHashCount(self.size, inputSize)
        self.mask = np.zeros(self.size, int)
        self.cap = cap

        self.addAll(inputArray)

    def addAll(self, inputArray):
        for i in range(len(inputArray)):
            for j in range(self.hashCount):
                digit = mmh3.hash(inputArray[i], j) % self.size
                if self.mask[digit] < self.cap - 1:
                    self.mask[digit] += 1

    def intMask(self):
        return int("".join(str(i) for i in self.mask), self.cap)

    # for testing purpose
    def check(self, item):
        for i in range(self.hashCount):
            digit = mmh3.hash(item, i) % self.size
            if self.mask[digit] == False:
                return False
        return True

    def getSize(self, n, p):
        m = - (n * np.log(p)) / (np.log(2) ** 2)
        return int(m)

    def getHashCount(self, m, n):
        k = (m / n) * np.log(2)
        return int(k)
