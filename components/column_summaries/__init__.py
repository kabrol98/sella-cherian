import numpy as np
from bloomfilter import BloomFilter

a = np.array(['cat', 'dog', 'wow'])
bloomfilter = BloomFilter(a, 0.01)
print(bloomfilter.intMask())

print(bloomfilter.check('hh'))
