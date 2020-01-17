from bloomfilter import BloomFilter

import numpy as np

a = np.array(['cat', 'dog', 'wow'])
bloomfilter = BloomFilter(a, 0.01)
print(bloomfilter.intMask())

print(bloomfilter.check('hh'))
