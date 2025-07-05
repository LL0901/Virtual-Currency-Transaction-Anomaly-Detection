import hashlib
from datetime import datetime
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class Block:
    """
            prev_hash：      父区块哈希值
            data：           区块内容
            timestamp：      区块创建时间
            hash：           区块哈希值
    """

    def __init__(self, data, prev_hash):
        # 将传入的父区块哈希值和数据保存到变量中
        self.prev_hash = prev_hash
        self.data = data

        # 获得当前的时间
        # self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 计算区块哈希值
        # 获取哈希对象
        message = hashlib.sha256()

        message.update(str(self.prev_hash).encode('utf-8'))
        message.update(str(self.prev_hash).encode('utf-8'))
        message.update(str(self.prev_hash).encode('utf-8'))

        self.hash = message.hexdigest()


a = Block(1, 1.6)


# print(a.hash)


class BlockChain:
    """
        区块链结构体
            blocks：         包含区块的列表
    """

    def __init__(self):
        self.blocks = []

    def add_block(self, block):
        """
        添加区块
        """
        self.blocks.append(block)


def gen_data():
    timestamp = int(time.time() * 1000)
    random_part = random.randint(1000, 9999)
    unique_number = timestamp * 10000 + random_part
    # print(unique_number)
    # 新建区块
    num_in = random.randint(1, 10)
    num_out = random.randint(1, 10)
    genesis_block = Block(data="模拟区块", prev_hash=unique_number)
    new_block1 = Block(data=f"{num_in} bit from **", prev_hash=genesis_block.hash)
    new_block2 = Block(data=f"{num_out} bit to **", prev_hash=genesis_block.hash)

    # 新建一个区块链对象
    blockChain = BlockChain()
    # 将刚才新建的区块加入区块链
    blockChain.add_block(genesis_block)
    blockChain.add_block(new_block1)
    blockChain.add_block(new_block2)

    content = "区块链包含区块个数为：%d\n" % len(blockChain.blocks)
    blockHeight = 0
    for block in blockChain.blocks:
        content += f"本区块高度为：{blockHeight}\n"
        content += f"父区块哈希：{block.prev_hash}\n"
        content += f"区块内容：{block.data}\n"
        content += f"区块哈希：{block.hash}\n"
        blockHeight += 1
    return content, num_in, num_out


# b = gen_data()


def draw(x):
    for i in transaction_datas:
        if x in i:
            in_ = i[x][1]
            ot_ = i[x][2]
            fit = np.polyfit(in_, ot_, 5)
            pere = np.poly1d(fit)
            in_np = np.array(in_)
            ot_np = np.array(ot_)
            r = ot_np / in_np
            r_ot_pred = pere(in_) / in_
            bias = r - r_ot_pred
            indexes = np.arange(len(ot_))
            indx_max = np.argmax(np.abs(bias))
            y_max = r[indx_max]
            plt.scatter(indexes, r)
            plt.plot(indexes, r_ot_pred, 'r')
            plt.scatter([indx_max], [y_max], color='red', zorder=5)
            plt.text(indx_max, y_max, f'x={indx_max}', ha='right', va='bottom',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            plt.ylabel('sell')
            plt.xlabel('transaction')
            show = key_data[:10] + '当天异常交易信息：\n'
            show += i[x][0][indx_max]
            print(show)
            plt.legend()
            plt.show()


df = pd.read_excel('bit_v.xlsx')
# print(df)
transaction_datas = []
# print(type(df.loc[3, 'open']))
trans_rs = []

for i in range(len(df)):
    date = df.loc[i, 'date']
    value = []
    x_in_f = []
    y_ot_f = []
    feature = []
    feature.append(df.loc[i, 'open'])
    feature.append(df.loc[i, 'high'])
    feature.append(df.loc[i, 'low'])
    feature.append(df.loc[i, 'close'])

    for p in range(0, 100):
        value_tran, x, y = gen_data()
        value.append(value_tran)
        x_in_f.append(x)
        y_ot_f.append(y)
    x_fit = np.array(x_in_f)
    y_fit = np.array(y_ot_f)
    fit = np.polyfit(x_fit, y_fit, 4)
    pere = np.poly1d(fit)
    r = pere(1)
    trans_rs.append(r)
    dict_day = {f'{date}': [value, x_in_f, y_ot_f, feature, r]}
    transaction_datas.append(dict_day)


class zoom(nn.Module):
    def __init__(self, ic=4, oc=256, ratio=8, num=3):
        super().__init__()
        self.ic = ic
        block = []
        for i in range(0, num):
            hc = int(self.ic * ratio)
            block.append(nn.Linear(self.ic, hc))
            block.append(nn.ReLU())
            self.ic = hc

        block.append(nn.Linear(self.ic, oc))
        self.feature = nn.Sequential(*block)
        self.result = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(oc, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.result(x)
        x = torch.sigmoid(x)
        return x


class data_text(Dataset):
    def __init__(self, x, y):
        self.data_tr = x
        self.labels = y
        lens = len(self.data_tr)
        datas = []
        for i in range(lens):
            data = torch.tensor(self.data_tr[i])
            data = data.transpose(-1, 0)
            mean = torch.mean(data)
            std = torch.std(data)
            n_data = (data - mean) / std
            n_data = n_data.float()
            datas.append(n_data)
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        one_data = self.datas[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        label = label.view(-1, 1)
        return one_data, label


data_np = df[['open', 'high', 'low', 'close']].values


def train(mod, train_data):
    ranning_loss = 0.0001
    mod.train()
    for x, y in train_data:
        x, y = x.to(device), y.to(device)
        predictions = mod(x)
        loss = losss_function(predictions, y)
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss.backward()
        # c = 0.1  # 截断阈值
        # nn.utils.clip_grad_value_(model.parameters(), c)
        optimizer.step()
        ranning_loss += loss.item()
    epoch_loss = ranning_loss / len(train_data.dataset) * batch_size
    print(
        'training_loss:', epoch_loss
    )


batch_size = 4
dataset = data_text(data_np, trans_rs)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = zoom()
model.to(device)
losss_function = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 30
ic = 4
oc = 256
train_loss = []
path = './best_model.pt'
torch.save(model.state_dict(), path)
best_model_wts = model.state_dict()

for epoch in range(epochs):
    train(model,dataloader)
# '2013-12-28 00:00:00'
# print(transaction_datas[1])

key_data = '2014-01-01 00:00:00' #change_date
draw(key_data)

"""Random variable generators.

    bytes
    -----
           uniform bytes (values between 0 and 255)

    integers
    --------
           uniform within range

    sequences
    ---------
           pick random element
           pick random sample
           pick weighted random sample
           generate random permutation

    distributions on the real line:
    ------------------------------
           uniform
           triangular
           normal (Gaussian)
           lognormal
           negative exponential
           gamma
           beta
           pareto
           Weibull

    distributions on the circle (angles 0 to 2pi)
    ---------------------------------------------
           circular uniform
           von Mises

General notes on the underlying Mersenne Twister core generator:

* The period is 2**19937-1.
* It is one of the most extensively tested generators in existence.
* The random() method is implemented in C, executes in a single Python step,
  and is, therefore, threadsafe.

"""

# Translated by Guido van Rossum from C source provided by
# Adrian Baddeley.  Adapted by Raymond Hettinger for use with
# the Mersenne Twister  and os.urandom() core generators.

from warnings import warn as _warn
from math import log as _log, exp as _exp, pi as _pi, e as _e, ceil as _ceil
from math import sqrt as _sqrt, acos as _acos, cos as _cos, sin as _sin
from math import tau as TWOPI, floor as _floor, isfinite as _isfinite
from os import urandom as _urandom
from _collections_abc import Set as _Set, Sequence as _Sequence
from operator import index as _index
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import os as _os
import _random

try:
    # hashlib is pretty heavy to load, try lean internal module first
    from _sha512 import sha512 as _sha512
except ImportError:
    # fallback to official implementation
    from hashlib import sha512 as _sha512

__all__ = [
    "Random",
    "SystemRandom",
    "betavariate",
    "choice",
    "choices",
    "expovariate",
    "gammavariate",
    "gauss",
    "getrandbits",
    "getstate",
    "lognormvariate",
    "normalvariate",
    "paretovariate",
    "randbytes",
    "randint",
    "random",
    "randrange",
    "sample",
    "seed",
    "setstate",
    "shuffle",
    "triangular",
    "uniform",
    "vonmisesvariate",
    "weibullvariate",
]

NV_MAGICCONST = 4 * _exp(-0.5) / _sqrt(2.0)
LOG4 = _log(4.0)
SG_MAGICCONST = 1.0 + _log(4.5)
BPF = 53        # Number of bits in a float
RECIP_BPF = 2 ** -BPF
_ONE = 1


class Random(_random.Random):
    """Random number generator base class used by bound module functions.

    Used to instantiate instances of Random to get generators that don't
    share state.

    Class Random can also be subclassed if you want to use a different basic
    generator of your own devising: in that case, override the following
    methods:  random(), seed(), getstate(), and setstate().
    Optionally, implement a getrandbits() method so that randrange()
    can cover arbitrarily large ranges.

    """

    VERSION = 3     # used by getstate/setstate

    def __init__(self, x=None):
        """Initialize an instance.

        Optional argument x controls seeding, as for Random.seed().
        """

        self.seed(x)
        self.gauss_next = None

    def seed(self, a=None, version=2):
        """Initialize internal state from a seed.

        The only supported seed types are None, int, float,
        str, bytes, and bytearray.

        None or no argument seeds from current time or from an operating
        system specific randomness source if available.

        If *a* is an int, all bits are used.

        For version 2 (the default), all of the bits are used if *a* is a str,
        bytes, or bytearray.  For version 1 (provided for reproducing random
        sequences from older versions of Python), the algorithm for str and
        bytes generates a narrower range of seeds.

        """

        if version == 1 and isinstance(a, (str, bytes)):
            a = a.decode('latin-1') if isinstance(a, bytes) else a
            x = ord(a[0]) << 7 if a else 0
            for c in map(ord, a):
                x = ((1000003 * x) ^ c) & 0xFFFFFFFFFFFFFFFF
            x ^= len(a)
            a = -2 if x == -1 else x

        elif version == 2 and isinstance(a, (str, bytes, bytearray)):
            if isinstance(a, str):
                a = a.encode()
            a = int.from_bytes(a + _sha512(a).digest())

        elif not isinstance(a, (type(None), int, float, str, bytes, bytearray)):
            raise TypeError('The only supported seed types are: None,\n'
                            'int, float, str, bytes, and bytearray.')

        super().seed(a)
        self.gauss_next = None

    def getstate(self):
        """Return internal state; can be passed to setstate() later."""
        return self.VERSION, super().getstate(), self.gauss_next

    def setstate(self, state):
        """Restore internal state from object returned by getstate()."""
        version = state[0]
        if version == 3:
            version, internalstate, self.gauss_next = state
            super().setstate(internalstate)
        elif version == 2:
            version, internalstate, self.gauss_next = state
            # In version 2, the state was saved as signed ints, which causes
            #   inconsistencies between 32/64-bit systems. The state is
            #   really unsigned 32-bit ints, so we convert negative ints from
            #   version 2 to positive longs for version 3.
            try:
                internalstate = tuple(x % (2 ** 32) for x in internalstate)
            except ValueError as e:
                raise TypeError from e
            super().setstate(internalstate)
        else:
            raise ValueError("state with version %s passed to "
                             "Random.setstate() of version %s" %
                             (version, self.VERSION))


    ## -------------------------------------------------------
    ## ---- Methods below this point do not need to be overridden or extended
    ## ---- when subclassing for the purpose of using a different core generator.


    ## -------------------- pickle support  -------------------

    # Issue 17489: Since __reduce__ was defined to fix #759889 this is no
    # longer called; we leave it here because it has been here since random was
    # rewritten back in 2001 and why risk breaking something.
    def __getstate__(self):  # for pickle
        return self.getstate()

    def __setstate__(self, state):  # for pickle
        self.setstate(state)

    def __reduce__(self):
        return self.__class__, (), self.getstate()


    ## ---- internal support method for evenly distributed integers ----

    def __init_subclass__(cls, /, **kwargs):
        """Control how subclasses generate random integers.

        The algorithm a subclass can use depends on the random() and/or
        getrandbits() implementation available to it and determines
        whether it can generate random integers from arbitrarily large
        ranges.
        """

        for c in cls.__mro__:
            if '_randbelow' in c.__dict__:
                # just inherit it
                break
            if 'getrandbits' in c.__dict__:
                cls._randbelow = cls._randbelow_with_getrandbits
                break
            if 'random' in c.__dict__:
                cls._randbelow = cls._randbelow_without_getrandbits
                break

    def _randbelow_with_getrandbits(self, n):
        "Return a random int in the range [0,n).  Defined for n > 0."

        getrandbits = self.getrandbits
        k = n.bit_length()  # don't use (n-1) here because n can be 1
        r = getrandbits(k)  # 0 <= r < 2**k
        while r >= n:
            r = getrandbits(k)
        return r

    def _randbelow_without_getrandbits(self, n, maxsize=1<<BPF):
        """Return a random int in the range [0,n).  Defined for n > 0.

        The implementation does not use getrandbits, but only random.
        """

        random = self.random
        if n >= maxsize:
            _warn("Underlying random() generator does not supply \n"
                "enough bits to choose from a population range this large.\n"
                "To remove the range limitation, add a getrandbits() method.")
            return _floor(random() * n)
        rem = maxsize % n
        limit = (maxsize - rem) / maxsize   # int(limit * maxsize) % n == 0
        r = random()
        while r >= limit:
            r = random()
        return _floor(r * maxsize) % n

    _randbelow = _randbelow_with_getrandbits


    ## --------------------------------------------------------
    ## ---- Methods below this point generate custom distributions
    ## ---- based on the methods defined above.  They do not
    ## ---- directly touch the underlying generator and only
    ## ---- access randomness through the methods:  random(),
    ## ---- getrandbits(), or _randbelow().


    ## -------------------- bytes methods ---------------------

    def randbytes(self, n):
        """Generate n random bytes."""
        return self.getrandbits(n * 8).to_bytes(n, 'little')


    ## -------------------- integer methods  -------------------

    def randrange(self, start, stop=None, step=_ONE):
        """Choose a random item from range(stop) or range(start, stop[, step]).

        Roughly equivalent to ``choice(range(start, stop, step))`` but
        supports arbitrarily large ranges and is optimized for common cases.

        """

        # This code is a bit messy to make it fast for the
        # common case while still doing adequate error checking.
        try:
            istart = _index(start)
        except TypeError:
            istart = int(start)
            if istart != start:
                _warn('randrange() will raise TypeError in the future',
                      DeprecationWarning, 2)
                raise ValueError("non-integer arg 1 for randrange()")
            _warn('non-integer arguments to randrange() have been deprecated '
                  'since Python 3.10 and will be removed in a subsequent '
                  'version',
                  DeprecationWarning, 2)
        if stop is None:
            # We don't check for "step != 1" because it hasn't been
            # type checked and converted to an integer yet.
            if step is not _ONE:
                raise TypeError('Missing a non-None stop argument')
            if istart > 0:
                return self._randbelow(istart)
            raise ValueError("empty range for randrange()")

        # stop argument supplied.
        try:
            istop = _index(stop)
        except TypeError:
            istop = int(stop)
            if istop != stop:
                _warn('randrange() will raise TypeError in the future',
                      DeprecationWarning, 2)
                raise ValueError("non-integer stop for randrange()")
            _warn('non-integer arguments to randrange() have been deprecated '
                  'since Python 3.10 and will be removed in a subsequent '
                  'version',
                  DeprecationWarning, 2)
        width = istop - istart
        try:
            istep = _index(step)
        except TypeError:
            istep = int(step)
            if istep != step:
                _warn('randrange() will raise TypeError in the future',
                      DeprecationWarning, 2)
                raise ValueError("non-integer step for randrange()")
            _warn('non-integer arguments to randrange() have been deprecated '
                  'since Python 3.10 and will be removed in a subsequent '
                  'version',
                  DeprecationWarning, 2)
        # Fast path.
        if istep == 1:
            if width > 0:
                return istart + self._randbelow(width)
            raise ValueError("empty range for randrange() (%d, %d, %d)" % (istart, istop, width))

        # Non-unit step argument supplied.
        if istep > 0:
            n = (width + istep - 1) // istep
        elif istep < 0:
            n = (width + istep + 1) // istep
        else:
            raise ValueError("zero step for randrange()")
        if n <= 0:
            raise ValueError("empty range for randrange()")
        return istart + istep * self._randbelow(n)

    def randint(self, a, b):
        """Return random integer in range [a, b], including both end points.
        """

        return self.randrange(a, b+1)


    ## -------------------- sequence methods  -------------------

    def choice(self, seq):
        """Choose a random element from a non-empty sequence."""

        # As an accommodation for NumPy, we don't use "if not seq"
        # because bool(numpy.array()) raises a ValueError.
        if not len(seq):
            raise IndexError('Cannot choose from an empty sequence')
        return seq[self._randbelow(len(seq))]

    def shuffle(self, x):
        """Shuffle list x in place, and return None."""

        randbelow = self._randbelow
        for i in reversed(range(1, len(x))):
            # pick an element in x[:i+1] with which to exchange x[i]
            j = randbelow(i + 1)
            x[i], x[j] = x[j], x[i]

    def sample(self, population, k, *, counts=None):
        """Chooses k unique random elements from a population sequence.

        Returns a new list containing elements from the population while
        leaving the original population unchanged.  The resulting list is
        in selection order so that all sub-slices will also be valid random
        samples.  This allows raffle winners (the sample) to be partitioned
        into grand prize and second place winners (the subslices).

        Members of the population need not be hashable or unique.  If the
        population contains repeats, then each occurrence is a possible
        selection in the sample.

        Repeated elements can be specified one at a time or with the optional
        counts parameter.  For example:

            sample(['red', 'blue'], counts=[4, 2], k=5)

        is equivalent to:

            sample(['red', 'red', 'red', 'red', 'blue', 'blue'], k=5)

        To choose a sample from a range of integers, use range() for the
        population argument.  This is especially fast and space efficient
        for sampling from a large population:

            sample(range(10000000), 60)

        """

        # Sampling without replacement entails tracking either potential
        # selections (the pool) in a list or previous selections in a set.

        # When the number of selections is small compared to the
        # population, then tracking selections is efficient, requiring
        # only a small set and an occasional reselection.  For
        # a larger number of selections, the pool tracking method is
        # preferred since the list takes less space than the
        # set and it doesn't suffer from frequent reselections.

        # The number of calls to _randbelow() is kept at or near k, the
        # theoretical minimum.  This is important because running time
        # is dominated by _randbelow() and because it extracts the
        # least entropy from the underlying random number generators.

        # Memory requirements are kept to the smaller of a k-length
        # set or an n-length list.

        # There are other sampling algorithms that do not require
        # auxiliary memory, but they were rejected because they made
        # too many calls to _randbelow(), making them slower and
        # causing them to eat more entropy than necessary.

        if not isinstance(population, _Sequence):
            raise TypeError("Population must be a sequence.  "
                            "For dicts or sets, use sorted(d).")
        n = len(population)
        if counts is not None:
            cum_counts = list(_accumulate(counts))
            if len(cum_counts) != n:
                raise ValueError('The number of counts does not match the population')
            total = cum_counts.pop()
            if not isinstance(total, int):
                raise TypeError('Counts must be integers')
            if total <= 0:
                raise ValueError('Total of counts must be greater than zero')
            selections = self.sample(range(total), k=k)
            bisect = _bisect
            return [population[bisect(cum_counts, s)] for s in selections]
        randbelow = self._randbelow
        if not 0 <= k <= n:
            raise ValueError("Sample larger than population or is negative")
        result = [None] * k
        setsize = 21        # size of a small set minus size of an empty list
        if k > 5:
            setsize += 4 ** _ceil(_log(k * 3, 4))  # table size for big sets
        if n <= setsize:
            # An n-length list is smaller than a k-length set.
            # Invariant:  non-selected at pool[0 : n-i]
            pool = list(population)
            for i in range(k):
                j = randbelow(n - i)
                result[i] = pool[j]
                pool[j] = pool[n - i - 1]  # move non-selected item into vacancy
        else:
            selected = set()
            selected_add = selected.add
            for i in range(k):
                j = randbelow(n)
                while j in selected:
                    j = randbelow(n)
                selected_add(j)
                result[i] = population[j]
        return result

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        """Return a k sized list of population elements chosen with replacement.

        If the relative weights or cumulative weights are not specified,
        the selections are made with equal probability.

        """
        random = self.random
        n = len(population)
        if cum_weights is None:
            if weights is None:
                floor = _floor
                n += 0.0    # convert to float for a small speed improvement
                return [population[floor(random() * n)] for i in _repeat(None, k)]
            try:
                cum_weights = list(_accumulate(weights))
            except TypeError:
                if not isinstance(weights, int):
                    raise
                k = weights
                raise TypeError(
                    f'The number of choices must be a keyword argument: {k=}'
                ) from None
        elif weights is not None:
            raise TypeError('Cannot specify both weights and cumulative weights')
        if len(cum_weights) != n:
            raise ValueError('The number of weights does not match the population')
        total = cum_weights[-1] + 0.0   # convert to float
        if total <= 0.0:
            raise ValueError('Total of weights must be greater than zero')
        if not _isfinite(total):
            raise ValueError('Total of weights must be finite')
        bisect = _bisect
        hi = n - 1
        return [population[bisect(cum_weights, random() * total, 0, hi)]
                for i in _repeat(None, k)]


    ## -------------------- real-valued distributions  -------------------

    def uniform(self, a, b):
        "Get a random number in the range [a, b) or [a, b] depending on rounding."
        return a + (b - a) * self.random()

    def triangular(self, low=0.0, high=1.0, mode=None):
        """Triangular distribution.

        Continuous distribution bounded by given lower and upper limits,
        and having a given mode value in-between.

        http://en.wikipedia.org/wiki/Triangular_distribution

        """
        u = self.random()
        try:
            c = 0.5 if mode is None else (mode - low) / (high - low)
        except ZeroDivisionError:
            return low
        if u > c:
            u = 1.0 - u
            c = 1.0 - c
            low, high = high, low
        return low + (high - low) * _sqrt(u * c)

    def normalvariate(self, mu=0.0, sigma=1.0):
        """Normal distribution.

        mu is the mean, and sigma is the standard deviation.

        """
        # Uses Kinderman and Monahan method. Reference: Kinderman,
        # A.J. and Monahan, J.F., "Computer generation of random
        # variables using the ratio of uniform deviates", ACM Trans
        # Math Software, 3, (1977), pp257-260.

        random = self.random
        while True:
            u1 = random()
            u2 = 1.0 - random()
            z = NV_MAGICCONST * (u1 - 0.5) / u2
            zz = z * z / 4.0
            if zz <= -_log(u2):
                break
        return mu + z * sigma

    def gauss(self, mu=0.0, sigma=1.0):
        """Gaussian distribution.

        mu is the mean, and sigma is the standard deviation.  This is
        slightly faster than the normalvariate() function.

        Not thread-safe without a lock around calls.

        """
        # When x and y are two variables from [0, 1), uniformly
        # distributed, then
        #
        #    cos(2*pi*x)*sqrt(-2*log(1-y))
        #    sin(2*pi*x)*sqrt(-2*log(1-y))
        #
        # are two *independent* variables with normal distribution
        # (mu = 0, sigma = 1).
        # (Lambert Meertens)
        # (corrected version; bug discovered by Mike Miller, fixed by LM)

        # Multithreading note: When two threads call this function
        # simultaneously, it is possible that they will receive the
        # same return value.  The window is very small though.  To
        # avoid this, you have to use a lock around all calls.  (I
        # didn't want to slow this down in the serial case by using a
        # lock here.)

        random = self.random
        z = self.gauss_next
        self.gauss_next = None
        if z is None:
            x2pi = random() * TWOPI
            g2rad = _sqrt(-2.0 * _log(1.0 - random()))
            z = _cos(x2pi) * g2rad
            self.gauss_next = _sin(x2pi) * g2rad

        return mu + z * sigma

    def lognormvariate(self, mu, sigma):
        """Log normal distribution.

        If you take the natural logarithm of this distribution, you'll get a
        normal distribution with mean mu and standard deviation sigma.
        mu can have any value, and sigma must be greater than zero.

        """
        return _exp(self.normalvariate(mu, sigma))

    def expovariate(self, lambd):
        """Exponential distribution.

        lambd is 1.0 divided by the desired mean.  It should be
        nonzero.  (The parameter would be called "lambda", but that is
        a reserved word in Python.)  Returned values range from 0 to
        positive infinity if lambd is positive, and from negative
        infinity to 0 if lambd is negative.

        """
        # lambd: rate lambd = 1/mean
        # ('lambda' is a Python reserved word)

        # we use 1-random() instead of random() to preclude the
        # possibility of taking the log of zero.
        return -_log(1.0 - self.random()) / lambd

    def vonmisesvariate(self, mu, kappa):
        """Circular data distribution.

        mu is the mean angle, expressed in radians between 0 and 2*pi, and
        kappa is the concentration parameter, which must be greater than or
        equal to zero.  If kappa is equal to zero, this distribution reduces
        to a uniform random angle over the range 0 to 2*pi.

        """
        # Based upon an algorithm published in: Fisher, N.I.,
        # "Statistical Analysis of Circular Data", Cambridge
        # University Press, 1993.

        # Thanks to Magnus Kessler for a correction to the
        # implementation of step 4.

        random = self.random
        if kappa <= 1e-6:
            return TWOPI * random()

        s = 0.5 / kappa
        r = s + _sqrt(1.0 + s * s)

        while True:
            u1 = random()
            z = _cos(_pi * u1)

            d = z / (r + z)
            u2 = random()
            if u2 < 1.0 - d * d or u2 <= (1.0 - d) * _exp(d):
                break

        q = 1.0 / r
        f = (q + z) / (1.0 + q * z)
        u3 = random()
        if u3 > 0.5:
            theta = (mu + _acos(f)) % TWOPI
        else:
            theta = (mu - _acos(f)) % TWOPI

        return theta

    def gammavariate(self, alpha, beta):
        """Gamma distribution.  Not the gamma function!

        Conditions on the parameters are alpha > 0 and beta > 0.

        The probability distribution function is:

                    x ** (alpha - 1) * math.exp(-x / beta)
          pdf(x) =  --------------------------------------
                      math.gamma(alpha) * beta ** alpha

        """
        # alpha > 0, beta > 0, mean is alpha*beta, variance is alpha*beta**2

        # Warning: a few older sources define the gamma distribution in terms
        # of alpha > -1.0
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError('gammavariate: alpha and beta must be > 0.0')

        random = self.random
        if alpha > 1.0:

            # Uses R.C.H. Cheng, "The generation of Gamma
            # variables with non-integral shape parameters",
            # Applied Statistics, (1977), 26, No. 1, p71-74

            ainv = _sqrt(2.0 * alpha - 1.0)
            bbb = alpha - LOG4
            ccc = alpha + ainv

            while True:
                u1 = random()
                if not 1e-7 < u1 < 0.9999999:
                    continue
                u2 = 1.0 - random()
                v = _log(u1 / (1.0 - u1)) / ainv
                x = alpha * _exp(v)
                z = u1 * u1 * u2
                r = bbb + ccc * v - x
                if r + SG_MAGICCONST - 4.5 * z >= 0.0 or r >= _log(z):
                    return x * beta

        elif alpha == 1.0:
            # expovariate(1/beta)
            return -_log(1.0 - random()) * beta

        else:
            # alpha is between 0 and 1 (exclusive)
            # Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle
            while True:
                u = random()
                b = (_e + alpha) / _e
                p = b * u
                if p <= 1.0:
                    x = p ** (1.0 / alpha)
                else:
                    x = -_log((b - p) / alpha)
                u1 = random()
                if p > 1.0:
                    if u1 <= x ** (alpha - 1.0):
                        break
                elif u1 <= _exp(-x):
                    break
            return x * beta

    def betavariate(self, alpha, beta):
        """Beta distribution.

        Conditions on the parameters are alpha > 0 and beta > 0.
        Returned values range between 0 and 1.

        """
        ## See
        ## http://mail.python.org/pipermail/python-bugs-list/2001-January/003752.html
        ## for Ivan Frohne's insightful analysis of why the original implementation:
        ##
        ##    def betavariate(self, alpha, beta):
        ##        # Discrete Event Simulation in C, pp 87-88.
        ##
        ##        y = self.expovariate(alpha)
        ##        z = self.expovariate(1.0/beta)
        ##        return z/(y+z)
        ##
        ## was dead wrong, and how it probably got that way.

        # This version due to Janne Sinkkonen, and matches all the std
        # texts (e.g., Knuth Vol 2 Ed 3 pg 134 "the beta distribution").
        y = self.gammavariate(alpha, 1.0)
        if y:
            return y / (y + self.gammavariate(beta, 1.0))
        return 0.0

    def paretovariate(self, alpha):
        """Pareto distribution.  alpha is the shape parameter."""
        # Jain, pg. 495

        u = 1.0 - self.random()
        return u ** (-1.0 / alpha)

    def weibullvariate(self, alpha, beta):
        """Weibull distribution.

        alpha is the scale parameter and beta is the shape parameter.

        """
        # Jain, pg. 499; bug fix courtesy Bill Arms

        u = 1.0 - self.random()
        return alpha * (-_log(u)) ** (1.0 / beta)


## ------------------------------------------------------------------
## --------------- Operating System Random Source  ------------------


class SystemRandom(Random):
    """Alternate random number generator using sources provided
    by the operating system (such as /dev/urandom on Unix or
    CryptGenRandom on Windows).

     Not available on all systems (see os.urandom() for details).

    """

    def random(self):
        """Get the next random number in the range 0.0 <= X < 1.0."""
        return (int.from_bytes(_urandom(7)) >> 3) * RECIP_BPF

    def getrandbits(self, k):
        """getrandbits(k) -> x.  Generates an int with k random bits."""
        if k < 0:
            raise ValueError('number of bits must be non-negative')
        numbytes = (k + 7) // 8                       # bits / 8 and rounded up
        x = int.from_bytes(_urandom(numbytes))
        return x >> (numbytes * 8 - k)                # trim excess bits

    def randbytes(self, n):
        """Generate n random bytes."""
        # os.urandom(n) fails with ValueError for n < 0
        # and returns an empty bytes string for n == 0.
        return _urandom(n)

    def seed(self, *args, **kwds):
        "Stub method.  Not used for a system random number generator."
        return None

    def _notimplemented(self, *args, **kwds):
        "Method should not be called for a system random number generator."
        raise NotImplementedError('System entropy source does not have state.')
    getstate = setstate = _notimplemented


# ----------------------------------------------------------------------
# Create one instance, seeded from current time, and export its methods
# as module-level functions.  The functions share state across all uses
# (both in the user's code and in the Python libraries), but that's fine
# for most programs and is easier for the casual user than making them
# instantiate their own Random() instance.

_inst = Random()
seed = _inst.seed
random = _inst.random
uniform = _inst.uniform
triangular = _inst.triangular
randint = _inst.randint
choice = _inst.choice
randrange = _inst.randrange
sample = _inst.sample
shuffle = _inst.shuffle
choices = _inst.choices
normalvariate = _inst.normalvariate
lognormvariate = _inst.lognormvariate
expovariate = _inst.expovariate
vonmisesvariate = _inst.vonmisesvariate
gammavariate = _inst.gammavariate
gauss = _inst.gauss
betavariate = _inst.betavariate
paretovariate = _inst.paretovariate
weibullvariate = _inst.weibullvariate
getstate = _inst.getstate
setstate = _inst.setstate
getrandbits = _inst.getrandbits
randbytes = _inst.randbytes


## ------------------------------------------------------
## ----------------- test program -----------------------

def _test_generator(n, func, args):
    from statistics import stdev, fmean as mean
    from time import perf_counter

    t0 = perf_counter()
    data = [func(*args) for i in _repeat(None, n)]
    t1 = perf_counter()

    xbar = mean(data)
    sigma = stdev(data, xbar)
    low = min(data)
    high = max(data)

    print(f'{t1 - t0:.3f} sec, {n} times {func.__name__}')
    print('avg %g, stddev %g, min %g, max %g\n' % (xbar, sigma, low, high))


def _test(N=2000):
    _test_generator(N, random, ())
    _test_generator(N, normalvariate, (0.0, 1.0))
    _test_generator(N, lognormvariate, (0.0, 1.0))
    _test_generator(N, vonmisesvariate, (0.0, 1.0))
    _test_generator(N, gammavariate, (0.01, 1.0))
    _test_generator(N, gammavariate, (0.1, 1.0))
    _test_generator(N, gammavariate, (0.1, 2.0))
    _test_generator(N, gammavariate, (0.5, 1.0))
    _test_generator(N, gammavariate, (0.9, 1.0))
    _test_generator(N, gammavariate, (1.0, 1.0))
    _test_generator(N, gammavariate, (2.0, 1.0))
    _test_generator(N, gammavariate, (20.0, 1.0))
    _test_generator(N, gammavariate, (200.0, 1.0))
    _test_generator(N, gauss, (0.0, 1.0))
    _test_generator(N, betavariate, (3.0, 3.0))
    _test_generator(N, triangular, (0.0, 1.0, 1.0 / 3.0))


## ------------------------------------------------------
## ------------------ fork support  ---------------------

if hasattr(_os, "fork"):
    _os.register_at_fork(after_in_child=_inst.seed)


if __name__ == '__main__':
    _test()

# encoding: utf-8
# module time
# from (built-in)
# by generator 1.147
"""
This module provides various functions to manipulate time values.

There are two standard representations of time.  One is the number
of seconds since the Epoch, in UTC (a.k.a. GMT).  It may be an integer
or a floating point number (to represent fractions of seconds).
The Epoch is system-defined; on Unix, it is generally January 1st, 1970.
The actual value can be retrieved by calling gmtime(0).

The other representation is a tuple of 9 integers giving local time.
The tuple items are:
  year (including century, e.g. 1998)
  month (1-12)
  day (1-31)
  hours (0-23)
  minutes (0-59)
  seconds (0-59)
  weekday (0-6, Monday is 0)
  Julian day (day in the year, 1-366)
  DST (Daylight Savings Time) flag (-1, 0 or 1)
If the DST flag is 0, the time is given in the regular time zone;
if it is 1, the time is given in the DST time zone;
if it is -1, mktime() should guess based on the date and time.
"""
# no imports

# Variables with simple values

altzone = -32400

daylight = 0

timezone = -28800

_STRUCT_TM_ITEMS = 11


# functions

def asctime(p_tuple=None):  # real signature unknown; restored from __doc__
    """
    asctime([tuple]) -> string

    Convert a time tuple to a string, e.g. 'Sat Jun 06 16:26:11 1998'.
    When the time tuple is not present, current time as returned by localtime()
    is used.
    """
    return ""


def ctime(seconds=None):  # known case of time.ctime
    """
    ctime(seconds) -> string

    Convert a time in seconds since the Epoch to a string in local time.
    This is equivalent to asctime(localtime(seconds)). When the time tuple is
    not present, current time as returned by localtime() is used.
    """
    return ""


def get_clock_info(name):  # real signature unknown; restored from __doc__
    """
    get_clock_info(name: str) -> dict

    Get information of the specified clock.
    """
    return {}


def gmtime(seconds=None):  # real signature unknown; restored from __doc__
    """
    gmtime([seconds]) -> (tm_year, tm_mon, tm_mday, tm_hour, tm_min,
                           tm_sec, tm_wday, tm_yday, tm_isdst)

    Convert seconds since the Epoch to a time tuple expressing UTC (a.k.a.
    GMT).  When 'seconds' is not passed in, convert the current time instead.

    If the platform supports the tm_gmtoff and tm_zone, they are available as
    attributes only.
    """
    pass


def localtime(seconds=None):  # real signature unknown; restored from __doc__
    """
    localtime([seconds]) -> (tm_year,tm_mon,tm_mday,tm_hour,tm_min,
                              tm_sec,tm_wday,tm_yday,tm_isdst)

    Convert seconds since the Epoch to a time tuple expressing local time.
    When 'seconds' is not passed in, convert the current time instead.
    """
    pass


def mktime(p_tuple):  # real signature unknown; restored from __doc__
    """
    mktime(tuple) -> floating point number

    Convert a time tuple in local time to seconds since the Epoch.
    Note that mktime(gmtime(0)) will not generally return zero for most
    time zones; instead the returned value will either be equal to that
    of the timezone or altzone attributes on the time module.
    """
    return 0.0


def monotonic():  # real signature unknown; restored from __doc__
    """
    monotonic() -> float

    Monotonic clock, cannot go backward.
    """
    return 0.0


def monotonic_ns():  # real signature unknown; restored from __doc__
    """
    monotonic_ns() -> int

    Monotonic clock, cannot go backward, as nanoseconds.
    """
    return 0


def perf_counter():  # real signature unknown; restored from __doc__
    """
    perf_counter() -> float

    Performance counter for benchmarking.
    """
    return 0.0


def perf_counter_ns():  # real signature unknown; restored from __doc__
    """
    perf_counter_ns() -> int

    Performance counter for benchmarking as nanoseconds.
    """
    return 0


def process_time():  # real signature unknown; restored from __doc__
    """
    process_time() -> float

    Process time for profiling: sum of the kernel and user-space CPU time.
    """
    return 0.0


def process_time_ns(*args, **kwargs):  # real signature unknown
    """
    process_time() -> int

    Process time for profiling as nanoseconds:
    sum of the kernel and user-space CPU time.
    """
    pass


def sleep(seconds):  # real signature unknown; restored from __doc__
    """
    sleep(seconds)

    Delay execution for a given number of seconds.  The argument may be
    a floating point number for subsecond precision.
    """
    pass


def strftime(format, p_tuple=None):  # real signature unknown; restored from __doc__
    """
    strftime(format[, tuple]) -> string

    Convert a time tuple to a string according to a format specification.
    See the library reference manual for formatting codes. When the time tuple
    is not present, current time as returned by localtime() is used.

    Commonly used format codes:

    %Y  Year with century as a decimal number.
    %m  Month as a decimal number [01,12].
    %d  Day of the month as a decimal number [01,31].
    %H  Hour (24-hour clock) as a decimal number [00,23].
    %M  Minute as a decimal number [00,59].
    %S  Second as a decimal number [00,61].
    %z  Time zone offset from UTC.
    %a  Locale's abbreviated weekday name.
    %A  Locale's full weekday name.
    %b  Locale's abbreviated month name.
    %B  Locale's full month name.
    %c  Locale's appropriate date and time representation.
    %I  Hour (12-hour clock) as a decimal number [01,12].
    %p  Locale's equivalent of either AM or PM.

    Other codes may be available on your platform.  See documentation for
    the C library strftime function.
    """
    return ""


def strptime(string, format):  # real signature unknown; restored from __doc__
    """
    strptime(string, format) -> struct_time

    Parse a string to a time tuple according to a format specification.
    See the library reference manual for formatting codes (same as
    strftime()).

    Commonly used format codes:

    %Y  Year with century as a decimal number.
    %m  Month as a decimal number [01,12].
    %d  Day of the month as a decimal number [01,31].
    %H  Hour (24-hour clock) as a decimal number [00,23].
    %M  Minute as a decimal number [00,59].
    %S  Second as a decimal number [00,61].
    %z  Time zone offset from UTC.
    %a  Locale's abbreviated weekday name.
    %A  Locale's full weekday name.
    %b  Locale's abbreviated month name.
    %B  Locale's full month name.
    %c  Locale's appropriate date and time representation.
    %I  Hour (12-hour clock) as a decimal number [01,12].
    %p  Locale's equivalent of either AM or PM.

    Other codes may be available on your platform.  See documentation for
    the C library strftime function.
    """
    return struct_time


def thread_time():  # real signature unknown; restored from __doc__
    """
    thread_time() -> float

    Thread time for profiling: sum of the kernel and user-space CPU time.
    """
    return 0.0


def thread_time_ns(*args, **kwargs):  # real signature unknown
    """
    thread_time() -> int

    Thread time for profiling as nanoseconds:
    sum of the kernel and user-space CPU time.
    """
    pass


def time():  # real signature unknown; restored from __doc__
    """
    time() -> floating point number

    Return the current time in seconds since the Epoch.
    Fractions of a second may be present if the system clock provides them.
    """
    return 0.0


def time_ns():  # real signature unknown; restored from __doc__
    """
    time_ns() -> int

    Return the current time in nanoseconds since the Epoch.
    """
    return 0


# classes

class struct_time(tuple):
    """
    The time value as returned by gmtime(), localtime(), and strptime(), and
     accepted by asctime(), mktime() and strftime().  May be considered as a
     sequence of 9 integers.

     Note that several fields' values are not the same as those defined by
     the C language standard for struct tm.  For example, the value of the
     field tm_year is the actual year, not year - 1900.  See individual
     fields' descriptions for details.
    """

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    def __repr__(self, *args, **kwargs):  # real signature unknown
        """ Return repr(self). """
        pass

    tm_gmtoff = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """offset from UTC in seconds"""

    tm_hour = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """hours, range [0, 23]"""

    tm_isdst = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """1 if summer time is in effect, 0 if not, and -1 if unknown"""

    tm_mday = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """day of month, range [1, 31]"""

    tm_min = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """minutes, range [0, 59]"""

    tm_mon = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """month of year, range [1, 12]"""

    tm_sec = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """seconds, range [0, 61])"""

    tm_wday = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """day of week, range [0, 6], Monday is 0"""

    tm_yday = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """day of year, range [1, 366]"""

    tm_year = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """year, for example, 1993"""

    tm_zone = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """abbreviation of timezone name"""

    n_fields = 11
    n_sequence_fields = 9
    n_unnamed_fields = 0
    __match_args__ = (
        'tm_year',
        'tm_mon',
        'tm_mday',
        'tm_hour',
        'tm_min',
        'tm_sec',
        'tm_wday',
        'tm_yday',
        'tm_isdst',
    )


class __loader__(object):
    """
    Meta path import for built-in modules.

        All methods are either class or static methods to avoid the need to
        instantiate the class.
    """

    def create_module(spec):  # reliably restored by inspect
        """ Create a built-in module """
        pass

    def exec_module(module):  # reliably restored by inspect
        """ Exec a built-in module """
        pass

    @classmethod
    def find_module(cls, *args, **kwargs):  # real signature unknown
        """
        Find the built-in module.

                If 'path' is ever specified then the search is considered a failure.

                This method is deprecated.  Use find_spec() instead.
        """
        pass

    @classmethod
    def find_spec(cls, *args, **kwargs):  # real signature unknown
        pass

    @classmethod
    def get_code(cls, *args, **kwargs):  # real signature unknown
        """ Return None as built-in modules do not have code objects. """
        pass

    @classmethod
    def get_source(cls, *args, **kwargs):  # real signature unknown
        """ Return None as built-in modules do not have source code. """
        pass

    @classmethod
    def is_package(cls, *args, **kwargs):  # real signature unknown
        """ Return False as built-in modules are never packages. """
        pass

    @classmethod
    def load_module(cls, *args, **kwargs):  # real signature unknown
        """
        Load the specified module into sys.modules and return it.

            This method is deprecated.  Use loader.exec_module() instead.
        """
        pass

    def module_repr(module):  # reliably restored by inspect
        """
        Return repr for the module.

                The method is deprecated.  The import machinery does the job itself.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    __weakref__ = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """list of weak references to the object (if defined)"""

    _ORIGIN = 'built-in'
    __dict__ = None  # (!) real value is "mappingproxy({'__module__': '_frozen_importlib', '__doc__': 'Meta path import for built-in modules.\\n\\n    All methods are either class or static methods to avoid the need to\\n    instantiate the class.\\n\\n    ', '_ORIGIN': 'built-in', 'module_repr': <staticmethod(<function BuiltinImporter.module_repr at 0x000001E2B9E228E0>)>, 'find_spec': <classmethod(<function BuiltinImporter.find_spec at 0x000001E2B9E22980>)>, 'find_module': <classmethod(<function BuiltinImporter.find_module at 0x000001E2B9E22A20>)>, 'create_module': <staticmethod(<function BuiltinImporter.create_module at 0x000001E2B9E22AC0>)>, 'exec_module': <staticmethod(<function BuiltinImporter.exec_module at 0x000001E2B9E22B60>)>, 'get_code': <classmethod(<function BuiltinImporter.get_code at 0x000001E2B9E22CA0>)>, 'get_source': <classmethod(<function BuiltinImporter.get_source at 0x000001E2B9E22DE0>)>, 'is_package': <classmethod(<function BuiltinImporter.is_package at 0x000001E2B9E22F20>)>, 'load_module': <classmethod(<function _load_module_shim at 0x000001E2B9E21C60>)>, '__dict__': <attribute '__dict__' of 'BuiltinImporter' objects>, '__weakref__': <attribute '__weakref__' of 'BuiltinImporter' objects>})"


# variables with complex values

tzname = (
    '中国标准时间',
    '中国夏令时',
)

__spec__ = None  # (!) real value is "ModuleSpec(name='time', loader=<class '_frozen_importlib.BuiltinImporter'>, origin='built-in')"

#.  Copyright (C) 2005-2010   Gregory P. Smith (greg@krypto.org)
#  Licensed to PSF under a Contributor Agreement.
#

__doc__ = """hashlib module - A common interface to many hash functions.

new(name, data=b'', **kwargs) - returns a new hash object implementing the
                                given hash function; initializing the hash
                                using the given binary data.

Named constructor functions are also available, these are faster
than using new(name):

md5(), sha1(), sha224(), sha256(), sha384(), sha512(), blake2b(), blake2s(),
sha3_224, sha3_256, sha3_384, sha3_512, shake_128, and shake_256.

More algorithms may be available on your platform but the above are guaranteed
to exist.  See the algorithms_guaranteed and algorithms_available attributes
to find out what algorithm names can be passed to new().

NOTE: If you want the adler32 or crc32 hash functions they are available in
the zlib module.

Choose your hash function wisely.  Some have known collision weaknesses.
sha384 and sha512 will be slow on 32 bit platforms.

Hash objects have these methods:
 - update(data): Update the hash object with the bytes in data. Repeated calls
                 are equivalent to a single call with the concatenation of all
                 the arguments.
 - digest():     Return the digest of the bytes passed to the update() method
                 so far as a bytes object.
 - hexdigest():  Like digest() except the digest is returned as a string
                 of double length, containing only hexadecimal digits.
 - copy():       Return a copy (clone) of the hash object. This can be used to
                 efficiently compute the digests of datas that share a common
                 initial substring.

For example, to obtain the digest of the byte string 'Nobody inspects the
spammish repetition':

    >>> import hashlib
    >>> m = hashlib.md5()
    >>> m.update(b"Nobody inspects")
    >>> m.update(b" the spammish repetition")
    >>> m.digest()
    b'\\xbbd\\x9c\\x83\\xdd\\x1e\\xa5\\xc9\\xd9\\xde\\xc9\\xa1\\x8d\\xf0\\xff\\xe9'

More condensed:

    >>> hashlib.sha224(b"Nobody inspects the spammish repetition").hexdigest()
    'a4337bc45a8fc544c03f52dc550cd6e1e87021bc896588bd79e901e2'

"""

# This tuple and __get_builtin_constructor() must be modified if a new
# always available algorithm is added.
__always_supported = ('md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512',
                      'blake2b', 'blake2s',
                      'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
                      'shake_128', 'shake_256')


algorithms_guaranteed = set(__always_supported)
algorithms_available = set(__always_supported)

__all__ = __always_supported + ('new', 'algorithms_guaranteed',
                                'algorithms_available', 'pbkdf2_hmac', 'file_digest')


__builtin_constructor_cache = {}

# Prefer our blake2 implementation
# OpenSSL 1.1.0 comes with a limited implementation of blake2b/s. The OpenSSL
# implementations neither support keyed blake2 (blake2 MAC) nor advanced
# features like salt, personalization, or tree hashing. OpenSSL hash-only
# variants are available as 'blake2b512' and 'blake2s256', though.
__block_openssl_constructor = {
    'blake2b', 'blake2s',
}

def __get_builtin_constructor(name):
    cache = __builtin_constructor_cache
    constructor = cache.get(name)
    if constructor is not None:
        return constructor
    try:
        if name in {'SHA1', 'sha1'}:
            import _sha1
            cache['SHA1'] = cache['sha1'] = _sha1.sha1
        elif name in {'MD5', 'md5'}:
            import _md5
            cache['MD5'] = cache['md5'] = _md5.md5
        elif name in {'SHA256', 'sha256', 'SHA224', 'sha224'}:
            import _sha256
            cache['SHA224'] = cache['sha224'] = _sha256.sha224
            cache['SHA256'] = cache['sha256'] = _sha256.sha256
        elif name in {'SHA512', 'sha512', 'SHA384', 'sha384'}:
            import _sha512
            cache['SHA384'] = cache['sha384'] = _sha512.sha384
            cache['SHA512'] = cache['sha512'] = _sha512.sha512
        elif name in {'blake2b', 'blake2s'}:
            import _blake2
            cache['blake2b'] = _blake2.blake2b
            cache['blake2s'] = _blake2.blake2s
        elif name in {'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512'}:
            import _sha3
            cache['sha3_224'] = _sha3.sha3_224
            cache['sha3_256'] = _sha3.sha3_256
            cache['sha3_384'] = _sha3.sha3_384
            cache['sha3_512'] = _sha3.sha3_512
        elif name in {'shake_128', 'shake_256'}:
            import _sha3
            cache['shake_128'] = _sha3.shake_128
            cache['shake_256'] = _sha3.shake_256
    except ImportError:
        pass  # no extension module, this hash is unsupported.

    constructor = cache.get(name)
    if constructor is not None:
        return constructor

    raise ValueError('unsupported hash type ' + name)


def __get_openssl_constructor(name):
    if name in __block_openssl_constructor:
        # Prefer our builtin blake2 implementation.
        return __get_builtin_constructor(name)
    try:
        # MD5, SHA1, and SHA2 are in all supported OpenSSL versions
        # SHA3/shake are available in OpenSSL 1.1.1+
        f = getattr(_hashlib, 'openssl_' + name)
        # Allow the C module to raise ValueError.  The function will be
        # defined but the hash not actually available.  Don't fall back to
        # builtin if the current security policy blocks a digest, bpo#40695.
        f(usedforsecurity=False)
        # Use the C function directly (very fast)
        return f
    except (AttributeError, ValueError):
        return __get_builtin_constructor(name)


def __py_new(name, data=b'', **kwargs):
    """new(name, data=b'', **kwargs) - Return a new hashing object using the
    named algorithm; optionally initialized with data (which must be
    a bytes-like object).
    """
    return __get_builtin_constructor(name)(data, **kwargs)


def __hash_new(name, data=b'', **kwargs):
    """new(name, data=b'') - Return a new hashing object using the named algorithm;
    optionally initialized with data (which must be a bytes-like object).
    """
    if name in __block_openssl_constructor:
        # Prefer our builtin blake2 implementation.
        return __get_builtin_constructor(name)(data, **kwargs)
    try:
        return _hashlib.new(name, data, **kwargs)
    except ValueError:
        # If the _hashlib module (OpenSSL) doesn't support the named
        # hash, try using our builtin implementations.
        # This allows for SHA224/256 and SHA384/512 support even though
        # the OpenSSL library prior to 0.9.8 doesn't provide them.
        return __get_builtin_constructor(name)(data)


try:
    import _hashlib
    new = __hash_new
    __get_hash = __get_openssl_constructor
    algorithms_available = algorithms_available.union(
            _hashlib.openssl_md_meth_names)
except ImportError:
    _hashlib = None
    new = __py_new
    __get_hash = __get_builtin_constructor

try:
    # OpenSSL's PKCS5_PBKDF2_HMAC requires OpenSSL 1.0+ with HMAC and SHA
    from _hashlib import pbkdf2_hmac
except ImportError:
    from warnings import warn as _warn
    _trans_5C = bytes((x ^ 0x5C) for x in range(256))
    _trans_36 = bytes((x ^ 0x36) for x in range(256))

    def pbkdf2_hmac(hash_name, password, salt, iterations, dklen=None):
        """Password based key derivation function 2 (PKCS #5 v2.0)

        This Python implementations based on the hmac module about as fast
        as OpenSSL's PKCS5_PBKDF2_HMAC for short passwords and much faster
        for long passwords.
        """
        _warn(
            "Python implementation of pbkdf2_hmac() is deprecated.",
            category=DeprecationWarning,
            stacklevel=2
        )
        if not isinstance(hash_name, str):
            raise TypeError(hash_name)

        if not isinstance(password, (bytes, bytearray)):
            password = bytes(memoryview(password))
        if not isinstance(salt, (bytes, bytearray)):
            salt = bytes(memoryview(salt))

        # Fast inline HMAC implementation
        inner = new(hash_name)
        outer = new(hash_name)
        blocksize = getattr(inner, 'block_size', 64)
        if len(password) > blocksize:
            password = new(hash_name, password).digest()
        password = password + b'\x00' * (blocksize - len(password))
        inner.update(password.translate(_trans_36))
        outer.update(password.translate(_trans_5C))

        def prf(msg, inner=inner, outer=outer):
            # PBKDF2_HMAC uses the password as key. We can re-use the same
            # digest objects and just update copies to skip initialization.
            icpy = inner.copy()
            ocpy = outer.copy()
            icpy.update(msg)
            ocpy.update(icpy.digest())
            return ocpy.digest()

        if iterations < 1:
            raise ValueError(iterations)
        if dklen is None:
            dklen = outer.digest_size
        if dklen < 1:
            raise ValueError(dklen)

        dkey = b''
        loop = 1
        from_bytes = int.from_bytes
        while len(dkey) < dklen:
            prev = prf(salt + loop.to_bytes(4))
            # endianness doesn't matter here as long to / from use the same
            rkey = from_bytes(prev)
            for i in range(iterations - 1):
                prev = prf(prev)
                # rkey = rkey ^ prev
                rkey ^= from_bytes(prev)
            loop += 1
            dkey += rkey.to_bytes(inner.digest_size)

        return dkey[:dklen]

try:
    # OpenSSL's scrypt requires OpenSSL 1.1+
    from _hashlib import scrypt
except ImportError:
    pass


def file_digest(fileobj, digest, /, *, _bufsize=2**18):
    """Hash the contents of a file-like object. Returns a digest object.

    *fileobj* must be a file-like object opened for reading in binary mode.
    It accepts file objects from open(), io.BytesIO(), and SocketIO objects.
    The function may bypass Python's I/O and use the file descriptor *fileno*
    directly.

    *digest* must either be a hash algorithm name as a *str*, a hash
    constructor, or a callable that returns a hash object.
    """
    # On Linux we could use AF_ALG sockets and sendfile() to archive zero-copy
    # hashing with hardware acceleration.
    if isinstance(digest, str):
        digestobj = new(digest)
    else:
        digestobj = digest()

    if hasattr(fileobj, "getbuffer"):
        # io.BytesIO object, use zero-copy buffer
        digestobj.update(fileobj.getbuffer())
        return digestobj

    # Only binary files implement readinto().
    if not (
        hasattr(fileobj, "readinto")
        and hasattr(fileobj, "readable")
        and fileobj.readable()
    ):
        raise ValueError(
            f"'{fileobj!r}' is not a file-like object in binary reading mode."
        )

    # binary file, socket.SocketIO object
    # Note: socket I/O uses different syscalls than file I/O.
    buf = bytearray(_bufsize)  # Reusable buffer to reduce allocations.
    view = memoryview(buf)
    while True:
        size = fileobj.readinto(buf)
        if size == 0:
            break  # EOF
        digestobj.update(view[:size])

    return digestobj


for __func_name in __always_supported:
    # try them all, some may not work due to the OpenSSL
    # version not supporting that algorithm.
    try:
        globals()[__func_name] = __get_hash(__func_name)
    except ValueError:
        import logging
        logging.exception('code for hash %s was not found.', __func_name)


# Cleanup locals()
del __always_supported, __func_name, __get_hash
del __py_new, __hash_new, __get_openssl_constructor

