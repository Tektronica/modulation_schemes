import numpy as np
import numpy.matlib
import timeit
import matplotlib.pyplot as plt

fm = 100  # modulation frequency
fs = 1000  # sample rate
N = 41  # number of samples

T = int(fs / fm)

print('The number of samples:', N)
print('The period of one bit:', T, 'samples')
print('The samples per period', T)
print('The number of bits displayed:', int(N / T))
print()

"""
======================================================
MESSAGE CLOCK (message frequency):
======================================================
    +---+   +---+   +---+   +---+
    |   |   |   |   |   |   |   |
+---+   +---+   +---+   +---+   +---+
  0   1   0   1   0   1   0   1   0
|<----->|<----->|<----->|<----->|
    T       T       T       T

======================================================
MESSAGE (bit can only change on rising edge):
======================================================
    +---+---+       +---+---+
    |       |       |       |
+---+       +---+---+       +---+---+
        1       0       1       0   
    |<----->|<----->|<----->|<----->|
        T       T       T       T
"""

N_dt_per_bit = 1 / fm  # bit dt length is 1 period + 90deg shift

# REPEAT NUMPY ARRAY MANIPULATION ======================================================================================
binary_message = np.round(np.random.rand(1, int(N / T)))[0]
upsampled1 = np.repeat(binary_message, T)[:N]

# plot -----------------------------------------------------------------------------------------------------------------
figure = plt.figure()  # look into Figure((5, 4), 75)
ax1 = figure.add_subplot(111)
temporal, = ax1.plot(range(N-1), upsampled1, '-')  # sampled message
plt.show()

# KRONECKER PRODUCT ====================================================================================================
# array of ones
array_ones = np.ones(T)  # array of ones created that the binary message will be mapped to
upsampled2 = np.kron(binary_message, array_ones)[:N]
print('Does np.kron match np.repeat?', np.array_equal(upsampled1, upsampled2))

# plot -----------------------------------------------------------------------------------------------------------------
figure = plt.figure()  # look into Figure((5, 4), 75)
ax1 = figure.add_subplot(111)
temporal2, = ax1.plot(range(N-1), upsampled2, '-')  # sampled message
plt.show()

# MATLAB ===============================================================================================================
# matlab: reshape(repmat(round(rand(1,N_pts/N_dt_per_bit )),N_dt_per_bit ,1),[1,N_pts]);
# ravel performed in column-major order
upsampled3 = np.ravel(numpy.matlib.repmat(binary_message, T, 1), order='F')[:N]
print('Does reshape/repmat match np.repeat?', np.array_equal(upsampled1, upsampled3))

# plot -----------------------------------------------------------------------------------------------------------------
figure = plt.figure()  # look into Figure((5, 4), 75)
ax1 = figure.add_subplot(111)
temporal3, = ax1.plot(range(N-1), upsampled3, '-')  # sampled message
plt.show()

# test =================================================================================================================
#  For timeit, the main statement, the setup statement and the timer function to be used are passed to the constructor.
print('\ntiming test has started -----------------------------------')

# REPEAT ---------------------------------------------------------------------------------------------------------------
Ntests = 1000000
test1 = timeit.timeit("np.repeat(binary_message, T)[:N]",
                      setup='import numpy as np; from __main__ import ' + ', '.join(globals()),
                      number=Ntests)
print(f'time for repeat/slice: {round((test1 * 1e3 / Ntests), 3)}ms ({round((test1 * 1e6 / Ntests), 3)}us)')

# KRONECKER ------------------------------------------------------------------------------------------------------------
test2 = timeit.timeit("k = np.ones(T)\nnp.kron(binary_message, k)[:N]",
                      setup='import numpy as np\nfrom __main__ import ' + ', '.join(globals()),
                      number=Ntests)
print(f'time for kronecher/slice: {round((test2 * 1e3 / Ntests), 3)}ms ({round((test2 * 1e6 / Ntests), 3)}us)')

# RAVEL ----------------------------------------------------------------------------------------------------------------
test3 = timeit.timeit("np.ravel(numpy.matlib.repmat(binary_message, T, 1), order='F')[:N]",
                      setup='import numpy as np; import numpy.matlib; from __main__ import ' + ', '.join(globals()),
                      number=Ntests)
print(f'time for ravel/repmat : {round((test3 * 1e3 / Ntests), 3)}ms ({round((test3 * 1e6 / Ntests), 3)}us)')
