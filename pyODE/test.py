


from nn import Conv

convolution = Conv()


for i in range(0,9):
    kernel = np.asarray([0]*i + [1] + [0]*(9-i-1))
    Conv.forward(kernel=kernel)
