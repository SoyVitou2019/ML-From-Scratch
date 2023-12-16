import numpy as np
import matplotlib.pylab as plt

xs = np.asarray([[0, 1, 0, 1, 0],
                 [0, 0, 1, 1, 0],
                 [1, 1, 0, 1, 0],
                 [1, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

ys = np.asarray([[0],
                 [1],
                 [1],
                 [1],
                 [0]])

ins = 5
outs = 1


def weight(input_x, output):
    wss = np.random.randn(input_x, output)
    return wss


ws = weight(ins, outs)

ers = []
for i in range(5000):
    yh = xs @ ws
    e = yh - ys
    e = np.sum(np.abs(e))
    if e < 0.1:
        print("found solution")
        print(ws)
    else:
        ws = weight(ins, outs)

    ers.append(e)


print(min(ers))
plt.figure(1)
plt.plot(ers)
plt.show()