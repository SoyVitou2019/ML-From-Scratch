import numpy as np
import matplotlib.pylab as plt

xs = np.asarray([[0, 1, 0, 1, 0],
                 [0, 0, 2, 1, 0],
                 [1, 1, 0, 1, 0],
                 [1, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

ys = np.asarray([[0],
                 [8],
                 [3],
                 [3],
                 [0]])

ins = 5
outs = 1


def weight(input_x, output):
    wss = np.random.randn(input_x, output)
    return wss


ws = weight(ins, outs)

ers = []
for i in range(14000):
    yh = xs @ ws
    e = yh - ys
    e = np.sum(np.abs(e))
    if e < 0.05:
        print("found solution")
        print(ws)
        break
    else:
        mutation = weight(ins, outs) * 0.1
        cw = ws + mutation
        yh = xs @ cw
        ce = yh - ys
        ce = np.sum(np.abs(ce))
        if ce < e:
            ws = cw
    ers.append(e)


print(min(ers))
# plt.figure(1)
# plt.plot(ers)
# plt.show()