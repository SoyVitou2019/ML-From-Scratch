import numpy as np
import matplotlib.pyplot as plt

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
nodes = 15

# xs = np.hstack((xs, np.ones([xs.shape[0], 1])))

print(xs)


def weight(input_x, output):
    wss = np.random.randn(input_x, output)
    return wss


w1 = weight(ins, nodes)
w2 = weight(nodes, outs)

ers = []
for i in range(5000):
    y1 = xs @ w1
    y1 = np.sin(y1)
    yh = y1 @ w2
    e = yh - ys
    e = np.sum(np.abs(e))
    if e < 0.05:
        print("found solution")
        break
    else:
        cw1 = w1 + weight(ins, nodes) * 0.01
        cw2 = w2 + weight(nodes, outs) * 0.01
        yc1 = xs @ cw1
        yc1 = np.sin(yc1)
        yh = yc1 @ cw2
        ce = yh - ys
        ce = np.sum(np.abs(ce))
        if ce < e:
            w1 = cw1
            w2 = cw2
    ers.append(e)

print(min(ers))
plt.figure(1)
plt.plot(ers)
# plt.show()
