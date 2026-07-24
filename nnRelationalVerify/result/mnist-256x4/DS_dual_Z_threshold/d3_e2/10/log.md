## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.00128651


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3655797, 0.4323223, -0.3655797, 0.4323223, -0.7979019, 0.7979019)
1: (-0.3507084, 0.3995872, -0.3507084, 0.3995872, -0.7502956, 0.7502956)
2: (-0.4036356, 0.5031865, -0.4036356, 0.5031865, -0.9068221, 0.9068221)
3: (-0.2482407, 0.3800163, -0.2482407, 0.3800163, -0.6282570, 0.6282570)
4: (-0.3940735, 0.4997456, -0.3940735, 0.4997456, -0.8938191, 0.8938191)
5: (-0.3277625, 0.6990618, -0.3277625, 0.6990618, -1.0268242, 1.0268242)
6: (0.0166956, 1.4230980, 0.0166956, 1.4230980, -1.4064023, 1.4064023)
7: (-0.4457768, 0.4976362, -0.4457768, 0.4976362, -0.9434130, 0.9434130)
8: (-0.3735492, 0.4859239, -0.3735492, 0.4859239, -0.8594730, 0.8594730)
9: (-0.4441622, 0.4545314, -0.4441622, 0.4545314, -0.8986936, 0.8986936)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.99 + 3.26 = 5.25 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -1.0539859, upper bound: 1.0539859
