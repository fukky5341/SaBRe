## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2871263532805734


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.5152187, -5.5987096, -6.5152187, -5.5987096, -0.6983976, 0.6983979)
1: (2.9799585, 3.7526534, 2.9799585, 3.7526534, -0.5966959, 0.5966954)
2: (-6.4165950, -5.6416893, -6.4165950, -5.6416893, -0.4807303, 0.4807303)
3: (-12.0379648, -11.0129137, -12.0379648, -11.0129137, -0.5216932, 0.5216932)
4: (-4.8498316, -4.0119658, -4.8498316, -4.0119658, -0.5992346, 0.5992346)
5: (-10.7948637, -9.8428917, -10.7948637, -9.8428917, -0.5159090, 0.5159090)
6: (-8.7938385, -7.5921826, -8.7938385, -7.5921826, -0.7673030, 0.7673030)
7: (-4.4236746, -3.6883411, -4.4236746, -3.6883411, -0.4857280, 0.4857280)
8: (-1.8548965, -1.2010918, -1.8548965, -1.2010918, -0.3967638, 0.3967638)
9: (-9.9256058, -8.9051409, -9.9256058, -8.9051409, -0.7935338, 0.7935338)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.25 + 33.37 = 57.63 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.2609425, upper bound: 0.2609430
