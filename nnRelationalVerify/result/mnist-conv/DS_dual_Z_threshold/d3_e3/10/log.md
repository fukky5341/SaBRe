## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0349644734999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2198710, 2.2198710)
1: (-6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7507243, 1.7507243)
2: (-4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5917411, 1.5917413)
3: (-7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0713673, 2.0713677)
4: (-11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3726444, 2.3726444)
5: (-6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7345924, 1.7345924)
6: (-10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0136123, 2.0136118)
7: (-2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8201418, 1.8201420)
8: (1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3549571, 1.3549573)
9: (-8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0553026, 2.0553021)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.05 + 35.18 = 59.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.0401645, upper bound: 1.0401642

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 565

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0364074, upper bound: 1.0401588
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401591, upper bound: 1.0364071
time: 4.50 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.59
Output dim: 7, lower bound: -1.0364074, upper bound: 1.0401588
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.59
Output dim: 7, lower bound: -1.0401591, upper bound: 1.0364071

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1712499, 2.1615338
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7029428, 1.7127006
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5877018, 1.5832717
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0568724, 2.0508142
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3423910, 2.3567748
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7337861, 1.7355711
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9869342, 1.9816034
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8038301, 1.8165982
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2995079, 1.2825783
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0547900, 2.0559211

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 5856

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0307187, upper bound: 1.0401238
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0363725, upper bound: 1.0344544
time: 4.55 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1615338, 2.1712503
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7127008, 1.7029431
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5832720, 1.5877020
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0508142, 2.0568724
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3567743, 2.3423915
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7355709, 1.7337861
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9816031, 1.9869337
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8165979, 1.8038309
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2825785, 1.2995080
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0559211, 2.0547895

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 5856

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0344546, upper bound: 1.0363723
time: 4.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401241, upper bound: 1.0307170
time: 6.85 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 33.86 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.86
Output dim: 7, lower bound: -1.0307187, upper bound: 1.0401238
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.86
Output dim: 7, lower bound: -1.0363725, upper bound: 1.0344544
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.86
Output dim: 7, lower bound: -1.0344546, upper bound: 1.0363723
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.86
Output dim: 7, lower bound: -1.0401241, upper bound: 1.0307170

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1714664, 2.1611404
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7029738, 1.7126451
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5868750, 1.5837266
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0557866, 2.0514102
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3423986, 2.3567619
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7325311, 1.7362604
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9852247, 1.9825404
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8039799, 1.8163326
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2990913, 1.2828116
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0553112, 2.0549688

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4626

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0307117, upper bound: 1.0382499
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0288298, upper bound: 1.0401184
time: 4.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1708570, 2.1615338
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7028875, 1.7127006
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5877018, 1.5824447
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0568724, 2.0497284
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3423777, 2.3567748
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7337861, 1.7343161
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9869342, 1.9798942
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8035650, 1.8165982
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2995079, 1.2821618
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0538368, 2.0559211

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4626

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0363670, upper bound: 1.0325804
time: 4.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0344711, upper bound: 1.0344498
time: 4.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1617484, 2.1708570
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7127314, 1.7028875
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5824447, 1.5881581
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0497284, 2.0574679
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3567801, 2.3423786
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7343159, 1.7344761
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9798942, 1.9878721
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8167467, 1.8035653
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2821620, 1.2997376
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0564423, 2.0538378

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4626

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0344490, upper bound: 1.0344720
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0325806, upper bound: 1.0363678
time: 4.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1611404, 2.1712503
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7126451, 1.7029431
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5832720, 1.5868750
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0508142, 2.0557866
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3567610, 2.3423915
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7355709, 1.7325311
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9816031, 1.9852247
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8163328, 1.8038309
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2825785, 1.2990915
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0549688, 2.0547895

Time for backsubstitution: 22.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4626

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401185, upper bound: 1.0288293
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0382501, upper bound: 1.0307115
time: 4.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.75 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.75
Output dim: 7, lower bound: -1.0307117, upper bound: 1.0382499
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.75
Output dim: 7, lower bound: -1.0288298, upper bound: 1.0401184
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.75
Output dim: 7, lower bound: -1.0363670, upper bound: 1.0325804
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.75
Output dim: 7, lower bound: -1.0344711, upper bound: 1.0344498
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.75
Output dim: 7, lower bound: -1.0344490, upper bound: 1.0344720
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.75
Output dim: 7, lower bound: -1.0325806, upper bound: 1.0363678
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.75
Output dim: 7, lower bound: -1.0401185, upper bound: 1.0288293
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.75
Output dim: 7, lower bound: -1.0382501, upper bound: 1.0307115

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1747022, 2.1636930
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7093043, 1.7148857
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5810757, 1.5764933
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0422249, 2.0401034
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3418531, 2.3563032
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7320499, 1.7358594
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9807992, 1.9788473
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7971506, 1.8081388
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2873886, 1.2735020
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0500693, 2.0505981

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277836, upper bound: 1.0382457
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0307078, upper bound: 1.0353174
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1740189, 2.1643758
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7052145, 1.7189758
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5796413, 1.5779271
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0444803, 2.0378485
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3419399, 2.3562174
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7321301, 1.7357795
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9815316, 1.9781151
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7957869, 1.8095033
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2897816, 1.2711089
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0509400, 2.0497274

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0258668, upper bound: 1.0401142
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0288266, upper bound: 1.0371880
time: 4.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1740923, 2.1640873
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7092180, 1.7149417
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5819020, 1.5752113
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0433111, 2.0384216
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3418341, 2.3563161
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7333050, 1.7339151
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9825077, 1.9762011
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7967358, 1.8084052
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2878051, 1.2728522
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0485959, 2.0515504

Time for backsubstitution: 22.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0334205, upper bound: 1.0325774
time: 4.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0363630, upper bound: 1.0296698
time: 4.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1643009, 2.1740923
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7149720, 1.7092183
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5752115, 1.5823584
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0384216, 2.0439062
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3563223, 2.3418341
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7339153, 1.7339952
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9762011, 1.9834466
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8085537, 1.7967360
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2728519, 1.2880349
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0520711, 2.0485959

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0296702, upper bound: 1.0363629
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0325766, upper bound: 1.0334214
time: 4.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1643758, 2.1738038
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7189760, 1.7051842
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5774717, 1.5796416
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0372529, 2.0444803
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3562174, 2.3419328
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7350898, 1.7321301
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9771771, 1.9815314
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8095036, 1.7956378
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2708755, 1.2897818
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0497270, 2.0504189

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0371881, upper bound: 1.0288266
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401146, upper bound: 1.0258665
time: 4.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1636930, 2.1744866
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7148857, 1.7092743
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5760379, 1.5810754
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0395079, 2.0422249
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3563032, 2.3418465
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7351694, 1.7320502
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9779091, 1.9807992
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8081388, 1.7970023
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2732687, 1.2873888
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0505977, 2.0495481

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 6220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0353177, upper bound: 1.0307076
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0382462, upper bound: 1.0277835
time: 4.38 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0277836, upper bound: 1.0382457
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0307078, upper bound: 1.0353174
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0258668, upper bound: 1.0401142
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0288266, upper bound: 1.0371880
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0334205, upper bound: 1.0325774
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0363630, upper bound: 1.0296698
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0296702, upper bound: 1.0363629
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0325766, upper bound: 1.0334214
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0371881, upper bound: 1.0288266
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0401146, upper bound: 1.0258665
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0353177, upper bound: 1.0307076
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 7, lower bound: -1.0382462, upper bound: 1.0277835

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1680565, 2.1599731
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7033377, 1.7042031
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5762777, 1.5738029
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0399323, 2.0388317
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3362560, 2.3462639
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7282448, 1.7290912
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9777522, 1.9771340
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7967668, 1.8074720
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2730474, 1.2654998
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0459647, 2.0482969

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0277788, upper bound: 1.0361946
time: 5.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0257341, upper bound: 1.0382409
time: 5.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1709828, 2.1570473
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.6986217, 1.7089195
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5783854, 1.5716951
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0409532, 2.0378108
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3318138, 2.3507071
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7252817, 1.7320540
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9790864, 1.9758000
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7964840, 1.8077548
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2793875, 1.2591608
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0477681, 2.0464940

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0307030, upper bound: 1.0332654
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0286579, upper bound: 1.0353127
time: 4.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1673732, 2.1606565
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.6992478, 1.7082932
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5748434, 1.5752368
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0421877, 2.0365763
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3363428, 2.3461771
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7283244, 1.7290113
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9784846, 1.9764018
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7954021, 1.8088365
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2754405, 1.2631068
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0468364, 2.0474257

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0258620, upper bound: 1.0380633
time: 4.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0238172, upper bound: 1.0401097
time: 4.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1703000, 2.1577301
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.6945314, 1.7130096
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5769520, 1.5731289
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0432081, 2.0355558
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3318996, 2.3506203
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7253618, 1.7319741
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9798183, 1.9750679
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7951198, 1.8091192
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2817802, 1.2567675
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0486388, 2.0456233

Time for backsubstitution: 22.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0288218, upper bound: 1.0351346
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0267751, upper bound: 1.0371832
time: 4.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1703730, 2.1574416
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.6985354, 1.7089753
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5792122, 1.5704131
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0420389, 2.0361290
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3317938, 2.3507204
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7265368, 1.7301097
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9807954, 1.9731538
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7960691, 1.8080204
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2798038, 1.2585108
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0462947, 2.0474463

Time for backsubstitution: 22.39 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.23 + 543.15 = 602.39 seconds
