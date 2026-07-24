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
execution time: IAR + RelationalAnalysis = 21.96 + 33.67 = 55.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.0401645, upper bound: 1.0401642

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4573

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401567, upper bound: 1.0383071
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0383073, upper bound: 1.0401576
time: 4.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.58 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.58
Output dim: 7, lower bound: -1.0401567, upper bound: 1.0383071
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.58
Output dim: 7, lower bound: -1.0383073, upper bound: 1.0401576

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2108917, 2.2090945
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7394032, 1.7357254
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.6006489, 1.6024454
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0733547, 2.0740995
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3754549, 2.3749914
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7345924, 1.7352667
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0239863, 2.0222433
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8233275, 1.8239734
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3600392, 1.3591850
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0590677, 2.0598283

Time for backsubstitution: 20.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 5858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5871

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0285680, upper bound: 1.0383051
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401549, upper bound: 1.0267185
time: 4.35 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2090945, 2.2108917
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7357259, 1.7394028
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.6024451, 1.6006494
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0741000, 2.0733547
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3749905, 2.3754559
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7352667, 1.7345924
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0222435, 2.0239859
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8239732, 1.8233275
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3591850, 1.3600392
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0598288, 2.0590682

Time for backsubstitution: 20.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 887

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5801

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0383033, upper bound: 1.0386586
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0368081, upper bound: 1.0401524
time: 4.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.92 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.92
Output dim: 7, lower bound: -1.0285680, upper bound: 1.0383051
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.92
Output dim: 7, lower bound: -1.0401549, upper bound: 1.0267185
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.92
Output dim: 7, lower bound: -1.0383033, upper bound: 1.0386586
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.92
Output dim: 7, lower bound: -1.0368081, upper bound: 1.0401524

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1973681, 2.1922150
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7373981, 1.7333217
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5955696, 1.5963492
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0745215, 2.0750813
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3130045, 2.3229628
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7354417, 1.7359822
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0170002, 2.0133224
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7646174, 1.7750633
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3283846, 1.3211865
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0334301, 2.0391026

Time for backsubstitution: 20.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 565

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0247871, upper bound: 1.0383010
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0285630, upper bound: 1.0345490
time: 4.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1940117, 2.1955714
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7369986, 1.7337213
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5945535, 1.5973656
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0743365, 2.0752659
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3234272, 2.3125396
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7353077, 1.7361162
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0150652, 2.0152574
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7744174, 1.7652636
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3220410, 1.3275301
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0383425, 2.0341902

Time for backsubstitution: 20.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6139

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401545, upper bound: 1.0265872
time: 4.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0400232, upper bound: 1.0267184
time: 4.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2050171, 2.2059994
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7117214, 1.7193928
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5911169, 1.5870569
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0573797, 2.0532980
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3569813, 2.3538532
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7315350, 1.7288411
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0163150, 2.0190444
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8180280, 1.8161900
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3493536, 1.3518465
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0531979, 2.0511217

Time for backsubstitution: 20.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5856

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0325986, upper bound: 1.0386236
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0382683, upper bound: 1.0329558
time: 4.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2042027, 2.2068138
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7157154, 1.7153988
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5888529, 1.5893209
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0540428, 2.0566349
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3533878, 2.3574467
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7295151, 1.7308605
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0173020, 2.0180578
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8168359, 1.8173826
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3509920, 1.3502083
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0518818, 2.0524387

Time for backsubstitution: 20.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5871

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0252215, upper bound: 1.0401516
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0368063, upper bound: 1.0285637
time: 4.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.01 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -1.0247871, upper bound: 1.0383010
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 29.01
Output dim: 7, lower bound: -1.0285630, upper bound: 1.0345490
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -1.0401545, upper bound: 1.0265872
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -1.0400232, upper bound: 1.0267184
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -1.0325986, upper bound: 1.0386236
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -1.0382683, upper bound: 1.0329558
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -1.0252215, upper bound: 1.0401516
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -1.0368063, upper bound: 1.0285637

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1487355, 2.1338658
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.6896167, 1.6952977
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5915294, 1.5878799
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0600247, 2.0545268
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2827549, 2.3070974
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7346363, 1.7369618
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9903145, 1.9813061
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7483087, 1.7715228
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2729399, 1.2488121
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0329165, 2.0397205

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6139

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0247868, upper bound: 1.0381691
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0246509, upper bound: 1.0383008
time: 4.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1826944, 2.1819921
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7277894, 1.7260489
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5940490, 1.5969436
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0728993, 2.0735412
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2917385, 2.2861328
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7336235, 1.7340968
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9681234, 1.9761569
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7744346, 1.7646568
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3042092, 1.3037069
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0418205, 2.0402589

Time for backsubstitution: 20.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 565

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4626

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401490, upper bound: 1.0247068
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0382805, upper bound: 1.0265815
time: 4.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1804323, 2.1842546
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7293267, 1.7245111
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5941310, 1.5968611
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0726113, 2.0738292
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2970200, 2.2808514
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7332888, 1.7344322
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9759650, 1.9683154
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7738104, 1.7652807
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2982178, 1.3096988
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0444107, 2.0376678

Time for backsubstitution: 20.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4572

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0400190, upper bound: 1.0246672
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0379713, upper bound: 1.0267135
time: 4.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2052312, 2.2056055
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7117558, 1.7193403
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5902905, 1.5875125
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0562940, 2.0538936
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3569937, 2.3538456
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7302799, 1.7295313
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0146093, 2.0199847
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8181853, 1.8159328
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3489492, 1.3520918
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0537195, 2.0501699

Time for backsubstitution: 20.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 5858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 957

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0209611, upper bound: 1.0378080
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0317844, upper bound: 1.0269793
time: 4.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2046227, 2.2059994
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7116694, 1.7193928
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5911169, 1.5862305
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0573797, 2.0522118
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3569736, 2.3538532
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7315350, 1.7275863
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0163150, 2.0173385
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8177705, 1.8161900
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3493536, 1.3514419
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0522470, 2.0511217

Time for backsubstitution: 20.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 4572

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0382664, upper bound: 1.0315596
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0368717, upper bound: 1.0329536
time: 7.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1906805, 2.1899352
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7137098, 1.7129936
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5837736, 1.5832257
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0552115, 2.0576186
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2909365, 2.3054180
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7303658, 1.7315767
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0103154, 2.0091362
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7581263, 1.7684724
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3193381, 1.3122103
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0262442, 2.0317144

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 5858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 957

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0216660, upper bound: 1.0393341
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0216883, upper bound: 1.0250136
time: 4.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1873240, 2.1932912
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7133102, 1.7133932
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5827575, 1.5842421
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0550270, 2.0578032
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3013601, 2.2949944
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7302313, 1.7317107
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0083804, 2.0110712
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7679262, 1.7586727
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3129945, 1.3185539
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0311575, 2.0268016

Time for backsubstitution: 20.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 887

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0368025, upper bound: 1.0270664
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0368016, upper bound: 1.0285616
time: 4.12 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0247868, upper bound: 1.0381691
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0246509, upper bound: 1.0383008
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0401490, upper bound: 1.0247068
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0382805, upper bound: 1.0265815
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0400190, upper bound: 1.0246672
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0379713, upper bound: 1.0267135
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0209611, upper bound: 1.0378080
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0317844, upper bound: 1.0269793
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0382664, upper bound: 1.0315596
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0368717, upper bound: 1.0329536
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0216660, upper bound: 1.0393341
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0216883, upper bound: 1.0250136
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0368025, upper bound: 1.0270664
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.46
Output dim: 7, lower bound: -1.0368016, upper bound: 1.0285616

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1374183, 2.1202865
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.6804066, 1.6876256
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5910258, 1.5874588
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0585890, 2.0528035
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2510667, 2.2806902
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7329502, 1.7349408
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9433722, 1.9422059
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7483268, 1.7709162
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2551084, 1.2249889
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0363936, 2.0457892

Time for backsubstitution: 20.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 6140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0247755, upper bound: 1.0347753
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0214011, upper bound: 1.0381581
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1351562, 2.1225486
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.6819444, 1.6860881
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5911088, 1.5873761
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0583014, 2.0530915
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2563481, 2.2754087
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7326150, 1.7352760
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9512138, 1.9343643
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7477026, 1.7715404
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2491169, 1.2309808
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0389848, 2.0431976

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 5858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 887

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0246488, upper bound: 1.0368030
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0231811, upper bound: 1.0382978
time: 4.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1859307, 2.1845450
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7341204, 1.7282901
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5882483, 1.5897095
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0593367, 2.0622334
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2911940, 2.2856731
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7331424, 1.7336950
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9636974, 1.9724634
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7676053, 1.7564628
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2925069, 1.2943974
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0365782, 2.0358877

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 565

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5856

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0344446, upper bound: 1.0246715
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0401140, upper bound: 1.0190046
time: 4.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1852474, 2.1852279
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7300301, 1.7323799
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5868149, 1.5911434
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0615921, 2.0599785
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2912798, 2.2855864
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7332220, 1.7336152
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9644299, 1.9717312
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7662406, 1.7578273
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2948999, 1.2920043
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0374489, 2.0350170

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4572
type: DSZ, layer: 1, pos: 5801

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4571

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0382692, upper bound: 1.0231950
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0348982, upper bound: 1.0265722
time: 4.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1755967, 2.1784501
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7352614, 1.7294912
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.6015658, 1.6057334
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0790024, 2.0814548
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2986355, 2.2822156
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7347507, 1.7361720
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9744887, 1.9670846
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7751751, 1.7669077
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2983367, 1.3098352
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0479193, 2.0406084

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5856
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 5858
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 6220
type: DSZ, layer: 1, pos: 4626
type: DSZ, layer: 1, pos: 4571
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5856

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0343147, upper bound: 1.0246317
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0399841, upper bound: 1.0189624
time: 4.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.1746283, 2.1794171
1: -6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7343059, 1.7304473
2: -4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.6030030, 1.6042960
3: -7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0802374, 2.0802202
4: -11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.2983856, 2.2824697
5: -6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7350283, 1.7358942
6: -10.4613619, -7.9367504, -10.4613619, -7.9367504, -1.9747338, 1.9668388
7: -2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.7754374, 1.7666454
8: 1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.2983587, 1.3098176
9: -8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0473528, 2.0411768

Time for backsubstitution: 21.02 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.63 + 564.20 = 619.83 seconds
