## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.263905785


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1343257, 1.1343260)
1: (-6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9096284, 0.9096284)
2: (-0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7806470, 0.7806470)
3: (-2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6144085, 0.6144085)
4: (-9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8248234, 0.8248234)
5: (-8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5613703, 0.5613702)
6: (-10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7603209, 0.7603209)
7: (3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6888497, 0.6888497)
8: (-4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6941956, 0.6941956)
9: (-3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9527485, 0.9527488)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.30 + 35.12 = 59.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.2665715, upper bound: 0.2665707

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6135
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665712, upper bound: 0.2655082
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2655075, upper bound: 0.2665704
time: 3.72 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.24
Output dim: 7, lower bound: -0.2665712, upper bound: 0.2655082
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.24
Output dim: 7, lower bound: -0.2655075, upper bound: 0.2665704

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1344399, 1.1344554
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9094801, 0.9092140
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7801919, 0.7802513
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6143453, 0.6143365
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8252029, 0.8251584
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5618675, 0.5619337
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7529354, 0.7538815
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6889477, 0.6888402
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6928889, 0.6932174
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9502716, 0.9498281

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6135
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6135

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643270, upper bound: 0.2655055
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665688, upper bound: 0.2632645
time: 3.31 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1344552, 1.1344402
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9092140, 0.9094801
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7802513, 0.7801919
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6143365, 0.6143453
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8251584, 0.8252029
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5619336, 0.5618675
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7538815, 0.7529356
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6888402, 0.6889477
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6932174, 0.6928890
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9498281, 0.9502718

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2655068, upper bound: 0.2665693
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2655045, upper bound: 0.2665716
time: 3.28 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.32 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.32
Output dim: 7, lower bound: -0.2643270, upper bound: 0.2655055
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.32
Output dim: 7, lower bound: -0.2665688, upper bound: 0.2632645
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.32
Output dim: 7, lower bound: -0.2655068, upper bound: 0.2665693
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.32
Output dim: 7, lower bound: -0.2655045, upper bound: 0.2665716

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1395335, 1.1408405
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8781374, 0.8828819
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7754741, 0.7748160
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5938065, 0.5963590
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7918291, 0.7870395
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5307413, 0.5263721
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7360146, 0.7333617
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6747556, 0.6769993
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6927403, 0.6921347
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9465842, 0.9456202

Time for backsubstitution: 23.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643266, upper bound: 0.2655029
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643265, upper bound: 0.2655047
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1408253, 1.1395488
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8831480, 0.8778713
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7747569, 0.7755332
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5963678, 0.5937977
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7870839, 0.7917848
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5263059, 0.5308075
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7324159, 0.7369604
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6771069, 0.6746480
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6918062, 0.6930686
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9460638, 0.9461406

Time for backsubstitution: 22.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665681, upper bound: 0.2632636
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665680, upper bound: 0.2632637
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1332698, 1.1293578
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9090776, 0.9089005
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7792544, 0.7759209
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6135672, 0.6110569
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8237123, 0.8248656
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5617412, 0.5618221
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7528846, 0.7486784
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6886656, 0.6882033
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6931860, 0.6927549
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9496152, 0.9493682

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6135
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6135

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2632630, upper bound: 0.2665688
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2655040, upper bound: 0.2643273
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1293731, 1.1332548
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9086342, 0.9093435
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7759800, 0.7791951
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6110481, 0.6135761
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8248212, 0.8237567
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5618883, 0.5616751
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7496240, 0.7519395
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6880960, 0.6887732
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930833, 0.6928576
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9489243, 0.9500592

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2654281, upper bound: 0.2646363
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2635723, upper bound: 0.2664924
time: 3.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.55 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 7, lower bound: -0.2643266, upper bound: 0.2655029
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 7, lower bound: -0.2643265, upper bound: 0.2655047
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 7, lower bound: -0.2665681, upper bound: 0.2632636
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 7, lower bound: -0.2665680, upper bound: 0.2632637
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 7, lower bound: -0.2632630, upper bound: 0.2665688
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 7, lower bound: -0.2655040, upper bound: 0.2643273
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 7, lower bound: -0.2654281, upper bound: 0.2646363
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 7, lower bound: -0.2635723, upper bound: 0.2664924

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1383476, 1.1357572
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8780010, 0.8823023
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7744770, 0.7705450
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930374, 0.5930707
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7903836, 0.7867031
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5305492, 0.5263269
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7350190, 0.7291048
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6745811, 0.6762552
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6927088, 0.6920005
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9463723, 0.9447170

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642475, upper bound: 0.2635702
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2623916, upper bound: 0.2654260
time: 3.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1344504, 1.1396544
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8775580, 0.8827455
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7712028, 0.7738192
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5905182, 0.5955898
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7914925, 0.7855942
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5306962, 0.5261799
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7317579, 0.7323649
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6740112, 0.6768248
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6926059, 0.6921033
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9456811, 0.9454077

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642475, upper bound: 0.2635703
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2623915, upper bound: 0.2654261
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1396389, 1.1344657
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8830116, 0.8772917
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7737598, 0.7712622
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5955985, 0.5905094
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7856386, 0.7914481
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5261137, 0.5307623
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7314198, 0.7327034
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6769323, 0.6739039
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6917747, 0.6929344
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9458516, 0.9452375

Time for backsubstitution: 22.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664890, upper bound: 0.2613287
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2646332, upper bound: 0.2631845
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1357422, 1.1383626
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8825686, 0.8777349
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7704856, 0.7745364
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930796, 0.5930284
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7867475, 0.7903392
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5262607, 0.5306153
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7281587, 0.7359641
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6763625, 0.6744735
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6916722, 0.6930373
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9451606, 0.9459282

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664889, upper bound: 0.2613288
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2646333, upper bound: 0.2631845
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1383629, 1.1357422
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8777349, 0.8825684
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7745364, 0.7704856
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930285, 0.5930796
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7903392, 0.7867475
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5306153, 0.5262607
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7359641, 0.7281587
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6744735, 0.6763625
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930374, 0.6916720
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9459283, 0.9451605

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2631839, upper bound: 0.2646339
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2613281, upper bound: 0.2664896
time: 3.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1396542, 1.1344504
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8827455, 0.8775578
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7738192, 0.7712028
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5955899, 0.5905182
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7855942, 0.7914927
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5261799, 0.5306962
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7323649, 0.7317574
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6768248, 0.6740112
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6921033, 0.6926059
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9454076, 0.9456812

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2654254, upper bound: 0.2623922
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2635696, upper bound: 0.2642481
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1293778, 1.1332610
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9086351, 0.9093454
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7759840, 0.7791984
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6110494, 0.6135774
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8248191, 0.8237553
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5618857, 0.5616728
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7496266, 0.7519424
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6880965, 0.6887736
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930810, 0.6928556
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9489238, 0.9500586

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6135

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2631838, upper bound: 0.2646340
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2654253, upper bound: 0.2623923
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1293788, 1.1332598
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9086361, 0.9093449
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7759833, 0.7791989
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6110497, 0.6135772
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8248198, 0.8237545
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5618861, 0.5616726
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7496271, 0.7519419
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6880965, 0.6887739
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6930813, 0.6928552
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9489241, 0.9500586

Time for backsubstitution: 22.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6135

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2613281, upper bound: 0.2664897
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2635695, upper bound: 0.2642482
time: 3.41 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2642475, upper bound: 0.2635702
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2623916, upper bound: 0.2654260
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2642475, upper bound: 0.2635703
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2623915, upper bound: 0.2654261
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2664890, upper bound: 0.2613287
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2646332, upper bound: 0.2631845
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2664889, upper bound: 0.2613288
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2646333, upper bound: 0.2631845
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2631839, upper bound: 0.2646339
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2613281, upper bound: 0.2664896
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2654254, upper bound: 0.2623922
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2635696, upper bound: 0.2642481
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2631838, upper bound: 0.2646340
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2654253, upper bound: 0.2623923
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2613281, upper bound: 0.2664897
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.30
Output dim: 7, lower bound: -0.2635695, upper bound: 0.2642482

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1383524, 1.1357634
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8780026, 0.8823040
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7744808, 0.7705481
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930383, 0.5930719
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7903810, 0.7867012
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5305465, 0.5263246
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7350209, 0.7291076
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6745818, 0.6762557
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6927060, 0.6919981
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9463711, 0.9447159

Time for backsubstitution: 22.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2593
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 681
type: DSZ, layer: 3, pos: 1116
type: DSZ, layer: 3, pos: 583
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1681
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 606
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2360
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 711

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2621592, upper bound: 0.2596321
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2604304, upper bound: 0.2614653
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1383533, 1.1357625
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8780026, 0.8823037
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7744801, 0.7705486
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5930383, 0.5930718
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7903817, 0.7867002
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5305469, 0.5263244
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7350214, 0.7291071
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6745815, 0.6762557
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6927065, 0.6919978
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9463711, 0.9447159

Time for backsubstitution: 23.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 583
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 1116
type: DSZ, layer: 3, pos: 2593
type: DSZ, layer: 3, pos: 2360
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1681
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 606
type: DSZ, layer: 3, pos: 681

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2369

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2610025, upper bound: 0.2639386
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2609051, upper bound: 0.2640360
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1344552, 1.1396606
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8775592, 0.8827472
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7712064, 0.7738223
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5905192, 0.5955911
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7914898, 0.7855921
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5306935, 0.5261776
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7317598, 0.7323678
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6740119, 0.6768253
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6926030, 0.6921009
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9456801, 0.9454066

Time for backsubstitution: 23.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1116
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 1681
type: DSZ, layer: 3, pos: 583
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 606
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 2360
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 681
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 2593

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1116

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2617208, upper bound: 0.2608395
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2617208, upper bound: 0.2608751
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1344562, 1.1396594
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8775592, 0.8827469
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7712059, 0.7738228
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5905194, 0.5955908
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7914906, 0.7855914
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5306939, 0.5261774
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7317603, 0.7323673
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6740117, 0.6768253
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6926035, 0.6921006
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9456801, 0.9454066

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2853
type: DSZ, layer: 3, pos: 2138
type: DSZ, layer: 3, pos: 2641
type: DSZ, layer: 3, pos: 1445
type: DSZ, layer: 3, pos: 583
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1681
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 681
type: DSZ, layer: 3, pos: 2593
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 2360
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 606
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 228
type: DSZ, layer: 3, pos: 1116

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2853

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.2623820, upper bound: 0.2633084
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2602748, upper bound: 0.2654164
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1396441, 1.1344719
1: -6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.8830132, 0.8772933
2: -0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7737637, 0.7712653
3: -2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.5955997, 0.5905106
4: -9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.7856357, 0.7914462
5: -8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5261111, 0.5307600
6: -10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7314222, 0.7327063
7: 3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6769331, 0.6739042
8: -4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6917719, 0.6929320
9: -3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9458504, 0.9452366

Time for backsubstitution: 23.17 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.42 + 559.61 = 619.03 seconds
