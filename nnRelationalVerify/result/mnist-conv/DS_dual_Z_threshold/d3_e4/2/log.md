## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.579334386


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5872164, 3.5872157)
1: (-7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3974328, 3.3974328)
2: (-10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817)
3: (-12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7899914, 2.7899911)
4: (5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2979751, 3.2979755)
5: (-8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6998472, 2.6998470)
6: (-12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3918996, 2.3918998)
7: (-5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7433391, 2.7433386)
8: (-1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688)
9: (-6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5790672, 2.5790672)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.68 + 34.17 = 56.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.6115657, upper bound: 1.6115677

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6250
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6250

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115651, upper bound: 1.6115672
time: 4.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115651, upper bound: 1.6115672
time: 4.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.30 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.30
Output dim: 4, lower bound: -1.6115651, upper bound: 1.6115672
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.30
Output dim: 4, lower bound: -1.6115651, upper bound: 1.6115672

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5215502, 3.5349178
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3992057, 3.3989911
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7887111, 2.7949195
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7661753, 2.7576551
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2979774, 3.2979536
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6935310, 2.6926286
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3214431, 2.3310506
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7156959, 2.7116141
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5823317, 2.5799508

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115515, upper bound: 1.5954350
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954350, upper bound: 1.6115537
time: 5.12 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5349178, 3.5215499
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3989902, 3.3992057
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7949190, 2.7887111
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7576551, 2.7661750
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2979546, 3.2979774
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6926289, 2.6935308
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3310504, 2.3214428
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7116141, 2.7156959
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5799508, 2.5823317

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115515, upper bound: 1.5954350
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954350, upper bound: 1.6115537
time: 4.91 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.18 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.18
Output dim: 4, lower bound: -1.6115515, upper bound: 1.5954350
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.18
Output dim: 4, lower bound: -1.5954350, upper bound: 1.6115537
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.18
Output dim: 4, lower bound: -1.6115515, upper bound: 1.5954350
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.18
Output dim: 4, lower bound: -1.5954350, upper bound: 1.6115537

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5307102, 3.5454702
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3384037, 3.3518186
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7976165, 2.7849491
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2933469, 3.2926641
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6765862, 2.6698573
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3381786, 2.3455808
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6819925, 2.6821165
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6028948, 2.6036377

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115503, upper bound: 1.5953410
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895304, upper bound: 1.5954352
time: 4.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5321026, 3.5440779
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3520336, 3.3381877
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7934690, 2.7890966
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2926869, 3.2933240
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6707592, 2.6756840
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3359728, 2.3477862
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6861982, 2.6779118
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2211514, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6060185, 2.6005139

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954329, upper bound: 1.5895326
time: 4.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5953411, upper bound: 1.6115503
time: 4.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5440779, 3.5321026
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3381882, 3.3520336
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7890968, 2.7934690
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2933240, 3.2926874
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6756840, 2.6707594
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3477859, 2.3359730
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6779118, 2.6861982
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2211518
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6005139, 2.6060190

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115503, upper bound: 1.5953410
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895304, upper bound: 1.5954352
time: 4.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5454702, 3.5307102
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3518200, 3.3384032
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7849493, 2.7976165
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2926641, 3.2933469
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6698570, 2.6765862
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3455811, 2.3381784
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6821165, 2.6819930
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6036377, 2.6028948

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954329, upper bound: 1.5895327
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5953411, upper bound: 1.6115501
time: 4.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.10
Output dim: 4, lower bound: -1.6115503, upper bound: 1.5953410
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.10
Output dim: 4, lower bound: -1.5895304, upper bound: 1.5954352
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.10
Output dim: 4, lower bound: -1.5954329, upper bound: 1.5895326
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.10
Output dim: 4, lower bound: -1.5953411, upper bound: 1.6115503
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.10
Output dim: 4, lower bound: -1.6115503, upper bound: 1.5953410
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.10
Output dim: 4, lower bound: -1.5895304, upper bound: 1.5954352
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.10
Output dim: 4, lower bound: -1.5954329, upper bound: 1.5895327
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.10
Output dim: 4, lower bound: -1.5953411, upper bound: 1.6115501

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5314617, 3.5463638
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3365297, 3.3496890
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8030162, 2.7895017
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2943015, 3.2937865
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6768155, 2.6698515
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3346257, 2.3424723
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6841989, 2.6839814
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6035614, 2.6044288

Time for backsubstitution: 21.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1690

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5908308, upper bound: 1.5745847
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5908719, upper bound: 1.5745263
time: 4.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5316048, 3.5462217
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3362722, 3.3494201
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8021693, 2.7903485
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2941241, 3.2936182
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6765809, 2.6700864
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3342061, 2.3420281
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6838574, 2.6836324
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6036835, 2.6043043

Time for backsubstitution: 21.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1690

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5686495, upper bound: 1.5746833
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5687197, upper bound: 1.5746238
time: 5.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5328541, 3.5449715
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3496351, 3.3360581
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7988687, 2.7936492
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2936416, 3.2941008
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6709886, 2.6756783
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3324199, 2.3438141
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6877131, 2.6797762
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2205448, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6066852, 2.6013026

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1690

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5746233, upper bound: 1.5687194
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5746814, upper bound: 1.5686492
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5329952, 3.5448298
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3499041, 3.3363142
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7980218, 2.7944963
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2938104, 3.2942781
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6707540, 2.6759129
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3328643, 2.3442335
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6880622, 2.6801181
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2204609, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6068096, 2.6011806

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1690

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5745266, upper bound: 1.5908717
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5745849, upper bound: 1.5908303
time: 4.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5448303, 3.5329962
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3363142, 3.3499036
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7944961, 2.7980218
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2942777, 3.2938099
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6759133, 2.6707537
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3442330, 2.3328645
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6801181, 2.6880622
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2204604
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6011806, 2.6068096

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 1690

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5908308, upper bound: 1.5745847
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5908719, upper bound: 1.5745263
time: 4.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5449715, 3.5328541
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3360586, 3.3496351
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7936492, 2.7988684
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2941003, 3.2936416
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6756787, 2.6709886
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3438139, 2.3324203
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6797767, 2.6877131
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2205448
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6013026, 2.6066856

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1690

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5686495, upper bound: 1.5746833
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5687197, upper bound: 1.5746238
time: 5.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5462227, 3.5316038
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3494196, 3.3362727
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7903485, 2.8021691
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2936187, 3.2941236
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6700864, 2.6765804
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3420281, 2.3342063
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6836324, 2.6838574
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6043043, 2.6036835

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1690

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5746233, upper bound: 1.5687194
time: 5.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5746814, upper bound: 1.5686492
time: 4.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5463638, 3.5314617
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3496885, 3.3365293
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7895017, 2.8030162
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2937865, 3.2943010
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6698518, 2.6768150
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3424721, 2.3346257
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6839814, 2.6841989
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6044288, 2.6035614

Time for backsubstitution: 22.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1690

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5745266, upper bound: 1.5908717
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5745849, upper bound: 1.5908303
time: 5.08 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.80 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5908308, upper bound: 1.5745847
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5908719, upper bound: 1.5745263
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5686495, upper bound: 1.5746833
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5687197, upper bound: 1.5746238
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5746233, upper bound: 1.5687194
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5746814, upper bound: 1.5686492
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5745266, upper bound: 1.5908717
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5745849, upper bound: 1.5908303
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5908308, upper bound: 1.5745847
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5908719, upper bound: 1.5745263
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5686495, upper bound: 1.5746833
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5687197, upper bound: 1.5746238
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5746233, upper bound: 1.5687194
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5746814, upper bound: 1.5686492
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5745266, upper bound: 1.5908717
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.80
Output dim: 4, lower bound: -1.5745849, upper bound: 1.5908303

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.4612484, 3.4373305
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3016920, 3.2802043
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7753954, 2.7690639
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7822690, 2.7578101
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2493329, 3.2588491
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6580486, 2.6455107
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.2805467, 2.2729676
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6628728, 2.6532388
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.1984215, 3.2002573
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6020608, 2.6032081

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5736008, upper bound: 1.5598008
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5765166, upper bound: 1.5579794
time: 5.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.4224281, 3.4751585
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.2670450, 3.3140812
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7659569, 2.7780807
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7713246, 2.7684124
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2589822, 3.2488189
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6524734, 2.6508629
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.2651210, 2.2878244
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6534562, 2.6624331
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2005901, 3.1978412
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6023407, 2.6029148

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5735243, upper bound: 1.5597215
time: 5.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5765727, upper bound: 1.5579668
time: 7.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.4617901, 3.4357965
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3142958, 3.2668295
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7718725, 2.7721651
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7769322, 2.7628047
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2488427, 3.2589583
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6517649, 2.6515720
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.2782164, 2.2747288
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6665139, 2.6493754
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.1935081, 3.2049222
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6052957, 2.5999599

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 3, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5579670, upper bound: 1.5765723
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5597217, upper bound: 1.5735238
time: 4.32 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 31.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.60
Output dim: 4, lower bound: -1.5736008, upper bound: 1.5598008
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.60
Output dim: 4, lower bound: -1.5765166, upper bound: 1.5579794
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.60
Output dim: 4, lower bound: -1.5735243, upper bound: 1.5597215
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.60
Output dim: 4, lower bound: -1.5765727, upper bound: 1.5579668
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.60
Output dim: 4, lower bound: -1.5579670, upper bound: 1.5765723
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.60
Output dim: 4, lower bound: -1.5597217, upper bound: 1.5735238
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 4, lower bound: -1.5745849, upper bound: 1.5908303
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 4, lower bound: -1.5908308, upper bound: 1.5745847
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 4, lower bound: -1.5908719, upper bound: 1.5745263
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 4, lower bound: -1.5745266, upper bound: 1.5908717
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 4, lower bound: -1.5745849, upper bound: 1.5908303

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.85 + 546.58 = 603.43 seconds
