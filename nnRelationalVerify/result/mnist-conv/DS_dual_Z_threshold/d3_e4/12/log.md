## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1888011651


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3603992, 3.3603988)
1: (-11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0151329, 3.0151334)
2: (-10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6060905, 3.6060905)
3: (-5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4481053, 2.4481056)
4: (-11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5820861, 2.5820856)
5: (6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1325693, 2.1325696)
6: (-8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8638582, 2.8638577)
7: (-17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1436224, 3.1436229)
8: (-6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6549873, 2.6549873)
9: (-4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3357582, 2.3357592)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.93 + 40.31 = 63.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.1923782, upper bound: 1.1923775

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1873654, upper bound: 1.1923721
time: 25.54 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923708, upper bound: 1.1873651
time: 11.34 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 36.98 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 36.98
Output dim: 5, lower bound: -1.1873654, upper bound: 1.1923721
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 36.98
Output dim: 5, lower bound: -1.1923708, upper bound: 1.1873651

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3574286, 3.3589573
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0131721, 3.0141811
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6004429, 3.6033573
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4438601, 2.4393313
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5783925, 2.5802984
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1237822, 2.1283200
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8581381, 2.8611093
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1425409, 3.1430974
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6512146, 2.6471925
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3342609, 2.3350334

Time for backsubstitution: 20.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 444

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1873163, upper bound: 1.1879486
time: 12.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1829357, upper bound: 1.1829371
time: 114.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3589573, 3.3574286
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0141811, 3.0131726
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6033573, 3.6004415
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4393311, 2.4438605
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5802989, 2.5783920
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1283197, 2.1237824
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8611097, 2.8581386
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1430979, 3.1425409
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6471930, 2.6512144
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3350334, 2.3342605

Time for backsubstitution: 21.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 444

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923227, upper bound: 1.1829362
time: 10.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1879487, upper bound: 1.1873181
time: 36.54 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 68.46 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 68.46
Output dim: 5, lower bound: -1.1873163, upper bound: 1.1879486
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 68.46
Output dim: 5, lower bound: -1.1829357, upper bound: 1.1829371
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 68.46
Output dim: 5, lower bound: -1.1923227, upper bound: 1.1829362
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 68.46
Output dim: 5, lower bound: -1.1879487, upper bound: 1.1873181

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3608236, 3.3615503
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0179415, 3.0163894
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5885944, 3.5877986
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.3937616, 2.4039812
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5810337, 2.5776525
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.0992236, 2.0902178
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8752890, 2.8702669
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1117897, 3.1151414
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6456170, 2.6503916
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3375320, 2.3371811

Time for backsubstitution: 20.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923215, upper bound: 1.1821507
time: 8.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915436, upper bound: 1.1829343
time: 10.07 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 39.37 seconds
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.37
Output dim: 5, lower bound: -1.1923215, upper bound: 1.1821507
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.37
Output dim: 5, lower bound: -1.1915436, upper bound: 1.1829343

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3550653, 3.3549700
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9844341, 2.9780993
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5806398, 3.5772314
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.3911428, 2.4009891
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5608730, 2.5600095
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.0924006, 2.0819077
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8721685, 2.8655972
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1151648, 3.1192093
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6412034, 2.6453481
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3152742, 2.3177054

Time for backsubstitution: 20.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6136

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923214, upper bound: 1.1821385
time: 18.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923185, upper bound: 1.1821508
time: 15.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3542433, 3.3557920
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9796515, 2.9828815
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5780268, 3.5798445
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.3907690, 2.4013624
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5633907, 2.5574908
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.0909138, 2.0833950
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8706198, 2.8671460
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1158571, 3.1185169
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6405730, 2.6459780
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3180552, 2.3149242

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6136

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915436, upper bound: 1.1829216
time: 10.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915399, upper bound: 1.1829344
time: 9.82 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 41.07 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.07
Output dim: 5, lower bound: -1.1923214, upper bound: 1.1821385
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.07
Output dim: 5, lower bound: -1.1923185, upper bound: 1.1821508
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.07
Output dim: 5, lower bound: -1.1915436, upper bound: 1.1829216
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.07
Output dim: 5, lower bound: -1.1915399, upper bound: 1.1829344

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3545980, 3.3557487
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9756145, 2.9703813
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5726709, 3.5707359
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.3987246, 2.4076180
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5621562, 2.5611310
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1051750, 2.0930176
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8499222, 2.8461266
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1045446, 3.1070805
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6475534, 2.6526122
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3191819, 2.3221781

Time for backsubstitution: 20.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5777

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1765390, upper bound: 1.1821254
time: 18.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923070, upper bound: 1.1663736
time: 7.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3558435, 3.3545027
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9767160, 2.9692798
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5741434, 3.5692625
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.3977718, 2.4085710
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5619941, 2.5612931
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1035104, 2.0946825
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8526974, 2.8433518
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1030359, 3.1085892
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6484671, 2.6516979
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3197465, 2.3216136

Time for backsubstitution: 20.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5777

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1765360, upper bound: 1.1821384
time: 8.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923040, upper bound: 1.1663861
time: 9.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3537760, 3.3565707
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9708319, 2.9751639
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5700579, 3.5733490
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.3983507, 2.4079914
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5646758, 2.5586123
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1036878, 2.0945048
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8483734, 2.8476753
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1052370, 3.1063881
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6469221, 2.6532421
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3219628, 2.3193972

Time for backsubstitution: 20.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5777

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1757612, upper bound: 1.1829088
time: 18.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915292, upper bound: 1.1671572
time: 9.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3550215, 3.3553247
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9719334, 2.9740620
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5715303, 3.5718756
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.3973980, 2.4089444
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5645137, 2.5587745
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1020236, 2.0961695
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8511486, 2.8449001
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1037283, 3.1078968
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6478376, 2.6523278
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3225274, 2.3188326

Time for backsubstitution: 20.32 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 63.24 + 543.53 = 606.77 seconds
