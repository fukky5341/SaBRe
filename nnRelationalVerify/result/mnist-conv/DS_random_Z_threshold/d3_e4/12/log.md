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
execution time: IAR + RelationalAnalysis = 21.82 + 40.22 = 62.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.1923782, upper bound: 1.1923775

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 863

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 915

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923231, upper bound: 1.1920335
time: 11.45 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1920333, upper bound: 1.1923230
time: 7.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 19.41 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 19.41
Output dim: 5, lower bound: -1.1923231, upper bound: 1.1920335
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 19.41
Output dim: 5, lower bound: -1.1920333, upper bound: 1.1923230

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3599796, 3.3600316
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0064430, 3.0075288
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6048069, 3.6056495
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4486599, 2.4485860
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5844808, 2.5848513
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1324103, 2.1324298
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8589954, 2.8583021
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1367435, 3.1376028
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6547155, 2.6546881
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3373952, 2.3376489

Time for backsubstitution: 20.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 4636

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5777

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1765458, upper bound: 1.1920190
time: 19.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923088, upper bound: 1.1762471
time: 11.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3600321, 3.3599796
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0075283, 3.0064440
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6056499, 3.6048069
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4485855, 2.4486604
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5848508, 2.5844812
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1324303, 2.1324100
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8583012, 2.8589969
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1376028, 3.1367435
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6546879, 2.6547155
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3376489, 2.3373957

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1920321, upper bound: 1.1915395
time: 10.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1912499, upper bound: 1.1915396
time: 20.46 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 52.22 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 52.22
Output dim: 5, lower bound: -1.1765458, upper bound: 1.1920190
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 52.22
Output dim: 5, lower bound: -1.1923088, upper bound: 1.1762471
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 52.22
Output dim: 5, lower bound: -1.1920321, upper bound: 1.1915395
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 52.22
Output dim: 5, lower bound: -1.1912499, upper bound: 1.1915396

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3589106, 3.3588109
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9878569, 2.9852738
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6162539, 3.6154432
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4486599, 2.4486420
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5912604, 2.5927687
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.0602019, 2.0692410
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8233252, 2.8270812
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1359320, 3.1366758
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6254106, 2.6211951
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3199596, 2.3177214

Time for backsubstitution: 20.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1715361, upper bound: 1.1920117
time: 6.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1765391, upper bound: 1.1870011
time: 8.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3587580, 3.3589630
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9841890, 2.9889417
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6146011, 3.6170959
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4487162, 2.4485850
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5923982, 2.5916305
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.0692208, 2.0602219
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8277760, 2.8226314
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1358166, 3.1367922
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6212220, 2.6253834
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3174686, 2.3202124

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 863

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1762436, upper bound: 1.1759296
time: 13.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923044, upper bound: 1.1759207
time: 14.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3542728, 3.3533983
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9740200, 2.9681535
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5976954, 3.5942392
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4459667, 2.4456677
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5646892, 2.5668375
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1256063, 2.1240993
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8551817, 2.8543272
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1409769, 3.1408095
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6502743, 2.6496725
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3153920, 2.3179193

Time for backsubstitution: 20.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4636

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1920239, upper bound: 1.1895814
time: 10.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1900734, upper bound: 1.1915312
time: 9.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3534508, 3.3542204
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9692383, 2.9729362
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5950823, 3.5968523
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4455929, 2.4460409
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5672078, 2.5643189
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1241195, 2.1255863
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8536329, 2.8558760
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1416693, 3.1401176
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6496449, 2.6503024
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3181729, 2.3151383

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 5777

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1910060, upper bound: 1.1912985
time: 8.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1910057, upper bound: 1.1919113
time: 22.70 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 52.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 52.07
Output dim: 5, lower bound: -1.1715361, upper bound: 1.1920117
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 52.07
Output dim: 5, lower bound: -1.1765391, upper bound: 1.1870011
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 52.07
Output dim: 5, lower bound: -1.1762436, upper bound: 1.1759296
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 52.07
Output dim: 5, lower bound: -1.1923044, upper bound: 1.1759207
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 52.07
Output dim: 5, lower bound: -1.1920239, upper bound: 1.1895814
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 52.07
Output dim: 5, lower bound: -1.1900734, upper bound: 1.1915312
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 52.07
Output dim: 5, lower bound: -1.1910060, upper bound: 1.1912985
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 52.07
Output dim: 5, lower bound: -1.1910057, upper bound: 1.1919113

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3559408, 3.3573704
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9858990, 2.9843245
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6106062, 3.6127114
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4444156, 2.4398689
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5875659, 2.5909791
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.0514143, 2.0649862
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8176003, 2.8243265
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1348486, 3.1361494
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6216345, 2.6134005
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3184605, 2.3169961

Time for backsubstitution: 20.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 6136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 542

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1715348, upper bound: 1.1912284
time: 8.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1707527, upper bound: 1.1920107
time: 15.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3565435, 3.3570251
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9841843, 2.9894404
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6156397, 3.6183891
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4488997, 2.4485812
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5954742, 2.5941052
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.0549622, 2.0439243
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8214235, 2.8153710
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1363392, 3.1372175
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6137896, 2.6188810
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3133106, 2.3165751

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6136

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923043, upper bound: 1.1759189
time: 6.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1922934, upper bound: 1.1759207
time: 16.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3535843, 3.3536382
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9730186, 2.9684916
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5972404, 3.5943952
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4460325, 2.4454718
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5645413, 2.5668879
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1263909, 2.1217935
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8552637, 2.8540835
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1399736, 3.1411486
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6496811, 2.6498778
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3148198, 2.3181143

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1916089, upper bound: 1.1893375
time: 19.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1910016, upper bound: 1.1893376
time: 9.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3542728, 3.3527098
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9740200, 2.9671512
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5976954, 3.5937848
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4457712, 2.4456677
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5646892, 2.5666900
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1233010, 2.1240993
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8549376, 2.8543272
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1409769, 3.1398058
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6502743, 2.6490793
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3153920, 2.3173470

Time for backsubstitution: 21.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 6136
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1896597, upper bound: 1.1912899
time: 14.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1890518, upper bound: 1.1912882
time: 11.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3534508, 3.3542199
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9692383, 2.9729347
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5950851, 3.5968542
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4455924, 2.4460394
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5672011, 2.5643144
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1241157, 2.1255820
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8536329, 2.8558760
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1416693, 3.1401196
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6496410, 2.6502972
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3181729, 2.3151388

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 4636
type: DSZ, layer: 1, pos: 5777
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 863

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1794547, upper bound: 1.1912946
time: 5.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1910015, upper bound: 1.1797532
time: 5.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3534508, 3.3542199
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9692383, 2.9729357
2: -10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.5950832, 3.5968552
3: -5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4455924, 2.4460402
4: -11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5672030, 2.5643125
5: 6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1241167, 2.1255829
6: -8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8536329, 2.8558764
7: -17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1416712, 3.1401176
8: -6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6496391, 2.6502988
9: -4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3181729, 2.3151383

Time for backsubstitution: 21.15 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 62.04 + 541.84 = 603.88 seconds
