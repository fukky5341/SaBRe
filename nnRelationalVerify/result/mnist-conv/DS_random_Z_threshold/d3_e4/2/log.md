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
execution time: IAR + RelationalAnalysis = 23.75 + 35.22 = 58.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.6115657, upper bound: 1.6115677

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 6250
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115520, upper bound: 1.5954354
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954355, upper bound: 1.6115543
time: 4.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.24
Output dim: 4, lower bound: -1.6115520, upper bound: 1.5954354
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.24
Output dim: 4, lower bound: -1.5954355, upper bound: 1.6115543

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5963755, 3.5977683
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3366313, 3.3502622
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8214331, 2.8172858
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2933450, 3.2926850
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6829028, 2.6770763
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.4086351, 2.4064295
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7096357, 2.7138405
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5996289, 2.6027532

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115508, upper bound: 1.5953415
time: 4.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895310, upper bound: 1.5954357
time: 4.97 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5977678, 3.5963759
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3502622, 3.3366313
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8172855, 2.8214333
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2926850, 3.2933445
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6770759, 2.6829031
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.4064293, 2.4086349
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7138405, 2.7096357
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6027532, 2.5996294

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954335, upper bound: 1.5895332
time: 5.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5953416, upper bound: 1.6115507
time: 4.32 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.51 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.51
Output dim: 4, lower bound: -1.6115508, upper bound: 1.5953415
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.51
Output dim: 4, lower bound: -1.5895310, upper bound: 1.5954357
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.51
Output dim: 4, lower bound: -1.5954335, upper bound: 1.5895332
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.51
Output dim: 4, lower bound: -1.5953416, upper bound: 1.6115507

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5971284, 3.5986621
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3347592, 3.3481336
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8268328, 2.8218379
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2942986, 3.2938075
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6831303, 2.6770689
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.4050822, 2.4033213
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7118406, 2.7157040
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6002965, 2.6035447

Time for backsubstitution: 21.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115503, upper bound: 1.5953410
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115503, upper bound: 1.5953410
time: 4.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5972695, 3.5985205
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3345027, 3.3478646
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8259854, 2.8226848
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2941213, 3.2936397
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6828957, 2.6773038
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.4046626, 2.4028771
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7114992, 2.7153554
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6004186, 2.6034203

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895304, upper bound: 1.5954352
time: 4.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5895304, upper bound: 1.5954352
time: 4.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5985208, 3.5972700
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3478646, 3.3345027
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8226848, 2.8259854
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2936387, 3.2941217
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6773033, 2.6828957
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.4028773, 2.4046631
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7153559, 2.7114992
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6034207, 2.6004186

Time for backsubstitution: 21.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954329, upper bound: 1.5895326
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954329, upper bound: 1.5895327
time: 4.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5986619, 3.5971282
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3481336, 3.3347592
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8218379, 2.8268325
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2938085, 3.2942991
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6770687, 2.6831303
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.4033208, 2.4050825
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7157049, 2.7118406
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.6035447, 2.6002965

Time for backsubstitution: 21.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5953411, upper bound: 1.6115503
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5953411, upper bound: 1.6115501
time: 4.48 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.85 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.85
Output dim: 4, lower bound: -1.6115503, upper bound: 1.5953410
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.85
Output dim: 4, lower bound: -1.6115503, upper bound: 1.5953410
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.85
Output dim: 4, lower bound: -1.5895304, upper bound: 1.5954352
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.85
Output dim: 4, lower bound: -1.5895304, upper bound: 1.5954352
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.85
Output dim: 4, lower bound: -1.5954329, upper bound: 1.5895326
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.85
Output dim: 4, lower bound: -1.5954329, upper bound: 1.5895327
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.85
Output dim: 4, lower bound: -1.5953411, upper bound: 1.6115503
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.85
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

Time for backsubstitution: 21.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1509

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1402

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6101341, upper bound: 1.5949975
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6112066, upper bound: 1.5939832
time: 4.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 21.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 1850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2341

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6082532, upper bound: 1.5895860
time: 7.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6057479, upper bound: 1.5920348
time: 4.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 21.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1983

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2334

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5647451, upper bound: 1.5710084
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5647451, upper bound: 1.5710084
time: 4.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1516

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5889082, upper bound: 1.5901832
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5842823, upper bound: 1.5948116
time: 5.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 669

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1983

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5651457, upper bound: 1.5615173
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.5651457, upper bound: 1.5615173
time: 4.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1509

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1501

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5874330, upper bound: 1.5825755
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5883497, upper bound: 1.5818483
time: 4.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 21.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2860

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1432

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5817369, upper bound: 1.6068612
time: 5.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5906674, upper bound: 1.5979354
time: 4.70 seconds

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

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 1248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2488

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5912432, upper bound: 1.6078620
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5916275, upper bound: 1.6074507
time: 4.61 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.6101341, upper bound: 1.5949975
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.6112066, upper bound: 1.5939832
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.6082532, upper bound: 1.5895860
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.6057479, upper bound: 1.5920348
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5647451, upper bound: 1.5710084
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5647451, upper bound: 1.5710084
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5889082, upper bound: 1.5901832
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5842823, upper bound: 1.5948116
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5651457, upper bound: 1.5615173
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5651457, upper bound: 1.5615173
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5874330, upper bound: 1.5825755
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5883497, upper bound: 1.5818483
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5817369, upper bound: 1.6068612
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5906674, upper bound: 1.5979354
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5912432, upper bound: 1.6078620
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.48
Output dim: 4, lower bound: -1.5916275, upper bound: 1.6074507

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5170040, 3.5317967
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3347244, 3.3474011
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8035874, 2.7901914
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2938933, 3.2933578
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6755319, 2.6687298
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3214517, 2.3292172
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6793098, 2.6791859
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5984454, 2.5988436

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1395

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6068594, upper bound: 1.5865299
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016354, upper bound: 1.5917753
time: 4.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5168953, 3.5319188
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3342438, 3.3478847
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.8037100, 2.7900732
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2938733, 3.2933784
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6756949, 2.6685688
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3213706, 2.3293087
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6794043, 2.6790977
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5979800, 2.5993128

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2341
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 2371

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2369

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5990836, upper bound: 1.5930914
time: 5.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6102912, upper bound: 1.5819516
time: 4.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5578976, 3.5437822
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.2897220, 3.2935805
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7907290, 2.7801757
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7496142, 2.7538035
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2536821, 3.2538948
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6538734, 2.6485960
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3426447, 2.3309278
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6584573, 2.6671648
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.1079021, 3.0974798
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5917916, 2.5973058

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1852
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 409
type: DSZ, layer: 3, pos: 2236
type: DSZ, layer: 3, pos: 2333
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2594
type: DSZ, layer: 3, pos: 2369
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1509
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 2488
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 1395
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 1402
type: DSZ, layer: 3, pos: 709
type: DSZ, layer: 3, pos: 2314
type: DSZ, layer: 3, pos: 2860
type: DSZ, layer: 3, pos: 234
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1753
type: DSZ, layer: 3, pos: 170
type: DSZ, layer: 3, pos: 2371
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2537
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1983
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 654
type: DSZ, layer: 3, pos: 2572
type: DSZ, layer: 3, pos: 611
type: DSZ, layer: 3, pos: 1165
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1704
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 1145
type: DSZ, layer: 3, pos: 2132
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 2384
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 317
type: DSZ, layer: 3, pos: 2334
type: DSZ, layer: 3, pos: 2622
type: DSZ, layer: 3, pos: 2642
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 3105
type: DSZ, layer: 3, pos: 1782
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 166
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 2383
type: DSZ, layer: 3, pos: 1384
type: DSZ, layer: 3, pos: 411
type: DSZ, layer: 3, pos: 1199
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1690
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1922

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6038249, upper bound: 1.5806946
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5994135, upper bound: 1.5851065
time: 4.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5556908, 3.5460629
1: -7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.2799916, 3.3036165
2: -10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7898951, 2.7814195
3: -12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7504654, 2.7531400
4: 5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2543631, 3.2533598
5: -8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6538582, 2.6487141
6: -12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3423052, 2.3312755
7: -5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.6592202, 2.6664913
8: -1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.1072478, 3.0986571
9: -6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5916767, 2.5974650

Time for backsubstitution: 22.09 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.97 + 561.53 = 620.50 seconds
