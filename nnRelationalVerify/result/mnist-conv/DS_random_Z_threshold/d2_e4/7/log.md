## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6321674314


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5920982, 1.5920992)
1: (-12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7782269, 1.7782259)
2: (-8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9221311, 1.9221311)
3: (-10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9307184, 1.9307184)
4: (-4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6834741, 1.6834741)
5: (-2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6719646, 1.6719651)
6: (9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.1016169, 1.1016171)
7: (-21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7424874, 1.7424872)
8: (-2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3420033, 1.3420031)
9: (-13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5054874, 1.5054879)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.95 + 49.61 = 74.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.6334342, upper bound: 0.6334355

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5801
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5801

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6333609, upper bound: 0.6334347
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334333, upper bound: 0.6333623
time: 10.62 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.15 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.15
Output dim: 6, lower bound: -0.6333609, upper bound: 0.6334347
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.15
Output dim: 6, lower bound: -0.6334333, upper bound: 0.6333623

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5930090, 1.5916290
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7781410, 1.7783904
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9242325, 1.9210510
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9323053, 1.9299068
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6834359, 1.6835647
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6734028, 1.6712298
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.1014585, 1.1019199
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7461686, 1.7406023
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3422742, 1.3418643
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5058918, 1.5052800

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6168

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6333594, upper bound: 0.6308212
time: 11.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6307491, upper bound: 0.6334318
time: 15.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5916290, 1.5920992
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7782269, 1.7781405
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9210510, 1.9221311
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9299059, 1.9307184
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6834741, 1.6834350
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6712294, 1.6719651
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.1016169, 1.1014583
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7406020, 1.7424872
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3418651, 1.3420031
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5052795, 1.5054879

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 6168
type: DSZ, layer: 1, pos: 6184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6318689, upper bound: 0.6290902
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6291627, upper bound: 0.6317958
time: 6.54 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 35.54 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 35.54
Output dim: 6, lower bound: -0.6333594, upper bound: 0.6308212
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 35.54
Output dim: 6, lower bound: -0.6307491, upper bound: 0.6334318
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 35.54
Output dim: 6, lower bound: -0.6318689, upper bound: 0.6290902
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 35.54
Output dim: 6, lower bound: -0.6291627, upper bound: 0.6317958

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5898266, 1.5900908
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7759361, 1.7773218
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9205847, 1.9192801
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9307299, 1.9291444
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6793656, 1.6815901
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6708307, 1.6659322
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0994005, 1.0976715
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7391543, 1.7372062
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3418946, 1.3410842
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5015306, 1.5031672

Time for backsubstitution: 22.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 932

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329578, upper bound: 0.6285416
time: 6.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6310793, upper bound: 0.6304199
time: 19.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5914707, 1.5884471
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7770729, 1.7761855
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9224606, 1.9174037
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9315424, 1.9283314
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6814599, 1.6794944
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6681051, 1.6686583
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0972099, 1.0998619
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7427726, 1.7335877
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3414941, 1.3414855
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5037794, 1.5009184

Time for backsubstitution: 23.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6307481, upper bound: 0.6326758
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6299924, upper bound: 0.6334305
time: 6.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 35.63 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 35.63
Output dim: 6, lower bound: -0.6329578, upper bound: 0.6285416
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 35.63
Output dim: 6, lower bound: -0.6310793, upper bound: 0.6304199
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 35.63
Output dim: 6, lower bound: -0.6307481, upper bound: 0.6326758
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 35.63
Output dim: 6, lower bound: -0.6299924, upper bound: 0.6334305

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5895672, 1.5897231
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7754250, 1.7765994
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9178047, 1.9173174
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9279022, 1.9251513
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6801667, 1.6815882
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6708288, 1.6662111
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0993972, 1.0981464
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7393985, 1.7372046
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3413644, 1.3403351
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5011196, 1.5028772

Time for backsubstitution: 23.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 6213
type: DSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6297664, upper bound: 0.6256636
time: 5.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6300785, upper bound: 0.6253567
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5881605, 1.5837326
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7554712, 1.7572861
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9050031, 1.8974495
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9361110, 1.9323883
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6546597, 1.6560397
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6549368, 1.6571326
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0948279, 1.0957847
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7337308, 1.7297139
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3475289, 1.3442824
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4879370, 1.4870520

Time for backsubstitution: 23.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6307475, upper bound: 0.6322930
time: 7.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6303633, upper bound: 0.6326751
time: 10.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5867558, 1.5851374
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7581730, 1.7545848
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9025064, 1.8999457
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9355998, 1.9328990
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6580052, 1.6526933
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6565790, 1.6554904
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0931327, 1.0974798
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7388988, 1.7245460
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3442907, 1.3475208
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4899130, 1.4850769

Time for backsubstitution: 22.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 932

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5846

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6299779, upper bound: 0.6293565
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6259186, upper bound: 0.6334163
time: 13.31 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 41.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 41.04
Output dim: 6, lower bound: -0.6297664, upper bound: 0.6256636
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 41.04
Output dim: 6, lower bound: -0.6300785, upper bound: 0.6253567
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.04
Output dim: 6, lower bound: -0.6307475, upper bound: 0.6322930
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.04
Output dim: 6, lower bound: -0.6303633, upper bound: 0.6326751
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 41.04
Output dim: 6, lower bound: -0.6299779, upper bound: 0.6293565
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.04
Output dim: 6, lower bound: -0.6259186, upper bound: 0.6334163

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5874624, 1.5833931
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7549963, 1.7563109
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9036889, 1.8947544
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9357033, 1.9315553
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6533155, 1.6532907
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6499243, 1.6546922
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0937564, 1.0952601
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7320642, 1.7289002
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3473134, 1.3438425
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4819326, 1.4841280

Time for backsubstitution: 23.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 932

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5846

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6307328, upper bound: 0.6282174
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6266737, upper bound: 0.6322772
time: 7.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5878210, 1.5830336
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7544966, 1.7568111
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9023080, 1.8961349
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9352779, 1.9319801
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6519098, 1.6546955
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6524973, 1.6521206
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0943038, 1.0947137
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7329168, 1.7280471
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3470893, 1.3440673
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4850130, 1.4810462

Time for backsubstitution: 23.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 524

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6271754, upper bound: 0.6297912
time: 8.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6274836, upper bound: 0.6294756
time: 8.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5663252, 1.5672626
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7465506, 1.7445536
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8814149, 1.8758383
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.8797522, 1.8846068
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6249571, 1.6123595
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6474667, 1.6475134
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0616262, 1.0699160
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.6643929, 1.6390996
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.2687330, 1.2814138
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4654474, 1.4558501

Time for backsubstitution: 23.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 6113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6259181, upper bound: 0.6330328
time: 7.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6255344, upper bound: 0.6334156
time: 5.39 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 36.56 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 36.56
Output dim: 6, lower bound: -0.6307328, upper bound: 0.6282174
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.56
Output dim: 6, lower bound: -0.6266737, upper bound: 0.6322772
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 36.56
Output dim: 6, lower bound: -0.6271754, upper bound: 0.6297912
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 36.56
Output dim: 6, lower bound: -0.6274836, upper bound: 0.6294756
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.56
Output dim: 6, lower bound: -0.6259181, upper bound: 0.6330328
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.56
Output dim: 6, lower bound: -0.6255344, upper bound: 0.6334156

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5670319, 1.5655189
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7433748, 1.7462807
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8825979, 1.8706479
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.8798566, 1.8832645
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6202669, 1.6129565
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6408110, 1.6467152
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0622504, 1.0676963
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.6575584, 1.6434541
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.2717552, 1.2777350
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4574671, 1.4549007

Time for backsubstitution: 23.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6249618, upper bound: 0.6322762
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6266729, upper bound: 0.6305595
time: 7.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5656271, 1.5669236
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7460766, 1.7435794
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8801012, 1.8731441
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.8793454, 1.8837748
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6236124, 1.6096101
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6424532, 1.6450725
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0605547, 1.0693915
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.6627264, 1.6382861
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.2685165, 1.2809734
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4594412, 1.4529257

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 932

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6227236, upper bound: 0.6301561
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6230378, upper bound: 0.6298466
time: 4.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5659857, 1.5665641
1: -12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7455769, 1.7440791
2: -8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.8787212, 1.8745246
3: -10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.8789201, 1.8842001
4: -4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6222076, 1.6110148
5: -2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6450253, 1.6425014
6: 9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.0611017, 1.0688450
7: -21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.6635795, 1.6374331
8: -2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.2682924, 1.2811983
9: -13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.4625235, 1.4498444

Time for backsubstitution: 22.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 6184
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 6113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 932

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6254048, upper bound: 0.6292321
time: 17.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6254048, upper bound: 0.6330289
time: 4.71 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 45.14 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 45.14
Output dim: 6, lower bound: -0.6249618, upper bound: 0.6322762
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 45.14
Output dim: 6, lower bound: -0.6266729, upper bound: 0.6305595
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 45.14
Output dim: 6, lower bound: -0.6227236, upper bound: 0.6301561
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 45.14
Output dim: 6, lower bound: -0.6230378, upper bound: 0.6298466
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 45.14
Output dim: 6, lower bound: -0.6254048, upper bound: 0.6292321
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 45.14
Output dim: 6, lower bound: -0.6254048, upper bound: 0.6330289

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 74.56 + 526.14 = 600.70 seconds
