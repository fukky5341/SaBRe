## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00167283


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0013695, 0.0000382, -0.0013695, 0.0000382, -0.0009699, 0.0009699)
1: (-0.0077859, -0.0042137, -0.0077859, -0.0042137, -0.0024613, 0.0024613)
2: (0.0301996, 0.0324158, 0.0301996, 0.0324158, -0.0015270, 0.0015270)
3: (-0.0009661, 0.0031721, -0.0009661, 0.0031721, -0.0028513, 0.0028513)
4: (-0.0068126, -0.0031790, -0.0068126, -0.0031790, -0.0025035, 0.0025035)
5: (0.0111578, 0.0125340, 0.0111578, 0.0125340, -0.0009483, 0.0009483)
6: (-0.0008380, 0.0044140, -0.0008380, 0.0044140, -0.0036186, 0.0036186)
7: (0.9774730, 0.9811480, 0.9774730, 0.9811480, -0.0025321, 0.0025321)
8: (-0.0107168, -0.0067766, -0.0107168, -0.0067766, -0.0027148, 0.0027148)
9: (-0.0005233, 0.0020795, -0.0005233, 0.0020795, -0.0017933, 0.0017933)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.83 + 1.70 = 3.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0018429, upper bound: 0.0018429

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018155, upper bound: 0.0017520
time: 0.73 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018155, upper bound: 0.0018155
time: 0.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 7, lower bound: -0.0018155, upper bound: 0.0017520
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 7, lower bound: -0.0018155, upper bound: 0.0018155

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0013292, 0.0000337, -0.0013571, 0.0000368, -0.0009301, 0.0009535
1: -0.0076837, -0.0042251, -0.0077543, -0.0042171, -0.0023602, 0.0024198
2: 0.0302631, 0.0324088, 0.0302192, 0.0324137, -0.0014643, 0.0015012
3: -0.0009529, 0.0030537, -0.0009621, 0.0031356, -0.0028032, 0.0027342
4: -0.0067086, -0.0031906, -0.0067805, -0.0031826, -0.0024007, 0.0024613
5: 0.0111971, 0.0125296, 0.0111699, 0.0125327, -0.0009093, 0.0009323
6: -0.0008212, 0.0042637, -0.0008329, 0.0043676, -0.0035576, 0.0034700
7: 0.9774846, 0.9810428, 0.9774764, 0.9811155, -0.0024894, 0.0024282
8: -0.0107042, -0.0068893, -0.0107130, -0.0068114, -0.0026691, 0.0026034
9: -0.0004488, 0.0020712, -0.0005003, 0.0020770, -0.0017197, 0.0017631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017789, upper bound: 0.0017103
time: 0.72 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017790, upper bound: 0.0017103
time: 0.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0013462, 0.0000748, -0.0013582, 0.0000361, -0.0009395, 0.0009844
1: -0.0077268, -0.0041209, -0.0077573, -0.0042188, -0.0023842, 0.0024981
2: 0.0302363, 0.0324734, 0.0302174, 0.0324126, -0.0014792, 0.0015499
3: -0.0010736, 0.0031038, -0.0009601, 0.0031390, -0.0028940, 0.0027620
4: -0.0067525, -0.0030847, -0.0067835, -0.0031843, -0.0024251, 0.0025410
5: 0.0111805, 0.0125698, 0.0111688, 0.0125320, -0.0009186, 0.0009625
6: -0.0009744, 0.0043272, -0.0008303, 0.0043719, -0.0036728, 0.0035053
7: 0.9773775, 0.9810872, 0.9774782, 0.9811185, -0.0025701, 0.0024529
8: -0.0108192, -0.0068417, -0.0107111, -0.0068081, -0.0027555, 0.0026298
9: -0.0004803, 0.0021471, -0.0005024, 0.0020757, -0.0017372, 0.0018202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017789, upper bound: 0.0017790
time: 0.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017790, upper bound: 0.0017790
time: 0.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 7, lower bound: -0.0017789, upper bound: 0.0017103
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 7, lower bound: -0.0017790, upper bound: 0.0017103
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 7, lower bound: -0.0017789, upper bound: 0.0017790
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 7, lower bound: -0.0017790, upper bound: 0.0017790

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0013222, 0.0000320, -0.0013358, 0.0000317, -0.0009170, 0.0009274
1: -0.0076660, -0.0042294, -0.0077003, -0.0042302, -0.0023271, 0.0023535
2: 0.0302740, 0.0324061, 0.0302528, 0.0324056, -0.0014437, 0.0014601
3: -0.0009478, 0.0030332, -0.0009470, 0.0030730, -0.0027265, 0.0026958
4: -0.0066906, -0.0031951, -0.0067255, -0.0031958, -0.0023670, 0.0023939
5: 0.0112040, 0.0125280, 0.0111907, 0.0125277, -0.0008966, 0.0009068
6: -0.0008148, 0.0042377, -0.0008137, 0.0042881, -0.0034602, 0.0034213
7: 0.9774892, 0.9810246, 0.9774899, 0.9810599, -0.0024213, 0.0023941
8: -0.0106994, -0.0069088, -0.0106986, -0.0068710, -0.0025960, 0.0025668
9: -0.0004359, 0.0020680, -0.0004609, 0.0020674, -0.0016955, 0.0017148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017379, upper bound: 0.0016440
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017536, upper bound: 0.0016828
time: 0.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0013177, 0.0000312, -0.0013298, 0.0000513, -0.0009544, 0.0009267
1: -0.0076544, -0.0042314, -0.0076852, -0.0041803, -0.0024220, 0.0023516
2: 0.0302812, 0.0324048, 0.0302621, 0.0324366, -0.0015026, 0.0014589
3: -0.0009455, 0.0030198, -0.0010048, 0.0030555, -0.0027242, 0.0028057
4: -0.0066788, -0.0031971, -0.0067101, -0.0031451, -0.0024635, 0.0023920
5: 0.0112084, 0.0125272, 0.0111966, 0.0125469, -0.0009331, 0.0009060
6: -0.0008119, 0.0042207, -0.0008870, 0.0042659, -0.0034574, 0.0035608
7: 0.9774911, 0.9810126, 0.9774386, 0.9810444, -0.0024193, 0.0024917
8: -0.0106972, -0.0069216, -0.0107537, -0.0068877, -0.0025939, 0.0026715
9: -0.0004275, 0.0020665, -0.0004499, 0.0021038, -0.0017647, 0.0017134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017409, upper bound: 0.0016450
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017564, upper bound: 0.0016828
time: 0.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013387, 0.0000730, -0.0013362, 0.0000310, -0.0009259, 0.0009575
1: -0.0077076, -0.0041254, -0.0077013, -0.0042319, -0.0023497, 0.0024299
2: 0.0302482, 0.0324706, 0.0302521, 0.0324045, -0.0014578, 0.0015075
3: -0.0010684, 0.0030815, -0.0009450, 0.0030741, -0.0028149, 0.0027220
4: -0.0067330, -0.0030892, -0.0067265, -0.0031976, -0.0023900, 0.0024716
5: 0.0111879, 0.0125681, 0.0111904, 0.0125270, -0.0009053, 0.0009362
6: -0.0009678, 0.0042989, -0.0008112, 0.0042896, -0.0035724, 0.0034546
7: 0.9773820, 0.9810675, 0.9774916, 0.9810609, -0.0024998, 0.0024173
8: -0.0108142, -0.0068629, -0.0106967, -0.0068699, -0.0026802, 0.0025918
9: -0.0004663, 0.0021438, -0.0004616, 0.0020662, -0.0017120, 0.0017704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017378, upper bound: 0.0016944
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017536, upper bound: 0.0017563
time: 0.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013355, 0.0000721, -0.0013320, 0.0000506, -0.0009643, 0.0009573
1: -0.0076997, -0.0041276, -0.0076908, -0.0041821, -0.0024470, 0.0024294
2: 0.0302531, 0.0324693, 0.0302586, 0.0324355, -0.0015181, 0.0015072
3: -0.0010658, 0.0030723, -0.0010027, 0.0030620, -0.0028143, 0.0028347
4: -0.0067249, -0.0030915, -0.0067159, -0.0031469, -0.0024890, 0.0024711
5: 0.0111910, 0.0125672, 0.0111944, 0.0125462, -0.0009428, 0.0009360
6: -0.0009645, 0.0042873, -0.0008845, 0.0042742, -0.0035717, 0.0035976
7: 0.9773843, 0.9810593, 0.9774403, 0.9810502, -0.0024993, 0.0025174
8: -0.0108118, -0.0068716, -0.0107517, -0.0068814, -0.0026797, 0.0026991
9: -0.0004605, 0.0021422, -0.0004540, 0.0021025, -0.0017829, 0.0017701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017409, upper bound: 0.0016944
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017564, upper bound: 0.0017564
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0017379, upper bound: 0.0016440
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0017536, upper bound: 0.0016828
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0017409, upper bound: 0.0016450
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0017564, upper bound: 0.0016828
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0017378, upper bound: 0.0016944
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0017536, upper bound: 0.0017563
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0017409, upper bound: 0.0016944
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 7, lower bound: -0.0017564, upper bound: 0.0017564

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012581, 0.0000371, -0.0013099, 0.0000276, -0.0008409, 0.0008926
1: -0.0075031, -0.0042164, -0.0076345, -0.0042404, -0.0021338, 0.0022652
2: 0.0303751, 0.0324141, 0.0302935, 0.0323992, -0.0013238, 0.0014053
3: -0.0009629, 0.0028445, -0.0009351, 0.0029968, -0.0026241, 0.0024719
4: -0.0065249, -0.0031818, -0.0066586, -0.0032063, -0.0021705, 0.0023041
5: 0.0112667, 0.0125330, 0.0112161, 0.0125237, -0.0008221, 0.0008727
6: -0.0008339, 0.0039982, -0.0007986, 0.0041914, -0.0033303, 0.0031372
7: 0.9774758, 0.9808570, 0.9775005, 0.9809923, -0.0023304, 0.0021953
8: -0.0107138, -0.0070885, -0.0106873, -0.0069435, -0.0024986, 0.0023537
9: -0.0003172, 0.0020775, -0.0004130, 0.0020600, -0.0015547, 0.0016504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0015991
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016258
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0013081, 0.0000287, -0.0013339, 0.0000313, -0.0008721, 0.0009226
1: -0.0076300, -0.0042377, -0.0076955, -0.0042312, -0.0022132, 0.0023413
2: 0.0302963, 0.0324009, 0.0302557, 0.0324049, -0.0013731, 0.0014525
3: -0.0009382, 0.0029916, -0.0009457, 0.0030675, -0.0027123, 0.0025639
4: -0.0066541, -0.0032035, -0.0067207, -0.0031969, -0.0022512, 0.0023815
5: 0.0112178, 0.0125248, 0.0111926, 0.0125273, -0.0008527, 0.0009020
6: -0.0008026, 0.0041849, -0.0008122, 0.0042811, -0.0034422, 0.0032539
7: 0.9774977, 0.9809875, 0.9774910, 0.9810550, -0.0024087, 0.0022769
8: -0.0106903, -0.0069485, -0.0106975, -0.0068762, -0.0025825, 0.0024412
9: -0.0004097, 0.0020620, -0.0004574, 0.0020667, -0.0016126, 0.0017059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016265
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016641
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012520, 0.0000362, -0.0013025, 0.0000473, -0.0008787, 0.0008917
1: -0.0074878, -0.0042187, -0.0076158, -0.0041905, -0.0022298, 0.0022629
2: 0.0303846, 0.0324127, 0.0303051, 0.0324302, -0.0013834, 0.0014039
3: -0.0009603, 0.0028269, -0.0009930, 0.0029752, -0.0026215, 0.0025831
4: -0.0065094, -0.0031841, -0.0066396, -0.0031554, -0.0022680, 0.0023018
5: 0.0112726, 0.0125321, 0.0112233, 0.0125430, -0.0008591, 0.0008718
6: -0.0008306, 0.0039758, -0.0008721, 0.0041640, -0.0033270, 0.0032783
7: 0.9774780, 0.9808413, 0.9774490, 0.9809729, -0.0023281, 0.0022940
8: -0.0107113, -0.0071054, -0.0107424, -0.0069641, -0.0024960, 0.0024595
9: -0.0003061, 0.0020758, -0.0003994, 0.0020964, -0.0016246, 0.0016488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017231, upper bound: 0.0016031
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017231, upper bound: 0.0016262
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0013041, 0.0000279, -0.0013281, 0.0000509, -0.0009107, 0.0009220
1: -0.0076199, -0.0042397, -0.0076809, -0.0041814, -0.0023111, 0.0023398
2: 0.0303026, 0.0323997, 0.0302648, 0.0324359, -0.0014338, 0.0014516
3: -0.0009360, 0.0029798, -0.0010035, 0.0030505, -0.0027105, 0.0026773
4: -0.0066437, -0.0032055, -0.0067058, -0.0031462, -0.0023508, 0.0023800
5: 0.0112217, 0.0125240, 0.0111982, 0.0125465, -0.0008904, 0.0009015
6: -0.0007997, 0.0041699, -0.0008855, 0.0042597, -0.0034400, 0.0033978
7: 0.9774996, 0.9809771, 0.9774396, 0.9810399, -0.0024072, 0.0023776
8: -0.0106882, -0.0069597, -0.0107525, -0.0068924, -0.0025809, 0.0025492
9: -0.0004023, 0.0020605, -0.0004468, 0.0021030, -0.0016839, 0.0017048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017368, upper bound: 0.0016309
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017368, upper bound: 0.0016641
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012718, 0.0000767, -0.0013101, 0.0000270, -0.0008493, 0.0009213
1: -0.0075379, -0.0041159, -0.0076350, -0.0042421, -0.0021552, 0.0023378
2: 0.0303535, 0.0324765, 0.0302932, 0.0323982, -0.0013371, 0.0014504
3: -0.0010793, 0.0028849, -0.0009332, 0.0029974, -0.0027082, 0.0024967
4: -0.0065603, -0.0030796, -0.0066591, -0.0032079, -0.0021922, 0.0023779
5: 0.0112533, 0.0125717, 0.0112159, 0.0125231, -0.0008304, 0.0009007
6: -0.0009817, 0.0040494, -0.0007962, 0.0041922, -0.0034371, 0.0031687
7: 0.9773723, 0.9808928, 0.9775022, 0.9809928, -0.0024051, 0.0022173
8: -0.0108246, -0.0070501, -0.0106855, -0.0069430, -0.0025787, 0.0023773
9: -0.0003426, 0.0021507, -0.0004134, 0.0020588, -0.0015703, 0.0017034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016483
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016751
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0013252, 0.0000695, -0.0013344, 0.0000306, -0.0008825, 0.0009523
1: -0.0076734, -0.0041342, -0.0076969, -0.0042329, -0.0022396, 0.0024167
2: 0.0302694, 0.0324651, 0.0302548, 0.0324039, -0.0013894, 0.0014993
3: -0.0010581, 0.0030419, -0.0009438, 0.0030691, -0.0027996, 0.0025945
4: -0.0066982, -0.0030982, -0.0067221, -0.0031986, -0.0022780, 0.0024582
5: 0.0112011, 0.0125646, 0.0111920, 0.0125266, -0.0008629, 0.0009311
6: -0.0009547, 0.0042487, -0.0008097, 0.0042832, -0.0035531, 0.0032927
7: 0.9773912, 0.9810323, 0.9774927, 0.9810564, -0.0024863, 0.0023041
8: -0.0108044, -0.0069006, -0.0106956, -0.0068747, -0.0026657, 0.0024703
9: -0.0004413, 0.0021373, -0.0004585, 0.0020654, -0.0016318, 0.0017608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016948
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0017368
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012666, 0.0000758, -0.0013045, 0.0000467, -0.0008875, 0.0009208
1: -0.0075248, -0.0041183, -0.0076210, -0.0041922, -0.0022521, 0.0023367
2: 0.0303616, 0.0324750, 0.0303019, 0.0324292, -0.0013972, 0.0014497
3: -0.0010766, 0.0028697, -0.0009910, 0.0029811, -0.0027070, 0.0026090
4: -0.0065470, -0.0030820, -0.0066448, -0.0031572, -0.0022908, 0.0023768
5: 0.0112584, 0.0125708, 0.0112213, 0.0125423, -0.0008677, 0.0009003
6: -0.0009782, 0.0040301, -0.0008696, 0.0041715, -0.0034355, 0.0033112
7: 0.9773747, 0.9808794, 0.9774507, 0.9809783, -0.0024040, 0.0023170
8: -0.0108221, -0.0070646, -0.0107405, -0.0069585, -0.0025775, 0.0024842
9: -0.0003330, 0.0021490, -0.0004031, 0.0020951, -0.0016409, 0.0017026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017231, upper bound: 0.0016496
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017231, upper bound: 0.0016751
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0013221, 0.0000686, -0.0013304, 0.0000502, -0.0009217, 0.0009522
1: -0.0076656, -0.0041365, -0.0076866, -0.0041831, -0.0023390, 0.0024164
2: 0.0302742, 0.0324637, 0.0302612, 0.0324348, -0.0014511, 0.0014991
3: -0.0010555, 0.0030329, -0.0010015, 0.0030572, -0.0027993, 0.0027096
4: -0.0066903, -0.0031005, -0.0067116, -0.0031480, -0.0023791, 0.0024579
5: 0.0112041, 0.0125638, 0.0111960, 0.0125458, -0.0009012, 0.0009310
6: -0.0009514, 0.0042372, -0.0008829, 0.0042681, -0.0035527, 0.0034388
7: 0.9773936, 0.9810243, 0.9774415, 0.9810459, -0.0024860, 0.0024063
8: -0.0108019, -0.0069092, -0.0107505, -0.0068860, -0.0026654, 0.0025800
9: -0.0004357, 0.0021357, -0.0004510, 0.0021017, -0.0017042, 0.0017606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017368, upper bound: 0.0016948
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017368, upper bound: 0.0017368
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0015991
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016258
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016265
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016641
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017231, upper bound: 0.0016031
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017231, upper bound: 0.0016262
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017368, upper bound: 0.0016309
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017368, upper bound: 0.0016641
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016483
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016751
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016948
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0017368
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017231, upper bound: 0.0016496
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017231, upper bound: 0.0016751
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017368, upper bound: 0.0016948
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 7, lower bound: -0.0017368, upper bound: 0.0017368

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012560, 0.0000251, -0.0013044, -0.0000053, -0.0007913, 0.0008702
1: -0.0074980, -0.0042469, -0.0076207, -0.0043239, -0.0020080, 0.0022082
2: 0.0303783, 0.0323953, 0.0303021, 0.0323474, -0.0012458, 0.0013700
3: -0.0009276, 0.0028386, -0.0008384, 0.0029808, -0.0025581, 0.0023262
4: -0.0065197, -0.0032128, -0.0066446, -0.0032912, -0.0020425, 0.0022461
5: 0.0112687, 0.0125213, 0.0112214, 0.0124916, -0.0007736, 0.0008508
6: -0.0007892, 0.0039907, -0.0006759, 0.0041712, -0.0032466, 0.0029522
7: 0.9775071, 0.9808518, 0.9775863, 0.9809781, -0.0022718, 0.0020658
8: -0.0106802, -0.0070942, -0.0105952, -0.0069587, -0.0024357, 0.0022149
9: -0.0003135, 0.0020553, -0.0004030, 0.0019991, -0.0014631, 0.0016089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017042, upper bound: 0.0015669
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017042, upper bound: 0.0015818
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012566, 0.0000289, -0.0013247, 0.0000066, -0.0007936, 0.0009080
1: -0.0074995, -0.0042372, -0.0076722, -0.0042938, -0.0020138, 0.0023042
2: 0.0303773, 0.0324012, 0.0302701, 0.0323661, -0.0012494, 0.0014295
3: -0.0009388, 0.0028404, -0.0008732, 0.0030405, -0.0026693, 0.0023329
4: -0.0065213, -0.0032030, -0.0066970, -0.0032606, -0.0020484, 0.0023437
5: 0.0112681, 0.0125250, 0.0112015, 0.0125032, -0.0007759, 0.0008877
6: -0.0008033, 0.0039929, -0.0007201, 0.0042469, -0.0033877, 0.0029607
7: 0.9774972, 0.9808534, 0.9775553, 0.9810310, -0.0023705, 0.0020718
8: -0.0106908, -0.0070925, -0.0106284, -0.0069019, -0.0025416, 0.0022213
9: -0.0003146, 0.0020623, -0.0004405, 0.0020211, -0.0014673, 0.0016789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017042, upper bound: 0.0015937
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017042, upper bound: 0.0016091
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013060, 0.0000158, -0.0013285, -0.0000017, -0.0008231, 0.0008991
1: -0.0076248, -0.0042705, -0.0076817, -0.0043149, -0.0020889, 0.0022815
2: 0.0302996, 0.0323806, 0.0302642, 0.0323530, -0.0012959, 0.0014154
3: -0.0009003, 0.0029856, -0.0008488, 0.0030515, -0.0026430, 0.0024198
4: -0.0066487, -0.0032368, -0.0067066, -0.0032820, -0.0021247, 0.0023206
5: 0.0112198, 0.0125122, 0.0111979, 0.0124950, -0.0008048, 0.0008790
6: -0.0007544, 0.0041772, -0.0006891, 0.0042609, -0.0033543, 0.0030711
7: 0.9775314, 0.9809822, 0.9775770, 0.9810408, -0.0023472, 0.0021490
8: -0.0106542, -0.0069542, -0.0106052, -0.0068915, -0.0025165, 0.0023041
9: -0.0004059, 0.0020381, -0.0004474, 0.0020057, -0.0015220, 0.0016623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0015963
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0016094
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013067, 0.0000204, -0.0013500, 0.0000102, -0.0008280, 0.0009383
1: -0.0076265, -0.0042588, -0.0077364, -0.0042847, -0.0021011, 0.0023811
2: 0.0302985, 0.0323879, 0.0302303, 0.0323718, -0.0013036, 0.0014773
3: -0.0009138, 0.0029875, -0.0008838, 0.0031148, -0.0027584, 0.0024341
4: -0.0066505, -0.0032249, -0.0067622, -0.0032513, -0.0021372, 0.0024220
5: 0.0112192, 0.0125167, 0.0111768, 0.0125067, -0.0008095, 0.0009174
6: -0.0007717, 0.0041797, -0.0007336, 0.0043412, -0.0035008, 0.0030892
7: 0.9775193, 0.9809840, 0.9775460, 0.9810970, -0.0024497, 0.0021616
8: -0.0106671, -0.0069524, -0.0106385, -0.0068312, -0.0026265, 0.0023176
9: -0.0004072, 0.0020466, -0.0004872, 0.0020277, -0.0015309, 0.0017349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0016253
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0016475
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012500, 0.0000243, -0.0012967, 0.0000138, -0.0008288, 0.0008702
1: -0.0074826, -0.0042490, -0.0076010, -0.0042755, -0.0021033, 0.0022083
2: 0.0303878, 0.0323939, 0.0303143, 0.0323775, -0.0013049, 0.0013700
3: -0.0009252, 0.0028208, -0.0008945, 0.0029580, -0.0025582, 0.0024365
4: -0.0065041, -0.0032149, -0.0066246, -0.0032419, -0.0021394, 0.0022462
5: 0.0112746, 0.0125204, 0.0112290, 0.0125102, -0.0008103, 0.0008508
6: -0.0007861, 0.0039681, -0.0007471, 0.0041422, -0.0032467, 0.0030923
7: 0.9775092, 0.9808360, 0.9775364, 0.9809578, -0.0022719, 0.0021638
8: -0.0106779, -0.0071111, -0.0106487, -0.0069805, -0.0024358, 0.0023200
9: -0.0003023, 0.0020538, -0.0003886, 0.0020345, -0.0015325, 0.0016090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0015696
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0015858
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012506, 0.0000280, -0.0013206, 0.0000269, -0.0008301, 0.0009084
1: -0.0074841, -0.0042395, -0.0076617, -0.0042423, -0.0021065, 0.0023052
2: 0.0303868, 0.0323998, 0.0302766, 0.0323981, -0.0013069, 0.0014302
3: -0.0009362, 0.0028226, -0.0009329, 0.0030283, -0.0026705, 0.0024403
4: -0.0065057, -0.0032053, -0.0066863, -0.0032081, -0.0021427, 0.0023448
5: 0.0112740, 0.0125241, 0.0112056, 0.0125230, -0.0008116, 0.0008881
6: -0.0008000, 0.0039704, -0.0007959, 0.0042315, -0.0033892, 0.0030970
7: 0.9774995, 0.9808376, 0.9775023, 0.9810203, -0.0023716, 0.0021672
8: -0.0106884, -0.0071094, -0.0106853, -0.0069135, -0.0025427, 0.0023235
9: -0.0003034, 0.0020607, -0.0004328, 0.0020586, -0.0015348, 0.0016796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0015941
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016093
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013020, 0.0000151, -0.0013223, 0.0000174, -0.0008619, 0.0008996
1: -0.0076146, -0.0042723, -0.0076662, -0.0042665, -0.0021872, 0.0022829
2: 0.0303059, 0.0323795, 0.0302739, 0.0323831, -0.0013569, 0.0014163
3: -0.0008982, 0.0029738, -0.0009049, 0.0030335, -0.0026446, 0.0025337
4: -0.0066384, -0.0032387, -0.0066908, -0.0032328, -0.0022247, 0.0023221
5: 0.0112237, 0.0125115, 0.0112039, 0.0125137, -0.0008427, 0.0008795
6: -0.0007518, 0.0041622, -0.0007603, 0.0042380, -0.0033564, 0.0032156
7: 0.9775332, 0.9809718, 0.9775273, 0.9810248, -0.0023486, 0.0022501
8: -0.0106522, -0.0069655, -0.0106586, -0.0069086, -0.0025181, 0.0024125
9: -0.0003985, 0.0020368, -0.0004361, 0.0020410, -0.0015936, 0.0016633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0015984
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016142
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013027, 0.0000196, -0.0013466, 0.0000304, -0.0008652, 0.0009383
1: -0.0076163, -0.0042609, -0.0077277, -0.0042335, -0.0021956, 0.0023811
2: 0.0303048, 0.0323866, 0.0302357, 0.0324035, -0.0013622, 0.0014773
3: -0.0009114, 0.0029757, -0.0009431, 0.0031048, -0.0027584, 0.0025435
4: -0.0066401, -0.0032270, -0.0067534, -0.0031992, -0.0022333, 0.0024220
5: 0.0112231, 0.0125159, 0.0111802, 0.0125264, -0.0008459, 0.0009174
6: -0.0007686, 0.0041647, -0.0008088, 0.0043285, -0.0035008, 0.0032280
7: 0.9775214, 0.9809735, 0.9774933, 0.9810882, -0.0024497, 0.0022588
8: -0.0106648, -0.0069636, -0.0106949, -0.0068407, -0.0026264, 0.0024218
9: -0.0003997, 0.0020451, -0.0004809, 0.0020650, -0.0015997, 0.0017349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016256
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016475
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012699, 0.0000639, -0.0013045, -0.0000059, -0.0007998, 0.0008985
1: -0.0075331, -0.0041483, -0.0076211, -0.0043254, -0.0020296, 0.0022801
2: 0.0303564, 0.0324564, 0.0303019, 0.0323465, -0.0012592, 0.0014146
3: -0.0010418, 0.0028793, -0.0008366, 0.0029812, -0.0026414, 0.0023512
4: -0.0065555, -0.0031125, -0.0066449, -0.0032927, -0.0020644, 0.0023192
5: 0.0112551, 0.0125592, 0.0112213, 0.0124910, -0.0007820, 0.0008785
6: -0.0009341, 0.0040424, -0.0006736, 0.0041717, -0.0033522, 0.0029840
7: 0.9774057, 0.9808879, 0.9775879, 0.9809784, -0.0023457, 0.0020880
8: -0.0107889, -0.0070554, -0.0105935, -0.0069584, -0.0025150, 0.0022387
9: -0.0003391, 0.0021271, -0.0004032, 0.0019980, -0.0014788, 0.0016613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016483
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016483
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012706, 0.0000685, -0.0013258, 0.0000059, -0.0008026, 0.0009377
1: -0.0075349, -0.0041368, -0.0076750, -0.0042956, -0.0020368, 0.0023794
2: 0.0303554, 0.0324636, 0.0302684, 0.0323650, -0.0012637, 0.0014762
3: -0.0010552, 0.0028814, -0.0008712, 0.0030437, -0.0027565, 0.0023596
4: -0.0065573, -0.0031008, -0.0066998, -0.0032623, -0.0020718, 0.0024203
5: 0.0112545, 0.0125637, 0.0112005, 0.0125025, -0.0007847, 0.0009167
6: -0.0009510, 0.0040450, -0.0007175, 0.0042509, -0.0034983, 0.0029946
7: 0.9773938, 0.9808897, 0.9775572, 0.9810339, -0.0024480, 0.0020955
8: -0.0108017, -0.0070535, -0.0106265, -0.0068989, -0.0026246, 0.0022467
9: -0.0003404, 0.0021355, -0.0004425, 0.0020198, -0.0014841, 0.0017337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016751
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016751
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013232, 0.0000564, -0.0013290, -0.0000023, -0.0008335, 0.0009294
1: -0.0076685, -0.0041676, -0.0076831, -0.0043165, -0.0021152, 0.0023586
2: 0.0302725, 0.0324444, 0.0302634, 0.0323521, -0.0013123, 0.0014633
3: -0.0010195, 0.0030361, -0.0008470, 0.0030530, -0.0027323, 0.0024504
4: -0.0066931, -0.0031321, -0.0067080, -0.0032836, -0.0021516, 0.0023990
5: 0.0112030, 0.0125518, 0.0111974, 0.0124944, -0.0008149, 0.0009087
6: -0.0009057, 0.0042414, -0.0006868, 0.0042628, -0.0034676, 0.0031099
7: 0.9774255, 0.9810272, 0.9775786, 0.9810421, -0.0024265, 0.0021761
8: -0.0107677, -0.0069061, -0.0106034, -0.0068900, -0.0026016, 0.0023332
9: -0.0004377, 0.0021131, -0.0004484, 0.0020046, -0.0015412, 0.0017185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016948
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016948
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013240, 0.0000613, -0.0013517, 0.0000095, -0.0008383, 0.0009693
1: -0.0076703, -0.0041550, -0.0077406, -0.0042865, -0.0021273, 0.0024597
2: 0.0302713, 0.0324523, 0.0302277, 0.0323707, -0.0013198, 0.0015260
3: -0.0010341, 0.0030383, -0.0008817, 0.0031197, -0.0028494, 0.0024644
4: -0.0066950, -0.0031193, -0.0067665, -0.0032531, -0.0021638, 0.0025019
5: 0.0112023, 0.0125567, 0.0111752, 0.0125060, -0.0008196, 0.0009477
6: -0.0009243, 0.0042441, -0.0007309, 0.0043475, -0.0036163, 0.0031276
7: 0.9774125, 0.9810291, 0.9775478, 0.9811015, -0.0025305, 0.0021886
8: -0.0107816, -0.0069041, -0.0106365, -0.0068265, -0.0027131, 0.0023465
9: -0.0004391, 0.0021222, -0.0004903, 0.0020264, -0.0015500, 0.0017922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0016956
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017195, upper bound: 0.0017229
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012647, 0.0000631, -0.0012987, 0.0000132, -0.0008378, 0.0008991
1: -0.0075199, -0.0041506, -0.0076063, -0.0042771, -0.0021260, 0.0022815
2: 0.0303646, 0.0324550, 0.0303110, 0.0323765, -0.0013190, 0.0014154
3: -0.0010392, 0.0028641, -0.0008926, 0.0029641, -0.0026430, 0.0024628
4: -0.0065421, -0.0031148, -0.0066299, -0.0032436, -0.0021625, 0.0023207
5: 0.0112602, 0.0125584, 0.0112269, 0.0125096, -0.0008191, 0.0008790
6: -0.0009307, 0.0040230, -0.0007447, 0.0041500, -0.0033543, 0.0031257
7: 0.9774080, 0.9808744, 0.9775381, 0.9809633, -0.0023472, 0.0021872
8: -0.0107864, -0.0070699, -0.0106469, -0.0069747, -0.0025165, 0.0023450
9: -0.0003295, 0.0021255, -0.0003924, 0.0020333, -0.0015490, 0.0016623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016204
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016325
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012654, 0.0000676, -0.0013222, 0.0000262, -0.0008397, 0.0009378
1: -0.0075217, -0.0041390, -0.0076657, -0.0042440, -0.0021308, 0.0023798
2: 0.0303635, 0.0324622, 0.0302742, 0.0323970, -0.0013219, 0.0014764
3: -0.0010526, 0.0028661, -0.0009309, 0.0030330, -0.0027569, 0.0024684
4: -0.0065438, -0.0031030, -0.0066904, -0.0032099, -0.0021673, 0.0024207
5: 0.0112595, 0.0125628, 0.0112041, 0.0125224, -0.0008209, 0.0009169
6: -0.0009478, 0.0040256, -0.0007934, 0.0042373, -0.0034989, 0.0031327
7: 0.9773961, 0.9808763, 0.9775041, 0.9810243, -0.0024483, 0.0021921
8: -0.0107992, -0.0070680, -0.0106834, -0.0069091, -0.0026250, 0.0023503
9: -0.0003308, 0.0021339, -0.0004357, 0.0020574, -0.0015525, 0.0017340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016485
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016590
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013202, 0.0000555, -0.0013247, 0.0000167, -0.0008730, 0.0009302
1: -0.0076607, -0.0041697, -0.0076722, -0.0042682, -0.0022152, 0.0023606
2: 0.0302773, 0.0324431, 0.0302701, 0.0323820, -0.0013743, 0.0014645
3: -0.0010170, 0.0030271, -0.0009029, 0.0030405, -0.0027347, 0.0025663
4: -0.0066852, -0.0031343, -0.0066970, -0.0032345, -0.0022533, 0.0024011
5: 0.0112060, 0.0125510, 0.0112015, 0.0125130, -0.0008535, 0.0009095
6: -0.0009026, 0.0042299, -0.0007578, 0.0042469, -0.0034706, 0.0032569
7: 0.9774277, 0.9810192, 0.9775290, 0.9810310, -0.0024286, 0.0022790
8: -0.0107653, -0.0069147, -0.0106567, -0.0069019, -0.0026038, 0.0024435
9: -0.0004320, 0.0021115, -0.0004405, 0.0020397, -0.0016141, 0.0017200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016611
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016804
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013209, 0.0000604, -0.0013487, 0.0000297, -0.0008762, 0.0009688
1: -0.0076625, -0.0041573, -0.0077331, -0.0042353, -0.0022236, 0.0024586
2: 0.0302762, 0.0324508, 0.0302324, 0.0324024, -0.0013795, 0.0015253
3: -0.0010314, 0.0030292, -0.0009410, 0.0031110, -0.0028481, 0.0025759
4: -0.0066871, -0.0031217, -0.0067589, -0.0032011, -0.0022618, 0.0025008
5: 0.0112053, 0.0125558, 0.0111781, 0.0125257, -0.0008567, 0.0009472
6: -0.0009209, 0.0042326, -0.0008061, 0.0043364, -0.0036147, 0.0032692
7: 0.9774149, 0.9810210, 0.9774951, 0.9810937, -0.0025294, 0.0022876
8: -0.0107790, -0.0069127, -0.0106929, -0.0068348, -0.0027119, 0.0024527
9: -0.0004334, 0.0021206, -0.0004848, 0.0020637, -0.0016201, 0.0017914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016956
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0017229
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.28 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017042, upper bound: 0.0015669
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017042, upper bound: 0.0015818
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017042, upper bound: 0.0015937
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017042, upper bound: 0.0016091
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0015963
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0016094
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0016253
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0016475
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0015696
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0015858
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0015941
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016093
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0015984
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016142
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016256
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016475
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016483
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016483
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016751
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017192, upper bound: 0.0016751
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016948
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017334, upper bound: 0.0016948
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017194, upper bound: 0.0016956
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017195, upper bound: 0.0017229
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016204
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016325
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016485
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017079, upper bound: 0.0016590
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016611
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016804
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0016956
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0017229

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012352, 0.0000148, -0.0012963, -0.0000091, -0.0007688, 0.0008527
1: -0.0074451, -0.0042731, -0.0076002, -0.0043338, -0.0019510, 0.0021637
2: 0.0304110, 0.0323790, 0.0303149, 0.0323413, -0.0012104, 0.0013424
3: -0.0008972, 0.0027774, -0.0008269, 0.0029570, -0.0025066, 0.0022602
4: -0.0064660, -0.0032395, -0.0066237, -0.0033012, -0.0019845, 0.0022009
5: 0.0112890, 0.0125111, 0.0112293, 0.0124878, -0.0007517, 0.0008336
6: -0.0007506, 0.0039130, -0.0006614, 0.0041409, -0.0031812, 0.0028685
7: 0.9775340, 0.9807974, 0.9775965, 0.9809569, -0.0022260, 0.0020072
8: -0.0106513, -0.0071524, -0.0105843, -0.0069814, -0.0023866, 0.0021521
9: -0.0002750, 0.0020362, -0.0003880, 0.0019920, -0.0014216, 0.0015765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012448, 0.0000304, -0.0012992, -0.0000084, -0.0007757, 0.0008709
1: -0.0074695, -0.0042335, -0.0076076, -0.0043320, -0.0019685, 0.0022101
2: 0.0303959, 0.0324035, 0.0303103, 0.0323424, -0.0012213, 0.0013712
3: -0.0009431, 0.0028056, -0.0008290, 0.0029656, -0.0025603, 0.0022804
4: -0.0064907, -0.0031992, -0.0066312, -0.0032994, -0.0020023, 0.0022481
5: 0.0112797, 0.0125264, 0.0112265, 0.0124884, -0.0007584, 0.0008515
6: -0.0008088, 0.0039488, -0.0006640, 0.0041518, -0.0032494, 0.0028942
7: 0.9774932, 0.9808224, 0.9775947, 0.9809645, -0.0022738, 0.0020252
8: -0.0106950, -0.0071256, -0.0105863, -0.0069733, -0.0024378, 0.0021713
9: -0.0002928, 0.0020650, -0.0003934, 0.0019932, -0.0014343, 0.0016103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012358, 0.0000183, -0.0013166, 0.0000026, -0.0007710, 0.0008900
1: -0.0074466, -0.0042641, -0.0076517, -0.0043041, -0.0019564, 0.0022584
2: 0.0304101, 0.0323846, 0.0302829, 0.0323598, -0.0012138, 0.0014011
3: -0.0009077, 0.0027792, -0.0008614, 0.0030167, -0.0026163, 0.0022664
4: -0.0064675, -0.0032303, -0.0066761, -0.0032710, -0.0019900, 0.0022972
5: 0.0112885, 0.0125146, 0.0112095, 0.0124992, -0.0007538, 0.0008701
6: -0.0007638, 0.0039152, -0.0007050, 0.0042167, -0.0033204, 0.0028764
7: 0.9775248, 0.9807990, 0.9775659, 0.9810099, -0.0023234, 0.0020128
8: -0.0106612, -0.0071508, -0.0106171, -0.0069246, -0.0024911, 0.0021580
9: -0.0002761, 0.0020427, -0.0004255, 0.0020136, -0.0014255, 0.0016455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012454, 0.0000340, -0.0013195, 0.0000032, -0.0007783, 0.0009079
1: -0.0074710, -0.0042243, -0.0076591, -0.0043024, -0.0019751, 0.0023040
2: 0.0303950, 0.0324092, 0.0302783, 0.0323608, -0.0012253, 0.0014294
3: -0.0009538, 0.0028074, -0.0008633, 0.0030253, -0.0026691, 0.0022880
4: -0.0064923, -0.0031899, -0.0066836, -0.0032693, -0.0020090, 0.0023436
5: 0.0112791, 0.0125299, 0.0112066, 0.0124999, -0.0007609, 0.0008877
6: -0.0008223, 0.0039511, -0.0007075, 0.0042276, -0.0033874, 0.0029038
7: 0.9774839, 0.9808241, 0.9775642, 0.9810176, -0.0023703, 0.0020319
8: -0.0107051, -0.0071239, -0.0106189, -0.0069164, -0.0025414, 0.0021785
9: -0.0002939, 0.0020717, -0.0004309, 0.0020148, -0.0014390, 0.0016787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012836, 0.0000047, -0.0013202, -0.0000055, -0.0007998, 0.0008808
1: -0.0075680, -0.0042985, -0.0076609, -0.0043246, -0.0020296, 0.0022352
2: 0.0303348, 0.0323632, 0.0302772, 0.0323470, -0.0012592, 0.0013867
3: -0.0008678, 0.0029197, -0.0008375, 0.0030273, -0.0025894, 0.0023512
4: -0.0065909, -0.0032654, -0.0066854, -0.0032919, -0.0020644, 0.0022736
5: 0.0112417, 0.0125013, 0.0112059, 0.0124913, -0.0007819, 0.0008612
6: -0.0007132, 0.0040936, -0.0006748, 0.0042302, -0.0032863, 0.0029839
7: 0.9775602, 0.9809238, 0.9775871, 0.9810194, -0.0022996, 0.0020880
8: -0.0106232, -0.0070169, -0.0105944, -0.0069145, -0.0024655, 0.0022387
9: -0.0003645, 0.0020176, -0.0004322, 0.0019986, -0.0014788, 0.0016286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012957, 0.0000245, -0.0013234, -0.0000048, -0.0008080, 0.0009041
1: -0.0075987, -0.0042485, -0.0076689, -0.0043227, -0.0020504, 0.0022942
2: 0.0303158, 0.0323942, 0.0302722, 0.0323482, -0.0012721, 0.0014233
3: -0.0009257, 0.0029553, -0.0008398, 0.0030366, -0.0026578, 0.0023753
4: -0.0066222, -0.0032145, -0.0066935, -0.0032899, -0.0020856, 0.0023336
5: 0.0112299, 0.0125206, 0.0112028, 0.0124920, -0.0007900, 0.0008839
6: -0.0007867, 0.0041388, -0.0006777, 0.0042420, -0.0033730, 0.0030145
7: 0.9775088, 0.9809554, 0.9775851, 0.9810277, -0.0023603, 0.0021094
8: -0.0106784, -0.0069831, -0.0105966, -0.0069057, -0.0025306, 0.0022616
9: -0.0003869, 0.0020541, -0.0004380, 0.0020000, -0.0014939, 0.0016716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012843, 0.0000093, -0.0013417, 0.0000062, -0.0008043, 0.0009197
1: -0.0075697, -0.0042871, -0.0077154, -0.0042948, -0.0020411, 0.0023340
2: 0.0303338, 0.0323703, 0.0302434, 0.0323655, -0.0012663, 0.0014480
3: -0.0008810, 0.0029217, -0.0008721, 0.0030905, -0.0027038, 0.0023646
4: -0.0065927, -0.0032537, -0.0067409, -0.0032616, -0.0020762, 0.0023740
5: 0.0112411, 0.0125058, 0.0111849, 0.0125028, -0.0007864, 0.0008992
6: -0.0007300, 0.0040961, -0.0007186, 0.0043104, -0.0034315, 0.0030009
7: 0.9775484, 0.9809256, 0.9775563, 0.9810754, -0.0024012, 0.0020999
8: -0.0106358, -0.0070151, -0.0106273, -0.0068543, -0.0025744, 0.0022514
9: -0.0003658, 0.0020260, -0.0004719, 0.0020203, -0.0014872, 0.0017006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012964, 0.0000291, -0.0013450, 0.0000069, -0.0008127, 0.0009442
1: -0.0076003, -0.0042368, -0.0077237, -0.0042931, -0.0020623, 0.0023961
2: 0.0303148, 0.0324015, 0.0302382, 0.0323666, -0.0012795, 0.0014865
3: -0.0009393, 0.0029572, -0.0008741, 0.0031001, -0.0027757, 0.0023891
4: -0.0066238, -0.0032026, -0.0067493, -0.0032598, -0.0020977, 0.0024372
5: 0.0112293, 0.0125251, 0.0111817, 0.0125035, -0.0007946, 0.0009231
6: -0.0008039, 0.0041411, -0.0007212, 0.0043225, -0.0035228, 0.0030320
7: 0.9774967, 0.9809570, 0.9775546, 0.9810840, -0.0024651, 0.0021217
8: -0.0106913, -0.0069813, -0.0106293, -0.0068452, -0.0026429, 0.0022748
9: -0.0003881, 0.0020626, -0.0004780, 0.0020216, -0.0015026, 0.0017458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012291, 0.0000140, -0.0012884, 0.0000099, -0.0008057, 0.0008523
1: -0.0074297, -0.0042752, -0.0075802, -0.0042855, -0.0020445, 0.0021629
2: 0.0304206, 0.0323777, 0.0303272, 0.0323713, -0.0012684, 0.0013419
3: -0.0008948, 0.0027595, -0.0008828, 0.0029339, -0.0025056, 0.0023684
4: -0.0064503, -0.0032416, -0.0066033, -0.0032521, -0.0020796, 0.0022000
5: 0.0112950, 0.0125104, 0.0112370, 0.0125064, -0.0007877, 0.0008333
6: -0.0007475, 0.0038903, -0.0007323, 0.0041116, -0.0031800, 0.0030058
7: 0.9775361, 0.9807816, 0.9775468, 0.9809363, -0.0022252, 0.0021033
8: -0.0106490, -0.0071695, -0.0106376, -0.0070035, -0.0023857, 0.0022551
9: -0.0002637, 0.0020347, -0.0003734, 0.0020271, -0.0014896, 0.0015759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015696
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015696
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012389, 0.0000296, -0.0012915, 0.0000107, -0.0008128, 0.0008711
1: -0.0074543, -0.0042354, -0.0075879, -0.0042833, -0.0020626, 0.0022105
2: 0.0304053, 0.0324023, 0.0303224, 0.0323726, -0.0012797, 0.0013714
3: -0.0009409, 0.0027881, -0.0008854, 0.0029429, -0.0025608, 0.0023894
4: -0.0064753, -0.0032012, -0.0066112, -0.0032499, -0.0020980, 0.0022485
5: 0.0112855, 0.0125257, 0.0112340, 0.0125072, -0.0007947, 0.0008517
6: -0.0008060, 0.0039266, -0.0007355, 0.0041230, -0.0032500, 0.0030325
7: 0.9774953, 0.9808068, 0.9775446, 0.9809443, -0.0022742, 0.0021220
8: -0.0106928, -0.0071423, -0.0106400, -0.0069949, -0.0024383, 0.0022751
9: -0.0002817, 0.0020636, -0.0003791, 0.0020287, -0.0015028, 0.0016106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015858
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015858
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012297, 0.0000174, -0.0013124, 0.0000228, -0.0008069, 0.0008900
1: -0.0074312, -0.0042663, -0.0076409, -0.0042527, -0.0020475, 0.0022586
2: 0.0304197, 0.0323832, 0.0302896, 0.0323916, -0.0012703, 0.0014012
3: -0.0009051, 0.0027613, -0.0009208, 0.0030042, -0.0026164, 0.0023720
4: -0.0064518, -0.0032326, -0.0066651, -0.0032188, -0.0020827, 0.0022973
5: 0.0112944, 0.0125137, 0.0112136, 0.0125190, -0.0007889, 0.0008702
6: -0.0007605, 0.0038925, -0.0007805, 0.0042008, -0.0033206, 0.0030104
7: 0.9775271, 0.9807830, 0.9775131, 0.9809988, -0.0023236, 0.0021065
8: -0.0106587, -0.0071678, -0.0106737, -0.0069365, -0.0024912, 0.0022585
9: -0.0002649, 0.0020411, -0.0004176, 0.0020510, -0.0014919, 0.0016456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015941
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015941
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012395, 0.0000332, -0.0013153, 0.0000236, -0.0008145, 0.0009084
1: -0.0074559, -0.0042264, -0.0076484, -0.0042506, -0.0020668, 0.0023052
2: 0.0304043, 0.0324080, 0.0302849, 0.0323929, -0.0012823, 0.0014302
3: -0.0009513, 0.0027899, -0.0009233, 0.0030129, -0.0026705, 0.0023943
4: -0.0064769, -0.0031920, -0.0066728, -0.0032166, -0.0021023, 0.0023448
5: 0.0112849, 0.0125291, 0.0112107, 0.0125198, -0.0007963, 0.0008882
6: -0.0008193, 0.0039289, -0.0007836, 0.0042119, -0.0033892, 0.0030387
7: 0.9774860, 0.9808085, 0.9775109, 0.9810066, -0.0023716, 0.0021263
8: -0.0107028, -0.0071405, -0.0106761, -0.0069282, -0.0025427, 0.0022798
9: -0.0002829, 0.0020702, -0.0004231, 0.0020525, -0.0015059, 0.0016796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0016093
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0016093
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012797, 0.0000040, -0.0013140, 0.0000135, -0.0008376, 0.0008810
1: -0.0075580, -0.0043004, -0.0076449, -0.0042763, -0.0021255, 0.0022357
2: 0.0303410, 0.0323621, 0.0302871, 0.0323770, -0.0013187, 0.0013871
3: -0.0008657, 0.0029082, -0.0008935, 0.0030088, -0.0025900, 0.0024623
4: -0.0065808, -0.0032672, -0.0066692, -0.0032428, -0.0021620, 0.0022741
5: 0.0112456, 0.0125006, 0.0112121, 0.0125099, -0.0008189, 0.0008614
6: -0.0007105, 0.0040790, -0.0007459, 0.0042067, -0.0032870, 0.0031249
7: 0.9775621, 0.9809135, 0.9775374, 0.9810029, -0.0023001, 0.0021867
8: -0.0106212, -0.0070279, -0.0106477, -0.0069321, -0.0024661, 0.0023445
9: -0.0003573, 0.0020163, -0.0004206, 0.0020338, -0.0015486, 0.0016290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0015984
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0015984
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012916, 0.0000238, -0.0013172, 0.0000143, -0.0008462, 0.0009050
1: -0.0075882, -0.0042503, -0.0076532, -0.0042743, -0.0021474, 0.0022967
2: 0.0303223, 0.0323931, 0.0302819, 0.0323782, -0.0013322, 0.0014249
3: -0.0009236, 0.0029432, -0.0008959, 0.0030185, -0.0026606, 0.0024876
4: -0.0066115, -0.0032163, -0.0066776, -0.0032407, -0.0021842, 0.0023361
5: 0.0112339, 0.0125199, 0.0112089, 0.0125107, -0.0008273, 0.0008849
6: -0.0007841, 0.0041234, -0.0007489, 0.0042190, -0.0033766, 0.0031571
7: 0.9775106, 0.9809446, 0.9775352, 0.9810115, -0.0023628, 0.0022092
8: -0.0106764, -0.0069946, -0.0106500, -0.0069229, -0.0025333, 0.0023686
9: -0.0003793, 0.0020528, -0.0004266, 0.0020353, -0.0015646, 0.0016734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016142
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016142
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012804, 0.0000084, -0.0013383, 0.0000263, -0.0008408, 0.0009196
1: -0.0075597, -0.0042892, -0.0077067, -0.0042438, -0.0021337, 0.0023336
2: 0.0303400, 0.0323690, 0.0302487, 0.0323972, -0.0013237, 0.0014478
3: -0.0008786, 0.0029101, -0.0009312, 0.0030805, -0.0027033, 0.0024717
4: -0.0065825, -0.0032559, -0.0067321, -0.0032097, -0.0021703, 0.0023736
5: 0.0112449, 0.0125049, 0.0111882, 0.0125224, -0.0008220, 0.0008991
6: -0.0007269, 0.0040814, -0.0007937, 0.0042976, -0.0034309, 0.0031370
7: 0.9775506, 0.9809152, 0.9775039, 0.9810665, -0.0024008, 0.0021951
8: -0.0106335, -0.0070261, -0.0106836, -0.0068639, -0.0025740, 0.0023535
9: -0.0003585, 0.0020244, -0.0004656, 0.0020575, -0.0015546, 0.0017003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016256
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016256
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012922, 0.0000283, -0.0013415, 0.0000271, -0.0008496, 0.0009442
1: -0.0075898, -0.0042389, -0.0077149, -0.0042418, -0.0021559, 0.0023960
2: 0.0303213, 0.0324002, 0.0302437, 0.0323984, -0.0013375, 0.0014865
3: -0.0009369, 0.0029450, -0.0009335, 0.0030899, -0.0027757, 0.0024975
4: -0.0066131, -0.0032047, -0.0067403, -0.0032076, -0.0021929, 0.0024372
5: 0.0112333, 0.0125243, 0.0111851, 0.0125232, -0.0008306, 0.0009231
6: -0.0008009, 0.0041257, -0.0007966, 0.0043096, -0.0035227, 0.0031697
7: 0.9774988, 0.9809462, 0.9775018, 0.9810749, -0.0024650, 0.0022180
8: -0.0106890, -0.0069929, -0.0106858, -0.0068549, -0.0026429, 0.0023780
9: -0.0003804, 0.0020611, -0.0004715, 0.0020590, -0.0015708, 0.0017458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016475
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016475
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012569, 0.0000602, -0.0013045, -0.0000059, -0.0007850, 0.0008951
1: -0.0075001, -0.0041578, -0.0076211, -0.0043254, -0.0019921, 0.0022715
2: 0.0303770, 0.0324505, 0.0303019, 0.0323465, -0.0012359, 0.0014093
3: -0.0010308, 0.0028410, -0.0008366, 0.0029812, -0.0026315, 0.0023078
4: -0.0065218, -0.0031222, -0.0066449, -0.0032927, -0.0020263, 0.0023105
5: 0.0112679, 0.0125556, 0.0112213, 0.0124910, -0.0007675, 0.0008752
6: -0.0009201, 0.0039938, -0.0006736, 0.0041717, -0.0033397, 0.0029289
7: 0.9774154, 0.9808539, 0.9775879, 0.9809784, -0.0023369, 0.0020495
8: -0.0107785, -0.0070919, -0.0105935, -0.0069584, -0.0025056, 0.0021974
9: -0.0003150, 0.0021202, -0.0004032, 0.0019980, -0.0014515, 0.0016551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016483
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016483
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012493, 0.0000747, -0.0013045, -0.0000059, -0.0007913, 0.0009363
1: -0.0074809, -0.0041210, -0.0076211, -0.0043254, -0.0020081, 0.0023761
2: 0.0303889, 0.0324733, 0.0303019, 0.0323465, -0.0012459, 0.0014741
3: -0.0010734, 0.0028188, -0.0008366, 0.0029812, -0.0027526, 0.0023263
4: -0.0065023, -0.0030848, -0.0066449, -0.0032927, -0.0020426, 0.0024169
5: 0.0112753, 0.0125697, 0.0112213, 0.0124910, -0.0007737, 0.0009155
6: -0.0009741, 0.0039656, -0.0006736, 0.0041717, -0.0034934, 0.0029524
7: 0.9773776, 0.9808342, 0.9775879, 0.9809784, -0.0024445, 0.0020660
8: -0.0108190, -0.0071130, -0.0105935, -0.0069584, -0.0026209, 0.0022150
9: -0.0003011, 0.0021470, -0.0004032, 0.0019980, -0.0014632, 0.0017313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016483
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016483
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012576, 0.0000648, -0.0013258, 0.0000059, -0.0007879, 0.0009344
1: -0.0075018, -0.0041461, -0.0076750, -0.0042956, -0.0019993, 0.0023711
2: 0.0303759, 0.0324578, 0.0302684, 0.0323650, -0.0012404, 0.0014710
3: -0.0010443, 0.0028431, -0.0008712, 0.0030437, -0.0027468, 0.0023161
4: -0.0065236, -0.0031103, -0.0066998, -0.0032623, -0.0020336, 0.0024118
5: 0.0112672, 0.0125601, 0.0112005, 0.0125025, -0.0007703, 0.0009135
6: -0.0009373, 0.0039964, -0.0007175, 0.0042509, -0.0034860, 0.0029394
7: 0.9774034, 0.9808557, 0.9775572, 0.9810339, -0.0024394, 0.0020569
8: -0.0107913, -0.0070899, -0.0106265, -0.0068989, -0.0026154, 0.0022053
9: -0.0003163, 0.0021287, -0.0004425, 0.0020198, -0.0014567, 0.0017276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016751
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016751
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012500, 0.0000798, -0.0013258, 0.0000059, -0.0007943, 0.0009741
1: -0.0074827, -0.0041082, -0.0076750, -0.0042956, -0.0020156, 0.0024719
2: 0.0303877, 0.0324813, 0.0302684, 0.0323650, -0.0012505, 0.0015336
3: -0.0010883, 0.0028209, -0.0008712, 0.0030437, -0.0028636, 0.0023350
4: -0.0065042, -0.0030717, -0.0066998, -0.0032623, -0.0020502, 0.0025144
5: 0.0112746, 0.0125747, 0.0112005, 0.0125025, -0.0007766, 0.0009524
6: -0.0009930, 0.0039683, -0.0007175, 0.0042509, -0.0036343, 0.0029635
7: 0.9773644, 0.9808360, 0.9775572, 0.9810339, -0.0025431, 0.0020737
8: -0.0108332, -0.0071110, -0.0106265, -0.0068989, -0.0027266, 0.0022233
9: -0.0003024, 0.0021563, -0.0004425, 0.0020198, -0.0014686, 0.0018011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016751
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016751
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0013082, 0.0000527, -0.0013290, -0.0000023, -0.0008175, 0.0009262
1: -0.0076304, -0.0041768, -0.0076831, -0.0043165, -0.0020745, 0.0023504
2: 0.0302961, 0.0324387, 0.0302634, 0.0323521, -0.0012870, 0.0014582
3: -0.0010088, 0.0029920, -0.0008470, 0.0030530, -0.0027228, 0.0024032
4: -0.0066544, -0.0031415, -0.0067080, -0.0032836, -0.0021101, 0.0023907
5: 0.0112177, 0.0125482, 0.0111974, 0.0124944, -0.0007992, 0.0009055
6: -0.0008922, 0.0041854, -0.0006868, 0.0042628, -0.0034556, 0.0030499
7: 0.9774350, 0.9809880, 0.9775786, 0.9810421, -0.0024181, 0.0021342
8: -0.0107575, -0.0069481, -0.0106034, -0.0068900, -0.0025925, 0.0022882
9: -0.0004100, 0.0021063, -0.0004484, 0.0020046, -0.0015115, 0.0017125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016807, upper bound: 0.0016808
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016807, upper bound: 0.0016948
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0013056, 0.0000736, -0.0013290, -0.0000023, -0.0008247, 0.0009678
1: -0.0076238, -0.0041237, -0.0076831, -0.0043165, -0.0020927, 0.0024560
2: 0.0303002, 0.0324717, 0.0302634, 0.0323521, -0.0012983, 0.0015237
3: -0.0010703, 0.0029843, -0.0008470, 0.0030530, -0.0028451, 0.0024243
4: -0.0066477, -0.0030875, -0.0067080, -0.0032836, -0.0021286, 0.0024982
5: 0.0112202, 0.0125687, 0.0111974, 0.0124944, -0.0008063, 0.0009462
6: -0.0009703, 0.0041756, -0.0006868, 0.0042628, -0.0036109, 0.0030767
7: 0.9773803, 0.9809812, 0.9775786, 0.9810421, -0.0025267, 0.0021529
8: -0.0108161, -0.0069554, -0.0106034, -0.0068900, -0.0027090, 0.0023083
9: -0.0004052, 0.0021450, -0.0004484, 0.0020046, -0.0015247, 0.0017895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016807, upper bound: 0.0016808
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016807, upper bound: 0.0016948
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0013009, 0.0000501, -0.0013431, 0.0000055, -0.0008143, 0.0009498
1: -0.0076119, -0.0041835, -0.0077188, -0.0042966, -0.0020665, 0.0024103
2: 0.0303076, 0.0324346, 0.0302412, 0.0323644, -0.0012821, 0.0014953
3: -0.0010011, 0.0029706, -0.0008700, 0.0030945, -0.0027922, 0.0023940
4: -0.0066356, -0.0031483, -0.0067444, -0.0032634, -0.0021020, 0.0024516
5: 0.0112248, 0.0125457, 0.0111836, 0.0125021, -0.0007962, 0.0009286
6: -0.0008824, 0.0041582, -0.0007160, 0.0043154, -0.0035436, 0.0030382
7: 0.9774418, 0.9809690, 0.9775583, 0.9810789, -0.0024797, 0.0021260
8: -0.0107501, -0.0069685, -0.0106253, -0.0068505, -0.0026586, 0.0022794
9: -0.0003965, 0.0021015, -0.0004744, 0.0020190, -0.0015057, 0.0017561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016815
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016956
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0013136, 0.0000702, -0.0013468, 0.0000062, -0.0008228, 0.0009753
1: -0.0076441, -0.0041324, -0.0077282, -0.0042947, -0.0020880, 0.0024750
2: 0.0302876, 0.0324662, 0.0302354, 0.0323655, -0.0012954, 0.0015355
3: -0.0010602, 0.0030079, -0.0008722, 0.0031053, -0.0028672, 0.0024189
4: -0.0066684, -0.0030964, -0.0067539, -0.0032615, -0.0021239, 0.0025175
5: 0.0112124, 0.0125653, 0.0111800, 0.0125028, -0.0008045, 0.0009536
6: -0.0009574, 0.0042056, -0.0007188, 0.0043291, -0.0036389, 0.0030699
7: 0.9773893, 0.9810021, 0.9775563, 0.9810886, -0.0025463, 0.0021481
8: -0.0108064, -0.0069330, -0.0106274, -0.0068402, -0.0027300, 0.0023031
9: -0.0004200, 0.0021387, -0.0004812, 0.0020204, -0.0015214, 0.0018033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017079
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017229
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012440, 0.0000517, -0.0012905, 0.0000092, -0.0008140, 0.0008799
1: -0.0074673, -0.0041793, -0.0075854, -0.0042872, -0.0020657, 0.0022329
2: 0.0303973, 0.0324371, 0.0303240, 0.0323703, -0.0012816, 0.0013853
3: -0.0010059, 0.0028031, -0.0008810, 0.0029398, -0.0025868, 0.0023930
4: -0.0064885, -0.0031441, -0.0066086, -0.0032538, -0.0021012, 0.0022713
5: 0.0112805, 0.0125473, 0.0112350, 0.0125057, -0.0007959, 0.0008603
6: -0.0008884, 0.0039456, -0.0007299, 0.0041192, -0.0032829, 0.0030371
7: 0.9774377, 0.9808202, 0.9775485, 0.9809417, -0.0022972, 0.0021252
8: -0.0107547, -0.0071280, -0.0106358, -0.0069978, -0.0024630, 0.0022786
9: -0.0002911, 0.0021045, -0.0003772, 0.0020259, -0.0015051, 0.0016269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016204
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016204
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012532, 0.0000686, -0.0012936, 0.0000101, -0.0008218, 0.0009010
1: -0.0074907, -0.0041365, -0.0075933, -0.0042849, -0.0020854, 0.0022865
2: 0.0303828, 0.0324637, 0.0303191, 0.0323716, -0.0012938, 0.0014186
3: -0.0010554, 0.0028302, -0.0008836, 0.0029491, -0.0026488, 0.0024158
4: -0.0065123, -0.0031006, -0.0066167, -0.0032515, -0.0021212, 0.0023258
5: 0.0112715, 0.0125638, 0.0112319, 0.0125066, -0.0008035, 0.0008809
6: -0.0009514, 0.0039800, -0.0007332, 0.0041309, -0.0033617, 0.0030660
7: 0.9773935, 0.9808443, 0.9775462, 0.9809499, -0.0023524, 0.0021455
8: -0.0108019, -0.0071022, -0.0106382, -0.0069890, -0.0025221, 0.0023003
9: -0.0003082, 0.0021357, -0.0003830, 0.0020276, -0.0015195, 0.0016660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016325
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016325
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012446, 0.0000563, -0.0013139, 0.0000221, -0.0008158, 0.0009189
1: -0.0074690, -0.0041677, -0.0076448, -0.0042545, -0.0020703, 0.0023317
2: 0.0303962, 0.0324444, 0.0302872, 0.0323905, -0.0012844, 0.0014466
3: -0.0010193, 0.0028051, -0.0009188, 0.0030087, -0.0027012, 0.0023983
4: -0.0064903, -0.0031323, -0.0066690, -0.0032205, -0.0021058, 0.0023717
5: 0.0112798, 0.0125518, 0.0112121, 0.0125183, -0.0007976, 0.0008984
6: -0.0009055, 0.0039481, -0.0007780, 0.0042065, -0.0034282, 0.0030438
7: 0.9774257, 0.9808220, 0.9775149, 0.9810027, -0.0023989, 0.0021299
8: -0.0107675, -0.0071261, -0.0106718, -0.0069322, -0.0025720, 0.0022836
9: -0.0002924, 0.0021130, -0.0004205, 0.0020497, -0.0015084, 0.0016989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016485
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016484
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012539, 0.0000727, -0.0013171, 0.0000230, -0.0008238, 0.0009395
1: -0.0074925, -0.0041260, -0.0076530, -0.0042523, -0.0020904, 0.0023840
2: 0.0303816, 0.0324702, 0.0302821, 0.0323919, -0.0012969, 0.0014791
3: -0.0010676, 0.0028323, -0.0009213, 0.0030182, -0.0027618, 0.0024217
4: -0.0065142, -0.0030899, -0.0066774, -0.0032183, -0.0021263, 0.0024250
5: 0.0112708, 0.0125678, 0.0112090, 0.0125192, -0.0008054, 0.0009185
6: -0.0009668, 0.0039827, -0.0007812, 0.0042186, -0.0035051, 0.0030734
7: 0.9773827, 0.9808461, 0.9775126, 0.9810112, -0.0024527, 0.0021506
8: -0.0108135, -0.0071002, -0.0106742, -0.0069232, -0.0026297, 0.0023058
9: -0.0003095, 0.0021433, -0.0004264, 0.0020513, -0.0015231, 0.0017370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016590
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016590
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012973, 0.0000443, -0.0013161, 0.0000128, -0.0008484, 0.0009108
1: -0.0076026, -0.0041982, -0.0076505, -0.0042780, -0.0021530, 0.0023113
2: 0.0303133, 0.0324255, 0.0302836, 0.0323759, -0.0013357, 0.0014340
3: -0.0009840, 0.0029598, -0.0008916, 0.0030153, -0.0026776, 0.0024942
4: -0.0066261, -0.0031633, -0.0066748, -0.0032445, -0.0021900, 0.0023510
5: 0.0112284, 0.0125400, 0.0112099, 0.0125093, -0.0008295, 0.0008905
6: -0.0008608, 0.0041445, -0.0007434, 0.0042149, -0.0033982, 0.0031654
7: 0.9774570, 0.9809594, 0.9775391, 0.9810086, -0.0023779, 0.0022150
8: -0.0107339, -0.0069788, -0.0106459, -0.0069259, -0.0025495, 0.0023748
9: -0.0003897, 0.0020908, -0.0004246, 0.0020326, -0.0015687, 0.0016841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016466
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016611
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0013098, 0.0000647, -0.0013197, 0.0000137, -0.0008571, 0.0009357
1: -0.0076343, -0.0041465, -0.0076596, -0.0042759, -0.0021750, 0.0023745
2: 0.0302937, 0.0324575, 0.0302780, 0.0323773, -0.0013494, 0.0014731
3: -0.0010439, 0.0029965, -0.0008940, 0.0030259, -0.0027507, 0.0025196
4: -0.0066584, -0.0031107, -0.0066841, -0.0032423, -0.0022123, 0.0024152
5: 0.0112162, 0.0125599, 0.0112064, 0.0125101, -0.0008380, 0.0009148
6: -0.0009368, 0.0041911, -0.0007465, 0.0042283, -0.0034910, 0.0031977
7: 0.9774037, 0.9809920, 0.9775370, 0.9810181, -0.0024428, 0.0022376
8: -0.0107910, -0.0069438, -0.0106482, -0.0069159, -0.0026191, 0.0023990
9: -0.0004128, 0.0021284, -0.0004313, 0.0020342, -0.0015847, 0.0017301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016648
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016804
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012980, 0.0000491, -0.0013402, 0.0000256, -0.0008516, 0.0009492
1: -0.0076044, -0.0041859, -0.0077115, -0.0042456, -0.0021611, 0.0024088
2: 0.0303122, 0.0324331, 0.0302458, 0.0323960, -0.0013407, 0.0014944
3: -0.0009983, 0.0029619, -0.0009291, 0.0030859, -0.0027905, 0.0025035
4: -0.0066279, -0.0031508, -0.0067369, -0.0032115, -0.0021982, 0.0024502
5: 0.0112277, 0.0125448, 0.0111864, 0.0125218, -0.0008326, 0.0009281
6: -0.0008788, 0.0041471, -0.0007910, 0.0043046, -0.0035415, 0.0031773
7: 0.9774443, 0.9809612, 0.9775057, 0.9810714, -0.0024782, 0.0022233
8: -0.0107475, -0.0069768, -0.0106816, -0.0068587, -0.0026570, 0.0023837
9: -0.0003910, 0.0020997, -0.0004691, 0.0020562, -0.0015746, 0.0017551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016815
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016956
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0013105, 0.0000693, -0.0013437, 0.0000264, -0.0008603, 0.0009750
1: -0.0076361, -0.0041347, -0.0077203, -0.0042435, -0.0021831, 0.0024741
2: 0.0302925, 0.0324648, 0.0302403, 0.0323974, -0.0013544, 0.0015350
3: -0.0010575, 0.0029986, -0.0009316, 0.0030962, -0.0028662, 0.0025290
4: -0.0066602, -0.0030987, -0.0067459, -0.0032093, -0.0022205, 0.0025166
5: 0.0112155, 0.0125645, 0.0111830, 0.0125226, -0.0008411, 0.0009532
6: -0.0009540, 0.0041938, -0.0007941, 0.0043176, -0.0036375, 0.0032096
7: 0.9773917, 0.9809939, 0.9775036, 0.9810805, -0.0025454, 0.0022459
8: -0.0108039, -0.0069418, -0.0106840, -0.0068489, -0.0027290, 0.0024080
9: -0.0004142, 0.0021370, -0.0004755, 0.0020578, -0.0015906, 0.0018027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017079
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017229
time: 0.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.44 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015696
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015696
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015858
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015858
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015941
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0015941
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0016093
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016866, upper bound: 0.0016093
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0015984
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0015984
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016142
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016142
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016256
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016256
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016475
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016940, upper bound: 0.0016475
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016483
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016483
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016483
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016483
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016751
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016751
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016751
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016559, upper bound: 0.0016751
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016807, upper bound: 0.0016808
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016807, upper bound: 0.0016948
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016807, upper bound: 0.0016808
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016807, upper bound: 0.0016948
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016815
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016956
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017079
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017229
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016204
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016204
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016325
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016325
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016485
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016484
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016590
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016344, upper bound: 0.0016590
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016466
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016611
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016648
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016804
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016815
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016956
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017079
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017229

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012352, 0.0000148, -0.0012693, -0.0000120, -0.0007660, 0.0008270
1: -0.0074451, -0.0042731, -0.0075316, -0.0043411, -0.0019439, 0.0020987
2: 0.0304110, 0.0323790, 0.0303574, 0.0323368, -0.0012060, 0.0013020
3: -0.0008972, 0.0027774, -0.0008185, 0.0028776, -0.0024312, 0.0022519
4: -0.0064660, -0.0032395, -0.0065540, -0.0033086, -0.0019773, 0.0021347
5: 0.0112890, 0.0125111, 0.0112557, 0.0124850, -0.0007489, 0.0008086
6: -0.0007506, 0.0039130, -0.0006507, 0.0040402, -0.0030855, 0.0028580
7: 0.9775340, 0.9807974, 0.9776040, 0.9808864, -0.0021591, 0.0019999
8: -0.0106513, -0.0071524, -0.0105763, -0.0070570, -0.0023149, 0.0021442
9: -0.0002750, 0.0020362, -0.0003380, 0.0019867, -0.0014163, 0.0015291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012352, 0.0000148, -0.0012836, 0.0000263, -0.0007966, 0.0008352
1: -0.0074451, -0.0042731, -0.0075679, -0.0042439, -0.0020214, 0.0021195
2: 0.0304110, 0.0323790, 0.0303349, 0.0323971, -0.0012541, 0.0013149
3: -0.0008972, 0.0027774, -0.0009311, 0.0029196, -0.0024553, 0.0023417
4: -0.0064660, -0.0032395, -0.0065908, -0.0032098, -0.0020561, 0.0021559
5: 0.0112890, 0.0125111, 0.0112418, 0.0125224, -0.0007788, 0.0008166
6: -0.0007506, 0.0039130, -0.0007935, 0.0040935, -0.0031162, 0.0029720
7: 0.9775340, 0.9807974, 0.9775040, 0.9809237, -0.0021805, 0.0020796
8: -0.0106513, -0.0071524, -0.0106835, -0.0070171, -0.0023379, 0.0022297
9: -0.0002750, 0.0020362, -0.0003644, 0.0020574, -0.0014728, 0.0015443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012448, 0.0000304, -0.0012724, -0.0000113, -0.0007729, 0.0008452
1: -0.0074695, -0.0042335, -0.0075394, -0.0043393, -0.0019613, 0.0021448
2: 0.0303959, 0.0324035, 0.0303526, 0.0323379, -0.0012168, 0.0013306
3: -0.0009431, 0.0028056, -0.0008205, 0.0028866, -0.0024846, 0.0022721
4: -0.0064907, -0.0031992, -0.0065619, -0.0033069, -0.0019950, 0.0021816
5: 0.0112797, 0.0125264, 0.0112527, 0.0124856, -0.0007557, 0.0008263
6: -0.0008088, 0.0039488, -0.0006532, 0.0040516, -0.0031533, 0.0028836
7: 0.9774932, 0.9808224, 0.9776021, 0.9808944, -0.0022065, 0.0020178
8: -0.0106950, -0.0071256, -0.0105782, -0.0070485, -0.0023657, 0.0021634
9: -0.0002928, 0.0020650, -0.0003437, 0.0019879, -0.0014290, 0.0015627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012448, 0.0000304, -0.0012869, 0.0000270, -0.0008035, 0.0008541
1: -0.0074695, -0.0042335, -0.0075763, -0.0042420, -0.0020390, 0.0021673
2: 0.0303959, 0.0324035, 0.0303297, 0.0323982, -0.0012650, 0.0013446
3: -0.0009431, 0.0028056, -0.0009332, 0.0029294, -0.0025107, 0.0023621
4: -0.0064907, -0.0031992, -0.0065994, -0.0032079, -0.0020740, 0.0022045
5: 0.0112797, 0.0125264, 0.0112385, 0.0125231, -0.0007856, 0.0008350
6: -0.0008088, 0.0039488, -0.0007962, 0.0041059, -0.0031864, 0.0029978
7: 0.9774932, 0.9808224, 0.9775021, 0.9809324, -0.0022297, 0.0020977
8: -0.0106950, -0.0071256, -0.0106855, -0.0070078, -0.0023906, 0.0022491
9: -0.0002928, 0.0020650, -0.0003706, 0.0020588, -0.0014857, 0.0015791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012358, 0.0000183, -0.0012888, -0.0000003, -0.0007682, 0.0008628
1: -0.0074466, -0.0042641, -0.0075812, -0.0043115, -0.0019493, 0.0021894
2: 0.0304101, 0.0323846, 0.0303266, 0.0323552, -0.0012094, 0.0013583
3: -0.0009077, 0.0027792, -0.0008528, 0.0029350, -0.0025363, 0.0022582
4: -0.0064675, -0.0032303, -0.0066043, -0.0032785, -0.0019828, 0.0022270
5: 0.0112885, 0.0125146, 0.0112366, 0.0124964, -0.0007510, 0.0008435
6: -0.0007638, 0.0039152, -0.0006942, 0.0041130, -0.0032189, 0.0028659
7: 0.9775248, 0.9807990, 0.9775735, 0.9809374, -0.0022525, 0.0020055
8: -0.0106612, -0.0071508, -0.0106090, -0.0070024, -0.0024150, 0.0021502
9: -0.0002761, 0.0020427, -0.0003741, 0.0020082, -0.0014203, 0.0015952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012358, 0.0000183, -0.0013051, 0.0000404, -0.0008010, 0.0008732
1: -0.0074466, -0.0042641, -0.0076225, -0.0042081, -0.0020327, 0.0022160
2: 0.0304101, 0.0323846, 0.0303010, 0.0324193, -0.0012611, 0.0013748
3: -0.0009077, 0.0027792, -0.0009726, 0.0029829, -0.0025671, 0.0023548
4: -0.0064675, -0.0032303, -0.0066464, -0.0031733, -0.0020676, 0.0022540
5: 0.0112885, 0.0125146, 0.0112207, 0.0125362, -0.0007832, 0.0008538
6: -0.0007638, 0.0039152, -0.0008462, 0.0041739, -0.0032580, 0.0029886
7: 0.9775248, 0.9807990, 0.9774671, 0.9809799, -0.0022798, 0.0020913
8: -0.0106612, -0.0071508, -0.0107230, -0.0069567, -0.0024443, 0.0022422
9: -0.0002761, 0.0020427, -0.0004043, 0.0020836, -0.0014811, 0.0016146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012454, 0.0000340, -0.0012915, 0.0000003, -0.0007755, 0.0008806
1: -0.0074710, -0.0042243, -0.0075880, -0.0043099, -0.0019679, 0.0022347
2: 0.0303950, 0.0324092, 0.0303224, 0.0323562, -0.0012209, 0.0013864
3: -0.0009538, 0.0028074, -0.0008547, 0.0029429, -0.0025888, 0.0022797
4: -0.0064923, -0.0031899, -0.0066113, -0.0032769, -0.0020017, 0.0022731
5: 0.0112791, 0.0125299, 0.0112340, 0.0124970, -0.0007582, 0.0008610
6: -0.0008223, 0.0039511, -0.0006966, 0.0041230, -0.0032855, 0.0028932
7: 0.9774839, 0.9808241, 0.9775718, 0.9809444, -0.0022990, 0.0020246
8: -0.0107051, -0.0071239, -0.0106107, -0.0069949, -0.0024649, 0.0021706
9: -0.0002939, 0.0020717, -0.0003791, 0.0020094, -0.0014338, 0.0016282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012454, 0.0000340, -0.0013089, 0.0000411, -0.0008084, 0.0008919
1: -0.0074710, -0.0042243, -0.0076320, -0.0042063, -0.0020515, 0.0022634
2: 0.0303950, 0.0324092, 0.0302951, 0.0324204, -0.0012728, 0.0014042
3: -0.0009538, 0.0028074, -0.0009746, 0.0029939, -0.0026220, 0.0023766
4: -0.0064923, -0.0031899, -0.0066561, -0.0031715, -0.0020868, 0.0023022
5: 0.0112791, 0.0125299, 0.0112170, 0.0125369, -0.0007904, 0.0008720
6: -0.0008223, 0.0039511, -0.0008488, 0.0041878, -0.0033277, 0.0030162
7: 0.9774839, 0.9808241, 0.9774653, 0.9809897, -0.0023286, 0.0021106
8: -0.0107051, -0.0071239, -0.0107250, -0.0069463, -0.0024966, 0.0022629
9: -0.0002939, 0.0020717, -0.0004112, 0.0020849, -0.0014948, 0.0016491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012836, 0.0000047, -0.0012931, -0.0000085, -0.0007971, 0.0008554
1: -0.0075680, -0.0042985, -0.0075920, -0.0043322, -0.0020227, 0.0021707
2: 0.0303348, 0.0323632, 0.0303199, 0.0323423, -0.0012549, 0.0013467
3: -0.0008678, 0.0029197, -0.0008288, 0.0029475, -0.0025146, 0.0023432
4: -0.0065909, -0.0032654, -0.0066153, -0.0032996, -0.0020574, 0.0022079
5: 0.0112417, 0.0125013, 0.0112325, 0.0124884, -0.0007793, 0.0008363
6: -0.0007132, 0.0040936, -0.0006637, 0.0041289, -0.0031914, 0.0029738
7: 0.9775602, 0.9809238, 0.9775949, 0.9809485, -0.0022332, 0.0020809
8: -0.0106232, -0.0070169, -0.0105861, -0.0069905, -0.0023943, 0.0022311
9: -0.0003645, 0.0020176, -0.0003820, 0.0019931, -0.0014738, 0.0015816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015963
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012836, 0.0000047, -0.0013081, 0.0000298, -0.0008258, 0.0008626
1: -0.0075680, -0.0042985, -0.0076302, -0.0042350, -0.0020955, 0.0021890
2: 0.0303348, 0.0323632, 0.0302962, 0.0324026, -0.0013000, 0.0013581
3: -0.0008678, 0.0029197, -0.0009414, 0.0029918, -0.0025359, 0.0024275
4: -0.0065909, -0.0032654, -0.0066542, -0.0032007, -0.0021314, 0.0022266
5: 0.0112417, 0.0125013, 0.0112178, 0.0125258, -0.0008073, 0.0008434
6: -0.0007132, 0.0040936, -0.0008066, 0.0041851, -0.0032184, 0.0030808
7: 0.9775602, 0.9809238, 0.9774948, 0.9809877, -0.0022521, 0.0021558
8: -0.0106232, -0.0070169, -0.0106933, -0.0069483, -0.0024146, 0.0023114
9: -0.0003645, 0.0020176, -0.0004098, 0.0020639, -0.0015268, 0.0015950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012957, 0.0000245, -0.0012963, -0.0000078, -0.0008053, 0.0008785
1: -0.0075987, -0.0042485, -0.0076000, -0.0043303, -0.0020435, 0.0022294
2: 0.0303158, 0.0323942, 0.0303149, 0.0323435, -0.0012678, 0.0013831
3: -0.0009257, 0.0029553, -0.0008310, 0.0029568, -0.0025827, 0.0023672
4: -0.0066222, -0.0032145, -0.0066235, -0.0032977, -0.0020785, 0.0022677
5: 0.0112299, 0.0125206, 0.0112294, 0.0124891, -0.0007873, 0.0008589
6: -0.0007867, 0.0041388, -0.0006665, 0.0041407, -0.0032777, 0.0030043
7: 0.9775088, 0.9809554, 0.9775929, 0.9809567, -0.0022936, 0.0021023
8: -0.0106784, -0.0069831, -0.0105882, -0.0069816, -0.0024591, 0.0022540
9: -0.0003869, 0.0020541, -0.0003879, 0.0019945, -0.0014889, 0.0016244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012957, 0.0000245, -0.0013120, 0.0000306, -0.0008340, 0.0008867
1: -0.0075987, -0.0042485, -0.0076400, -0.0042328, -0.0021165, 0.0022502
2: 0.0303158, 0.0323942, 0.0302901, 0.0324040, -0.0013131, 0.0013960
3: -0.0009257, 0.0029553, -0.0009439, 0.0030032, -0.0026068, 0.0024519
4: -0.0066222, -0.0032145, -0.0066642, -0.0031985, -0.0021528, 0.0022888
5: 0.0112299, 0.0125206, 0.0112139, 0.0125267, -0.0008154, 0.0008669
6: -0.0007867, 0.0041388, -0.0008098, 0.0041996, -0.0033083, 0.0031117
7: 0.9775088, 0.9809554, 0.9774926, 0.9809979, -0.0023150, 0.0021774
8: -0.0106784, -0.0069831, -0.0106957, -0.0069374, -0.0024820, 0.0023346
9: -0.0003869, 0.0020541, -0.0004170, 0.0020655, -0.0015421, 0.0016395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012843, 0.0000093, -0.0013133, 0.0000032, -0.0008017, 0.0008922
1: -0.0075697, -0.0042871, -0.0076433, -0.0043025, -0.0020344, 0.0022642
2: 0.0303338, 0.0323703, 0.0302881, 0.0323607, -0.0012621, 0.0014047
3: -0.0008810, 0.0029217, -0.0008632, 0.0030070, -0.0026230, 0.0023567
4: -0.0065927, -0.0032537, -0.0066676, -0.0032694, -0.0020693, 0.0023031
5: 0.0112411, 0.0125058, 0.0112127, 0.0124998, -0.0007838, 0.0008723
6: -0.0007300, 0.0040961, -0.0007073, 0.0042044, -0.0033289, 0.0029910
7: 0.9775484, 0.9809256, 0.9775643, 0.9810013, -0.0023294, 0.0020929
8: -0.0106358, -0.0070151, -0.0106188, -0.0069338, -0.0024975, 0.0022440
9: -0.0003658, 0.0020260, -0.0004194, 0.0020147, -0.0014823, 0.0016497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012843, 0.0000093, -0.0013308, 0.0000440, -0.0008331, 0.0009025
1: -0.0075697, -0.0042871, -0.0076877, -0.0041990, -0.0021140, 0.0022902
2: 0.0303338, 0.0323703, 0.0302605, 0.0324249, -0.0013115, 0.0014208
3: -0.0008810, 0.0029217, -0.0009830, 0.0030584, -0.0026530, 0.0024490
4: -0.0065927, -0.0032537, -0.0067127, -0.0031641, -0.0021503, 0.0023295
5: 0.0112411, 0.0125058, 0.0111956, 0.0125397, -0.0008145, 0.0008823
6: -0.0007300, 0.0040961, -0.0008595, 0.0042697, -0.0033671, 0.0031081
7: 0.9775484, 0.9809256, 0.9774578, 0.9810469, -0.0023561, 0.0021749
8: -0.0106358, -0.0070151, -0.0107330, -0.0068849, -0.0025261, 0.0023318
9: -0.0003658, 0.0020260, -0.0004518, 0.0020901, -0.0015403, 0.0016686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012964, 0.0000291, -0.0013164, 0.0000039, -0.0008100, 0.0009169
1: -0.0076003, -0.0042368, -0.0076510, -0.0043008, -0.0020554, 0.0023269
2: 0.0303148, 0.0324015, 0.0302833, 0.0323618, -0.0012752, 0.0014436
3: -0.0009393, 0.0029572, -0.0008651, 0.0030159, -0.0026956, 0.0023811
4: -0.0066238, -0.0032026, -0.0066754, -0.0032677, -0.0020907, 0.0023668
5: 0.0112293, 0.0125251, 0.0112097, 0.0125005, -0.0007919, 0.0008965
6: -0.0008039, 0.0041411, -0.0007099, 0.0042157, -0.0034210, 0.0030219
7: 0.9774967, 0.9809570, 0.9775626, 0.9810092, -0.0023939, 0.0021146
8: -0.0106913, -0.0069813, -0.0106207, -0.0069253, -0.0025666, 0.0022672
9: -0.0003881, 0.0020626, -0.0004250, 0.0020160, -0.0014976, 0.0016954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012964, 0.0000291, -0.0013346, 0.0000448, -0.0008415, 0.0009273
1: -0.0076003, -0.0042368, -0.0076973, -0.0041970, -0.0021354, 0.0023530
2: 0.0303148, 0.0324015, 0.0302546, 0.0324262, -0.0013248, 0.0014598
3: -0.0009393, 0.0029572, -0.0009854, 0.0030695, -0.0027259, 0.0024738
4: -0.0066238, -0.0032026, -0.0067225, -0.0031621, -0.0021721, 0.0023934
5: 0.0112293, 0.0125251, 0.0111919, 0.0125405, -0.0008227, 0.0009066
6: -0.0008039, 0.0041411, -0.0008625, 0.0042838, -0.0034595, 0.0031395
7: 0.9774967, 0.9809570, 0.9774557, 0.9810569, -0.0024208, 0.0021969
8: -0.0106913, -0.0069813, -0.0107352, -0.0068743, -0.0025955, 0.0023554
9: -0.0003881, 0.0020626, -0.0004587, 0.0020916, -0.0015559, 0.0017145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012291, 0.0000140, -0.0012606, 0.0000070, -0.0008027, 0.0008260
1: -0.0074297, -0.0042752, -0.0075094, -0.0042928, -0.0020369, 0.0020962
2: 0.0304206, 0.0323777, 0.0303711, 0.0323667, -0.0012637, 0.0013005
3: -0.0008948, 0.0027595, -0.0008744, 0.0028519, -0.0024283, 0.0023596
4: -0.0064503, -0.0032416, -0.0065314, -0.0032595, -0.0020719, 0.0021322
5: 0.0112950, 0.0125104, 0.0112643, 0.0125036, -0.0007848, 0.0008076
6: -0.0007475, 0.0038903, -0.0007216, 0.0040075, -0.0030819, 0.0029947
7: 0.9775361, 0.9807816, 0.9775543, 0.9808636, -0.0021565, 0.0020955
8: -0.0106490, -0.0071695, -0.0106295, -0.0070815, -0.0023122, 0.0022467
9: -0.0002637, 0.0020347, -0.0003219, 0.0020218, -0.0014841, 0.0015273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015696
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015670
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012291, 0.0000140, -0.0012794, 0.0000466, -0.0008364, 0.0008355
1: -0.0074297, -0.0042752, -0.0075573, -0.0041923, -0.0021225, 0.0021201
2: 0.0304206, 0.0323777, 0.0303415, 0.0324291, -0.0013168, 0.0013153
3: -0.0008948, 0.0027595, -0.0009908, 0.0029073, -0.0024560, 0.0024589
4: -0.0064503, -0.0032416, -0.0065801, -0.0031573, -0.0021590, 0.0021565
5: 0.0112950, 0.0125104, 0.0112458, 0.0125423, -0.0008178, 0.0008168
6: -0.0007475, 0.0038903, -0.0008693, 0.0040779, -0.0031170, 0.0031206
7: 0.9775361, 0.9807816, 0.9774510, 0.9809127, -0.0021811, 0.0021837
8: -0.0106490, -0.0071695, -0.0107404, -0.0070287, -0.0023385, 0.0023412
9: -0.0002637, 0.0020347, -0.0003567, 0.0020950, -0.0015465, 0.0015447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015696
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012389, 0.0000296, -0.0012638, 0.0000078, -0.0008098, 0.0008447
1: -0.0074543, -0.0042354, -0.0075176, -0.0042908, -0.0020549, 0.0021436
2: 0.0304053, 0.0324023, 0.0303661, 0.0323680, -0.0012749, 0.0013299
3: -0.0009409, 0.0027881, -0.0008767, 0.0028613, -0.0024833, 0.0023805
4: -0.0064753, -0.0032012, -0.0065397, -0.0032575, -0.0020902, 0.0021804
5: 0.0112855, 0.0125257, 0.0112611, 0.0125043, -0.0007917, 0.0008259
6: -0.0008060, 0.0039266, -0.0007246, 0.0040195, -0.0031516, 0.0030211
7: 0.9774953, 0.9808068, 0.9775523, 0.9808720, -0.0022053, 0.0021140
8: -0.0106928, -0.0071423, -0.0106318, -0.0070725, -0.0023644, 0.0022666
9: -0.0002817, 0.0020636, -0.0003278, 0.0020233, -0.0014972, 0.0015619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015858
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012389, 0.0000296, -0.0012829, 0.0000475, -0.0008435, 0.0008539
1: -0.0074543, -0.0042354, -0.0075662, -0.0041901, -0.0021405, 0.0021670
2: 0.0304053, 0.0324023, 0.0303359, 0.0324304, -0.0013280, 0.0013444
3: -0.0009409, 0.0027881, -0.0009933, 0.0029177, -0.0025104, 0.0024797
4: -0.0064753, -0.0032012, -0.0065891, -0.0031551, -0.0021773, 0.0022042
5: 0.0112855, 0.0125257, 0.0112424, 0.0125431, -0.0008247, 0.0008349
6: -0.0008060, 0.0039266, -0.0008726, 0.0040910, -0.0031860, 0.0031471
7: 0.9774953, 0.9808068, 0.9774487, 0.9809219, -0.0022294, 0.0022022
8: -0.0106928, -0.0071423, -0.0107428, -0.0070189, -0.0023903, 0.0023611
9: -0.0002817, 0.0020636, -0.0003632, 0.0020966, -0.0015596, 0.0015789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015858
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012297, 0.0000174, -0.0012837, 0.0000198, -0.0008039, 0.0008625
1: -0.0074312, -0.0042663, -0.0075681, -0.0042603, -0.0020399, 0.0021887
2: 0.0304197, 0.0323832, 0.0303347, 0.0323869, -0.0012656, 0.0013579
3: -0.0009051, 0.0027613, -0.0009120, 0.0029199, -0.0025355, 0.0023632
4: -0.0064518, -0.0032326, -0.0065910, -0.0032265, -0.0020750, 0.0022263
5: 0.0112944, 0.0125137, 0.0112417, 0.0125161, -0.0007859, 0.0008433
6: -0.0007605, 0.0038925, -0.0007693, 0.0040938, -0.0032179, 0.0029992
7: 0.9775271, 0.9807830, 0.9775209, 0.9809239, -0.0022517, 0.0020987
8: -0.0106587, -0.0071678, -0.0106654, -0.0070168, -0.0024142, 0.0022501
9: -0.0002649, 0.0020411, -0.0003646, 0.0020455, -0.0014863, 0.0015947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015941
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012297, 0.0000174, -0.0013019, 0.0000613, -0.0008391, 0.0008730
1: -0.0074312, -0.0042663, -0.0076143, -0.0041550, -0.0021293, 0.0022154
2: 0.0304197, 0.0323832, 0.0303061, 0.0324523, -0.0013211, 0.0013744
3: -0.0009051, 0.0027613, -0.0010341, 0.0029734, -0.0025664, 0.0024667
4: -0.0064518, -0.0032326, -0.0066381, -0.0031193, -0.0021659, 0.0022534
5: 0.0112944, 0.0125137, 0.0112239, 0.0125567, -0.0008204, 0.0008535
6: -0.0007605, 0.0038925, -0.0009243, 0.0041618, -0.0032571, 0.0031306
7: 0.9775271, 0.9807830, 0.9774125, 0.9809715, -0.0022792, 0.0021907
8: -0.0106587, -0.0071678, -0.0107816, -0.0069658, -0.0024436, 0.0023487
9: -0.0002649, 0.0020411, -0.0003983, 0.0021222, -0.0015515, 0.0016141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015941
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012395, 0.0000332, -0.0012867, 0.0000206, -0.0008115, 0.0008809
1: -0.0074559, -0.0042264, -0.0075758, -0.0042583, -0.0020592, 0.0022354
2: 0.0304043, 0.0324080, 0.0303300, 0.0323882, -0.0012775, 0.0013868
3: -0.0009513, 0.0027899, -0.0009144, 0.0029288, -0.0025896, 0.0023855
4: -0.0064769, -0.0031920, -0.0065989, -0.0032244, -0.0020946, 0.0022738
5: 0.0112849, 0.0125291, 0.0112387, 0.0125169, -0.0007934, 0.0008612
6: -0.0008193, 0.0039289, -0.0007724, 0.0041051, -0.0032865, 0.0030275
7: 0.9774860, 0.9808085, 0.9775187, 0.9809318, -0.0022997, 0.0021185
8: -0.0107028, -0.0071405, -0.0106676, -0.0070083, -0.0024657, 0.0022714
9: -0.0002829, 0.0020702, -0.0003702, 0.0020470, -0.0015004, 0.0016287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016093
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012395, 0.0000332, -0.0013055, 0.0000622, -0.0008467, 0.0008918
1: -0.0074559, -0.0042264, -0.0076235, -0.0041528, -0.0021486, 0.0022630
2: 0.0304043, 0.0324080, 0.0303004, 0.0324536, -0.0013330, 0.0014040
3: -0.0009513, 0.0027899, -0.0010366, 0.0029840, -0.0026216, 0.0024890
4: -0.0064769, -0.0031920, -0.0066474, -0.0031172, -0.0021855, 0.0023018
5: 0.0112849, 0.0125291, 0.0112203, 0.0125575, -0.0008278, 0.0008719
6: -0.0008193, 0.0039289, -0.0009274, 0.0041753, -0.0033271, 0.0031589
7: 0.9774860, 0.9808085, 0.9774103, 0.9809809, -0.0023281, 0.0022105
8: -0.0107028, -0.0071405, -0.0107839, -0.0069557, -0.0024961, 0.0023700
9: -0.0002829, 0.0020702, -0.0004050, 0.0021238, -0.0015655, 0.0016488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016093
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012797, 0.0000040, -0.0012862, 0.0000105, -0.0008347, 0.0008550
1: -0.0075580, -0.0043004, -0.0075745, -0.0042839, -0.0021182, 0.0021697
2: 0.0303410, 0.0323621, 0.0303308, 0.0323723, -0.0013142, 0.0013461
3: -0.0008657, 0.0029082, -0.0008848, 0.0029272, -0.0025135, 0.0024539
4: -0.0065808, -0.0032672, -0.0065975, -0.0032504, -0.0021546, 0.0022070
5: 0.0112456, 0.0125006, 0.0112392, 0.0125070, -0.0008161, 0.0008359
6: -0.0007105, 0.0040790, -0.0007348, 0.0041032, -0.0031900, 0.0031143
7: 0.9775621, 0.9809135, 0.9775451, 0.9809304, -0.0022322, 0.0021792
8: -0.0106212, -0.0070279, -0.0106394, -0.0070098, -0.0023932, 0.0023365
9: -0.0003573, 0.0020163, -0.0003692, 0.0020283, -0.0015434, 0.0015809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0015984
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0015963
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012797, 0.0000040, -0.0013054, 0.0000501, -0.0008676, 0.0008635
1: -0.0075580, -0.0043004, -0.0076233, -0.0041835, -0.0022017, 0.0021913
2: 0.0303410, 0.0323621, 0.0303005, 0.0324346, -0.0013660, 0.0013595
3: -0.0008657, 0.0029082, -0.0010011, 0.0029838, -0.0025385, 0.0025506
4: -0.0065808, -0.0032672, -0.0066472, -0.0031483, -0.0022395, 0.0022289
5: 0.0112456, 0.0125006, 0.0112204, 0.0125457, -0.0008483, 0.0008442
6: -0.0007105, 0.0040790, -0.0008824, 0.0041749, -0.0032216, 0.0032370
7: 0.9775621, 0.9809135, 0.9774418, 0.9809807, -0.0022544, 0.0022651
8: -0.0106212, -0.0070279, -0.0107501, -0.0069559, -0.0024170, 0.0024286
9: -0.0003573, 0.0020163, -0.0004048, 0.0021015, -0.0016042, 0.0015966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0015984
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0015964
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012916, 0.0000238, -0.0012894, 0.0000113, -0.0008433, 0.0008789
1: -0.0075882, -0.0042503, -0.0075825, -0.0042820, -0.0021400, 0.0022304
2: 0.0303223, 0.0323931, 0.0303258, 0.0323735, -0.0013277, 0.0013838
3: -0.0009236, 0.0029432, -0.0008870, 0.0029366, -0.0025838, 0.0024791
4: -0.0066115, -0.0032163, -0.0066057, -0.0032485, -0.0021767, 0.0022687
5: 0.0112339, 0.0125199, 0.0112361, 0.0125077, -0.0008245, 0.0008593
6: -0.0007841, 0.0041234, -0.0007375, 0.0041150, -0.0032792, 0.0031463
7: 0.9775106, 0.9809446, 0.9775432, 0.9809388, -0.0022946, 0.0022016
8: -0.0106764, -0.0069946, -0.0106415, -0.0070009, -0.0024602, 0.0023605
9: -0.0003793, 0.0020528, -0.0003751, 0.0020297, -0.0015592, 0.0016251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016142
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016095
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012916, 0.0000238, -0.0013090, 0.0000510, -0.0008763, 0.0008876
1: -0.0075882, -0.0042503, -0.0076324, -0.0041812, -0.0022237, 0.0022524
2: 0.0303223, 0.0323931, 0.0302948, 0.0324360, -0.0013796, 0.0013974
3: -0.0009236, 0.0029432, -0.0010037, 0.0029944, -0.0026093, 0.0025761
4: -0.0066115, -0.0032163, -0.0066565, -0.0031461, -0.0022619, 0.0022910
5: 0.0112339, 0.0125199, 0.0112169, 0.0125465, -0.0008568, 0.0008678
6: -0.0007841, 0.0041234, -0.0008856, 0.0041884, -0.0033115, 0.0032694
7: 0.9775106, 0.9809446, 0.9774396, 0.9809901, -0.0023172, 0.0022878
8: -0.0106764, -0.0069946, -0.0107526, -0.0069459, -0.0024844, 0.0024528
9: -0.0003793, 0.0020528, -0.0004115, 0.0021031, -0.0016202, 0.0016411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016142
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016095
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012804, 0.0000084, -0.0013096, 0.0000232, -0.0008379, 0.0008923
1: -0.0075597, -0.0042892, -0.0076339, -0.0042517, -0.0021262, 0.0022643
2: 0.0303400, 0.0323690, 0.0302939, 0.0323923, -0.0013191, 0.0014048
3: -0.0008786, 0.0029101, -0.0009221, 0.0029961, -0.0026231, 0.0024631
4: -0.0065825, -0.0032559, -0.0066579, -0.0032177, -0.0021627, 0.0023032
5: 0.0112449, 0.0125049, 0.0112163, 0.0125194, -0.0008192, 0.0008724
6: -0.0007269, 0.0040814, -0.0007821, 0.0041905, -0.0033291, 0.0031260
7: 0.9775506, 0.9809152, 0.9775120, 0.9809915, -0.0023295, 0.0021874
8: -0.0106335, -0.0070261, -0.0106749, -0.0069443, -0.0024976, 0.0023452
9: -0.0003585, 0.0020244, -0.0004125, 0.0020518, -0.0015492, 0.0016498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016256
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016253
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012804, 0.0000084, -0.0013284, 0.0000648, -0.0008729, 0.0009022
1: -0.0075597, -0.0042892, -0.0076815, -0.0041462, -0.0022150, 0.0022894
2: 0.0303400, 0.0323690, 0.0302644, 0.0324577, -0.0013742, 0.0014204
3: -0.0008786, 0.0029101, -0.0010443, 0.0030512, -0.0026522, 0.0025660
4: -0.0065825, -0.0032559, -0.0067064, -0.0031104, -0.0022530, 0.0023287
5: 0.0112449, 0.0125049, 0.0111980, 0.0125600, -0.0008534, 0.0008821
6: -0.0007269, 0.0040814, -0.0009372, 0.0042605, -0.0033659, 0.0032565
7: 0.9775506, 0.9809152, 0.9774035, 0.9810406, -0.0023553, 0.0022788
8: -0.0106335, -0.0070261, -0.0107913, -0.0068917, -0.0025253, 0.0024432
9: -0.0003585, 0.0020244, -0.0004472, 0.0021286, -0.0016139, 0.0016681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016256
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016253
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012922, 0.0000283, -0.0013127, 0.0000240, -0.0008466, 0.0009170
1: -0.0075898, -0.0042389, -0.0076419, -0.0042497, -0.0021484, 0.0023269
2: 0.0303213, 0.0324002, 0.0302890, 0.0323935, -0.0013329, 0.0014436
3: -0.0009369, 0.0029450, -0.0009244, 0.0030053, -0.0026957, 0.0024888
4: -0.0066131, -0.0032047, -0.0066661, -0.0032157, -0.0021853, 0.0023669
5: 0.0112333, 0.0125243, 0.0112132, 0.0125202, -0.0008277, 0.0008965
6: -0.0008009, 0.0041257, -0.0007850, 0.0042023, -0.0034211, 0.0031586
7: 0.9774988, 0.9809462, 0.9775100, 0.9809998, -0.0023939, 0.0022102
8: -0.0106890, -0.0069929, -0.0106771, -0.0069354, -0.0025667, 0.0023697
9: -0.0003804, 0.0020611, -0.0004183, 0.0020532, -0.0015653, 0.0016954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016475
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016475
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012922, 0.0000283, -0.0013319, 0.0000656, -0.0008817, 0.0009268
1: -0.0075898, -0.0042389, -0.0076905, -0.0041440, -0.0022373, 0.0023519
2: 0.0303213, 0.0324002, 0.0302588, 0.0324591, -0.0013881, 0.0014591
3: -0.0009369, 0.0029450, -0.0010468, 0.0030617, -0.0027246, 0.0025919
4: -0.0066131, -0.0032047, -0.0067156, -0.0031082, -0.0022758, 0.0023923
5: 0.0112333, 0.0125243, 0.0111945, 0.0125609, -0.0008620, 0.0009061
6: -0.0008009, 0.0041257, -0.0009404, 0.0042738, -0.0034578, 0.0032894
7: 0.9774988, 0.9809462, 0.9774013, 0.9810498, -0.0024196, 0.0023018
8: -0.0106890, -0.0069929, -0.0107937, -0.0068817, -0.0025942, 0.0024679
9: -0.0003804, 0.0020611, -0.0004538, 0.0021302, -0.0016302, 0.0017136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016475
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016475
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012576, 0.0000648, -0.0012968, 0.0000036, -0.0007844, 0.0009078
1: -0.0075018, -0.0041461, -0.0076014, -0.0043013, -0.0019905, 0.0023038
2: 0.0303759, 0.0324578, 0.0303141, 0.0323615, -0.0012349, 0.0014293
3: -0.0010443, 0.0028431, -0.0008645, 0.0029584, -0.0026688, 0.0023059
4: -0.0065236, -0.0031103, -0.0066249, -0.0032682, -0.0020246, 0.0023433
5: 0.0112672, 0.0125601, 0.0112288, 0.0125003, -0.0007669, 0.0008876
6: -0.0009373, 0.0039964, -0.0007091, 0.0041428, -0.0033871, 0.0029264
7: 0.9774034, 0.9808557, 0.9775631, 0.9809582, -0.0023701, 0.0020478
8: -0.0107913, -0.0070899, -0.0106201, -0.0069801, -0.0025411, 0.0021955
9: -0.0003163, 0.0021287, -0.0003889, 0.0020156, -0.0014503, 0.0016786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016506, upper bound: 0.0016649
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016506, upper bound: 0.0016747
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012576, 0.0000648, -0.0013137, 0.0000444, -0.0008015, 0.0009033
1: -0.0075018, -0.0041461, -0.0076443, -0.0041978, -0.0020339, 0.0022922
2: 0.0303759, 0.0324578, 0.0302874, 0.0324257, -0.0012618, 0.0014221
3: -0.0010443, 0.0028431, -0.0009845, 0.0030082, -0.0026554, 0.0023561
4: -0.0065236, -0.0031103, -0.0066686, -0.0031629, -0.0020688, 0.0023315
5: 0.0112672, 0.0125601, 0.0112123, 0.0125402, -0.0007836, 0.0008831
6: -0.0009373, 0.0039964, -0.0008613, 0.0042059, -0.0033700, 0.0029902
7: 0.9774034, 0.9808557, 0.9774566, 0.9810024, -0.0023582, 0.0020924
8: -0.0107913, -0.0070899, -0.0107344, -0.0069327, -0.0025283, 0.0022434
9: -0.0003163, 0.0021287, -0.0004202, 0.0020910, -0.0014819, 0.0016701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016506, upper bound: 0.0016649
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016506, upper bound: 0.0016747
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0012500, 0.0000798, -0.0012968, 0.0000036, -0.0007899, 0.0009476
1: -0.0074827, -0.0041082, -0.0076014, -0.0043013, -0.0020045, 0.0024046
2: 0.0303877, 0.0324813, 0.0303141, 0.0323615, -0.0012436, 0.0014918
3: -0.0010883, 0.0028209, -0.0008645, 0.0029584, -0.0027856, 0.0023221
4: -0.0065042, -0.0030717, -0.0066249, -0.0032682, -0.0020389, 0.0024459
5: 0.0112746, 0.0125747, 0.0112288, 0.0125003, -0.0007723, 0.0009264
6: -0.0009930, 0.0039683, -0.0007091, 0.0041428, -0.0035353, 0.0029470
7: 0.9773644, 0.9808360, 0.9775631, 0.9809582, -0.0024738, 0.0020622
8: -0.0108332, -0.0071110, -0.0106201, -0.0069801, -0.0026523, 0.0022110
9: -0.0003024, 0.0021563, -0.0003889, 0.0020156, -0.0014605, 0.0017520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016397, upper bound: 0.0016485
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016397, upper bound: 0.0016590
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0012500, 0.0000798, -0.0013137, 0.0000444, -0.0008079, 0.0009432
1: -0.0074827, -0.0041082, -0.0076443, -0.0041978, -0.0020502, 0.0023934
2: 0.0303877, 0.0324813, 0.0302874, 0.0324257, -0.0012719, 0.0014849
3: -0.0010883, 0.0028209, -0.0009845, 0.0030082, -0.0027727, 0.0023751
4: -0.0065042, -0.0030717, -0.0066686, -0.0031629, -0.0020854, 0.0024345
5: 0.0112746, 0.0125747, 0.0112123, 0.0125402, -0.0007899, 0.0009221
6: -0.0009930, 0.0039683, -0.0008613, 0.0042059, -0.0035189, 0.0030142
7: 0.9773644, 0.9808360, 0.9774566, 0.9810024, -0.0024624, 0.0021092
8: -0.0108332, -0.0071110, -0.0107344, -0.0069327, -0.0026400, 0.0022614
9: -0.0003024, 0.0021563, -0.0004202, 0.0020910, -0.0014938, 0.0017439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016397, upper bound: 0.0016485
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016397, upper bound: 0.0016590
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0013082, 0.0000527, -0.0012664, 0.0000047, -0.0008468, 0.0008574
1: -0.0076304, -0.0041768, -0.0075243, -0.0042985, -0.0021490, 0.0021758
2: 0.0302961, 0.0324387, 0.0303619, 0.0323632, -0.0013332, 0.0013499
3: -0.0010088, 0.0029920, -0.0008678, 0.0028691, -0.0025206, 0.0024895
4: -0.0066544, -0.0031415, -0.0065465, -0.0032654, -0.0021859, 0.0022131
5: 0.0112177, 0.0125482, 0.0112585, 0.0125013, -0.0008279, 0.0008383
6: -0.0008922, 0.0041854, -0.0007132, 0.0040294, -0.0031989, 0.0031595
7: 0.9774350, 0.9809880, 0.9775602, 0.9808788, -0.0022384, 0.0022108
8: -0.0107575, -0.0069481, -0.0106232, -0.0070651, -0.0024000, 0.0023704
9: -0.0004100, 0.0021063, -0.0003327, 0.0020176, -0.0015658, 0.0015853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016747, upper bound: 0.0016586
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016747, upper bound: 0.0016794
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0013082, 0.0000527, -0.0013167, -0.0000053, -0.0008143, 0.0008832
1: -0.0076304, -0.0041768, -0.0076518, -0.0043240, -0.0020664, 0.0022411
2: 0.0302961, 0.0324387, 0.0302828, 0.0323474, -0.0012820, 0.0013904
3: -0.0010088, 0.0029920, -0.0008383, 0.0030169, -0.0025963, 0.0023938
4: -0.0066544, -0.0031415, -0.0066762, -0.0032913, -0.0021019, 0.0022796
5: 0.0112177, 0.0125482, 0.0112094, 0.0124915, -0.0007961, 0.0008635
6: -0.0008922, 0.0041854, -0.0006758, 0.0042169, -0.0032950, 0.0030381
7: 0.9774350, 0.9809880, 0.9775864, 0.9810100, -0.0023057, 0.0021259
8: -0.0107575, -0.0069481, -0.0105951, -0.0069244, -0.0024720, 0.0022793
9: -0.0004100, 0.0021063, -0.0004256, 0.0019991, -0.0015056, 0.0016329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016747, upper bound: 0.0016682
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016747, upper bound: 0.0016901
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013056, 0.0000736, -0.0012664, 0.0000047, -0.0008529, 0.0008990
1: -0.0076238, -0.0041237, -0.0075243, -0.0042985, -0.0021643, 0.0022814
2: 0.0303002, 0.0324717, 0.0303619, 0.0323632, -0.0013427, 0.0014154
3: -0.0010703, 0.0029843, -0.0008678, 0.0028691, -0.0026429, 0.0025072
4: -0.0066477, -0.0030875, -0.0065465, -0.0032654, -0.0022014, 0.0023206
5: 0.0112202, 0.0125687, 0.0112585, 0.0125013, -0.0008338, 0.0008790
6: -0.0009703, 0.0041756, -0.0007132, 0.0040294, -0.0033542, 0.0031820
7: 0.9773803, 0.9809812, 0.9775602, 0.9808788, -0.0023471, 0.0022266
8: -0.0108161, -0.0069554, -0.0106232, -0.0070651, -0.0025164, 0.0023873
9: -0.0004052, 0.0021450, -0.0003327, 0.0020176, -0.0015769, 0.0016623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016466
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016648
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013056, 0.0000736, -0.0013167, -0.0000053, -0.0008215, 0.0009257
1: -0.0076238, -0.0041237, -0.0076518, -0.0043240, -0.0020846, 0.0023491
2: 0.0303002, 0.0324717, 0.0302828, 0.0323474, -0.0012933, 0.0014574
3: -0.0010703, 0.0029843, -0.0008383, 0.0030169, -0.0027213, 0.0024149
4: -0.0066477, -0.0030875, -0.0066762, -0.0032913, -0.0021204, 0.0023894
5: 0.0112202, 0.0125687, 0.0112094, 0.0124915, -0.0008032, 0.0009050
6: -0.0009703, 0.0041756, -0.0006758, 0.0042169, -0.0034537, 0.0030649
7: 0.9773803, 0.9809812, 0.9775864, 0.9810100, -0.0024167, 0.0021446
8: -0.0108161, -0.0069554, -0.0105951, -0.0069244, -0.0025911, 0.0022994
9: -0.0004052, 0.0021450, -0.0004256, 0.0019991, -0.0015189, 0.0017116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016611
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016803
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0013009, 0.0000501, -0.0012773, 0.0000111, -0.0008437, 0.0008770
1: -0.0076119, -0.0041835, -0.0075520, -0.0042824, -0.0021410, 0.0022256
2: 0.0303076, 0.0324346, 0.0303448, 0.0323732, -0.0013283, 0.0013808
3: -0.0010011, 0.0029706, -0.0008864, 0.0029012, -0.0025782, 0.0024802
4: -0.0066356, -0.0031483, -0.0065746, -0.0032490, -0.0021777, 0.0022638
5: 0.0112248, 0.0125457, 0.0112479, 0.0125076, -0.0008249, 0.0008575
6: -0.0008824, 0.0041582, -0.0007369, 0.0040701, -0.0032721, 0.0031477
7: 0.9774418, 0.9809690, 0.9775437, 0.9809073, -0.0022897, 0.0022026
8: -0.0107501, -0.0069685, -0.0106410, -0.0070346, -0.0024549, 0.0023616
9: -0.0003965, 0.0021015, -0.0003528, 0.0020294, -0.0015599, 0.0016216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016811
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016815
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0013009, 0.0000501, -0.0013311, 0.0000025, -0.0008112, 0.0009073
1: -0.0076119, -0.0041835, -0.0076883, -0.0043042, -0.0020584, 0.0023025
2: 0.0303076, 0.0324346, 0.0302601, 0.0323597, -0.0012771, 0.0014285
3: -0.0010011, 0.0029706, -0.0008613, 0.0030591, -0.0026674, 0.0023846
4: -0.0066356, -0.0031483, -0.0067134, -0.0032711, -0.0020938, 0.0023421
5: 0.0112248, 0.0125457, 0.0111953, 0.0124992, -0.0007931, 0.0008871
6: -0.0008824, 0.0041582, -0.0007049, 0.0042706, -0.0033852, 0.0030264
7: 0.9774418, 0.9809690, 0.9775660, 0.9810476, -0.0023688, 0.0021177
8: -0.0107501, -0.0069685, -0.0106170, -0.0068842, -0.0025397, 0.0022705
9: -0.0003965, 0.0021015, -0.0004522, 0.0020135, -0.0014998, 0.0016776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016952
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016956
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013136, 0.0000702, -0.0012802, 0.0000118, -0.0008522, 0.0009014
1: -0.0076441, -0.0041324, -0.0075593, -0.0042806, -0.0021625, 0.0022875
2: 0.0302876, 0.0324662, 0.0303402, 0.0323743, -0.0013416, 0.0014192
3: -0.0010602, 0.0030079, -0.0008886, 0.0029097, -0.0026500, 0.0025051
4: -0.0066684, -0.0030964, -0.0065821, -0.0032471, -0.0021996, 0.0023268
5: 0.0112124, 0.0125653, 0.0112451, 0.0125083, -0.0008332, 0.0008813
6: -0.0009574, 0.0042056, -0.0007396, 0.0040809, -0.0033631, 0.0031793
7: 0.9773893, 0.9810021, 0.9775417, 0.9809148, -0.0023534, 0.0022247
8: -0.0108064, -0.0069330, -0.0106430, -0.0070265, -0.0025232, 0.0023853
9: -0.0004200, 0.0021387, -0.0003582, 0.0020307, -0.0015756, 0.0016667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017051
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017079
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013136, 0.0000702, -0.0013348, 0.0000034, -0.0008197, 0.0009302
1: -0.0076441, -0.0041324, -0.0076977, -0.0043021, -0.0020802, 0.0023605
2: 0.0302876, 0.0324662, 0.0302543, 0.0323610, -0.0012906, 0.0014645
3: -0.0010602, 0.0030079, -0.0008637, 0.0030700, -0.0027346, 0.0024098
4: -0.0066684, -0.0030964, -0.0067229, -0.0032689, -0.0021159, 0.0024011
5: 0.0112124, 0.0125653, 0.0111917, 0.0125000, -0.0008015, 0.0009095
6: -0.0009574, 0.0042056, -0.0007080, 0.0042844, -0.0034705, 0.0030584
7: 0.9773893, 0.9810021, 0.9775638, 0.9810572, -0.0024285, 0.0021401
8: -0.0108064, -0.0069330, -0.0106193, -0.0068738, -0.0026037, 0.0022945
9: -0.0004200, 0.0021387, -0.0004591, 0.0020151, -0.0015157, 0.0017199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017197
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017229
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013098, 0.0000647, -0.0013081, 0.0000107, -0.0008540, 0.0008934
1: -0.0076343, -0.0041465, -0.0076299, -0.0042835, -0.0021670, 0.0022670
2: 0.0302937, 0.0324575, 0.0302964, 0.0323725, -0.0013444, 0.0014065
3: -0.0010439, 0.0029965, -0.0008852, 0.0029915, -0.0026263, 0.0025104
4: -0.0066584, -0.0031107, -0.0066540, -0.0032500, -0.0022043, 0.0023060
5: 0.0112162, 0.0125599, 0.0112178, 0.0125071, -0.0008349, 0.0008734
6: -0.0009368, 0.0041911, -0.0007353, 0.0041847, -0.0033331, 0.0031860
7: 0.9774037, 0.9809920, 0.9775447, 0.9809875, -0.0023323, 0.0022294
8: -0.0107910, -0.0069438, -0.0106398, -0.0069486, -0.0025006, 0.0023903
9: -0.0004128, 0.0021284, -0.0004097, 0.0020286, -0.0015789, 0.0016518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016804
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016804
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0012980, 0.0000491, -0.0012722, 0.0000269, -0.0008778, 0.0008765
1: -0.0076044, -0.0041859, -0.0075389, -0.0042423, -0.0022276, 0.0022242
2: 0.0303122, 0.0324331, 0.0303529, 0.0323981, -0.0013820, 0.0013799
3: -0.0009983, 0.0029619, -0.0009329, 0.0028860, -0.0025766, 0.0025805
4: -0.0066279, -0.0031508, -0.0065613, -0.0032082, -0.0022658, 0.0022624
5: 0.0112277, 0.0125448, 0.0112529, 0.0125230, -0.0008582, 0.0008569
6: -0.0008788, 0.0041471, -0.0007959, 0.0040509, -0.0032701, 0.0032750
7: 0.9774443, 0.9809612, 0.9775023, 0.9808939, -0.0022882, 0.0022917
8: -0.0107475, -0.0069768, -0.0106852, -0.0070490, -0.0024533, 0.0024571
9: -0.0003910, 0.0020997, -0.0003433, 0.0020586, -0.0016230, 0.0016206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016811
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016815
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0012980, 0.0000491, -0.0013288, 0.0000225, -0.0008484, 0.0009092
1: -0.0076044, -0.0041859, -0.0076826, -0.0042534, -0.0021530, 0.0023073
2: 0.0303122, 0.0324331, 0.0302637, 0.0323912, -0.0013357, 0.0014314
3: -0.0009983, 0.0029619, -0.0009200, 0.0030525, -0.0026729, 0.0024942
4: -0.0066279, -0.0031508, -0.0067075, -0.0032195, -0.0021900, 0.0023469
5: 0.0112277, 0.0125448, 0.0111976, 0.0125187, -0.0008295, 0.0008889
6: -0.0008788, 0.0041471, -0.0007795, 0.0042622, -0.0033922, 0.0031654
7: 0.9774443, 0.9809612, 0.9775138, 0.9810418, -0.0023737, 0.0022150
8: -0.0107475, -0.0069768, -0.0106730, -0.0068905, -0.0025450, 0.0023748
9: -0.0003910, 0.0020997, -0.0004480, 0.0020505, -0.0015687, 0.0016811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016952
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016956
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013105, 0.0000693, -0.0012751, 0.0000277, -0.0008865, 0.0009007
1: -0.0076361, -0.0041347, -0.0075464, -0.0042402, -0.0022497, 0.0022858
2: 0.0302925, 0.0324648, 0.0303482, 0.0323994, -0.0013958, 0.0014181
3: -0.0010575, 0.0029986, -0.0009354, 0.0028948, -0.0026480, 0.0026062
4: -0.0066602, -0.0030987, -0.0065690, -0.0032060, -0.0022884, 0.0023250
5: 0.0112155, 0.0125645, 0.0112500, 0.0125238, -0.0008668, 0.0008807
6: -0.0009540, 0.0041938, -0.0007990, 0.0040619, -0.0033606, 0.0033076
7: 0.9773917, 0.9809939, 0.9775002, 0.9809016, -0.0023516, 0.0023145
8: -0.0108039, -0.0069418, -0.0106876, -0.0070407, -0.0025213, 0.0024815
9: -0.0004142, 0.0021370, -0.0003488, 0.0020602, -0.0016392, 0.0016654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017042
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017079
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0013105, 0.0000693, -0.0013322, 0.0000234, -0.0008571, 0.0009319
1: -0.0076361, -0.0041347, -0.0076911, -0.0042512, -0.0021751, 0.0023648
2: 0.0302925, 0.0324648, 0.0302584, 0.0323926, -0.0013495, 0.0014672
3: -0.0010575, 0.0029986, -0.0009226, 0.0030623, -0.0027395, 0.0025198
4: -0.0066602, -0.0030987, -0.0067162, -0.0032172, -0.0022125, 0.0024054
5: 0.0112155, 0.0125645, 0.0111943, 0.0125196, -0.0008380, 0.0009111
6: -0.0009540, 0.0041938, -0.0007828, 0.0042746, -0.0034768, 0.0031979
7: 0.9773917, 0.9809939, 0.9775116, 0.9810504, -0.0024329, 0.0022377
8: -0.0108039, -0.0069418, -0.0106754, -0.0068811, -0.0026085, 0.0023992
9: -0.0004142, 0.0021370, -0.0004542, 0.0020521, -0.0015848, 0.0017230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017195
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017229
time: 1.00 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.96 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015963
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0015964
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016928, upper bound: 0.0016095
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016253
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016929, upper bound: 0.0016475
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015696
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015670
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015696
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015669
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015858
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015858
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015818
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015941
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015941
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0015937
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016093
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016093
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016839, upper bound: 0.0016091
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0015984
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0015963
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0015984
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0015964
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016142
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016095
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016142
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016095
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016256
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016253
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016256
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016253
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016475
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016475
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016475
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0016475
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016506, upper bound: 0.0016649
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016506, upper bound: 0.0016747
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016506, upper bound: 0.0016649
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016506, upper bound: 0.0016747
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016397, upper bound: 0.0016485
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016397, upper bound: 0.0016590
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016397, upper bound: 0.0016485
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016397, upper bound: 0.0016590
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016747, upper bound: 0.0016586
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016747, upper bound: 0.0016794
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016747, upper bound: 0.0016682
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016747, upper bound: 0.0016901
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016466
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016648
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016611
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016803
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016811
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016815
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016952
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016956
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017051
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017079
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017197
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0017229
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016804
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016804
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016811
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016815
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016952
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0016956
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017042
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017079
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017195
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.96
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0017229

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012222, 0.0000112, -0.0012693, -0.0000120, -0.0007517, 0.0008237
1: -0.0074120, -0.0042822, -0.0075316, -0.0043411, -0.0019076, 0.0020902
2: 0.0304316, 0.0323734, 0.0303574, 0.0323368, -0.0011835, 0.0012968
3: -0.0008867, 0.0027390, -0.0008185, 0.0028776, -0.0024214, 0.0022099
4: -0.0064323, -0.0032487, -0.0065540, -0.0033086, -0.0019404, 0.0021261
5: 0.0113018, 0.0125077, 0.0112557, 0.0124850, -0.0007350, 0.0008053
6: -0.0007373, 0.0038643, -0.0006507, 0.0040402, -0.0030730, 0.0028047
7: 0.9775434, 0.9807633, 0.9776040, 0.9808864, -0.0021504, 0.0019626
8: -0.0106413, -0.0071890, -0.0105763, -0.0070570, -0.0023055, 0.0021042
9: -0.0002509, 0.0020296, -0.0003380, 0.0019867, -0.0013899, 0.0015229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016054
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016054
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012121, 0.0000254, -0.0012693, -0.0000120, -0.0007564, 0.0008600
1: -0.0073866, -0.0042461, -0.0075316, -0.0043411, -0.0019194, 0.0021823
2: 0.0304474, 0.0323958, 0.0303574, 0.0323368, -0.0011908, 0.0013539
3: -0.0009286, 0.0027095, -0.0008185, 0.0028776, -0.0025281, 0.0022236
4: -0.0064064, -0.0032120, -0.0065540, -0.0033086, -0.0019524, 0.0022198
5: 0.0113116, 0.0125216, 0.0112557, 0.0124850, -0.0007395, 0.0008408
6: -0.0007904, 0.0038269, -0.0006507, 0.0040402, -0.0032085, 0.0028220
7: 0.9775063, 0.9807371, 0.9776040, 0.9808864, -0.0022451, 0.0019747
8: -0.0106811, -0.0072170, -0.0105763, -0.0070570, -0.0024071, 0.0021172
9: -0.0002323, 0.0020559, -0.0003380, 0.0019867, -0.0013985, 0.0015900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016054
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016054
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012222, 0.0000112, -0.0012836, 0.0000263, -0.0007823, 0.0008319
1: -0.0074120, -0.0042822, -0.0075679, -0.0042439, -0.0019852, 0.0021110
2: 0.0304316, 0.0323734, 0.0303349, 0.0323971, -0.0012316, 0.0013097
3: -0.0008867, 0.0027390, -0.0009311, 0.0029196, -0.0024455, 0.0022997
4: -0.0064323, -0.0032487, -0.0065908, -0.0032098, -0.0020193, 0.0021473
5: 0.0113018, 0.0125077, 0.0112418, 0.0125224, -0.0007648, 0.0008133
6: -0.0007373, 0.0038643, -0.0007935, 0.0040935, -0.0031037, 0.0029187
7: 0.9775434, 0.9807633, 0.9775040, 0.9809237, -0.0021718, 0.0020423
8: -0.0106413, -0.0071890, -0.0106835, -0.0070171, -0.0023285, 0.0021897
9: -0.0002509, 0.0020296, -0.0003644, 0.0020574, -0.0014464, 0.0015381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015669
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015670
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012121, 0.0000254, -0.0012836, 0.0000263, -0.0007869, 0.0008682
1: -0.0073866, -0.0042461, -0.0075679, -0.0042439, -0.0019970, 0.0022031
2: 0.0304474, 0.0323958, 0.0303349, 0.0323971, -0.0012389, 0.0013668
3: -0.0009286, 0.0027095, -0.0009311, 0.0029196, -0.0025522, 0.0023134
4: -0.0064064, -0.0032120, -0.0065908, -0.0032098, -0.0020313, 0.0022410
5: 0.0113116, 0.0125216, 0.0112418, 0.0125224, -0.0007694, 0.0008488
6: -0.0007904, 0.0038269, -0.0007935, 0.0040935, -0.0032391, 0.0029360
7: 0.9775063, 0.9807371, 0.9775040, 0.9809237, -0.0022666, 0.0020545
8: -0.0106811, -0.0072170, -0.0106835, -0.0070171, -0.0024301, 0.0022027
9: -0.0002323, 0.0020559, -0.0003644, 0.0020574, -0.0014550, 0.0016052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015669
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015669
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012317, 0.0000269, -0.0012724, -0.0000113, -0.0007585, 0.0008418
1: -0.0074363, -0.0042424, -0.0075394, -0.0043393, -0.0019247, 0.0021361
2: 0.0304165, 0.0323980, 0.0303526, 0.0323379, -0.0011941, 0.0013252
3: -0.0009328, 0.0027671, -0.0008205, 0.0028866, -0.0024746, 0.0022297
4: -0.0064570, -0.0032083, -0.0065619, -0.0033069, -0.0019578, 0.0021728
5: 0.0112925, 0.0125230, 0.0112527, 0.0124856, -0.0007416, 0.0008230
6: -0.0007957, 0.0039000, -0.0006532, 0.0040516, -0.0031405, 0.0028298
7: 0.9775024, 0.9807883, 0.9776021, 0.9808944, -0.0021976, 0.0019802
8: -0.0106851, -0.0071622, -0.0105782, -0.0070485, -0.0023562, 0.0021230
9: -0.0002686, 0.0020585, -0.0003437, 0.0019879, -0.0014024, 0.0015564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016187
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016187
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012216, 0.0000434, -0.0012724, -0.0000113, -0.0007634, 0.0008768
1: -0.0074106, -0.0042005, -0.0075394, -0.0043393, -0.0019374, 0.0022249
2: 0.0304325, 0.0324240, 0.0303526, 0.0323379, -0.0012019, 0.0013804
3: -0.0009813, 0.0027374, -0.0008205, 0.0028866, -0.0025775, 0.0022443
4: -0.0064308, -0.0031657, -0.0065619, -0.0033069, -0.0019706, 0.0022631
5: 0.0113024, 0.0125391, 0.0112527, 0.0124856, -0.0007464, 0.0008572
6: -0.0008573, 0.0038622, -0.0006532, 0.0040516, -0.0032712, 0.0028484
7: 0.9774593, 0.9807618, 0.9776021, 0.9808944, -0.0022890, 0.0019931
8: -0.0107313, -0.0071906, -0.0105782, -0.0070485, -0.0024542, 0.0021370
9: -0.0002498, 0.0020891, -0.0003437, 0.0019879, -0.0014116, 0.0016211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016187
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016187
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012317, 0.0000269, -0.0012869, 0.0000270, -0.0007891, 0.0008506
1: -0.0074363, -0.0042424, -0.0075763, -0.0042420, -0.0020024, 0.0021586
2: 0.0304165, 0.0323980, 0.0303297, 0.0323982, -0.0012423, 0.0013392
3: -0.0009328, 0.0027671, -0.0009332, 0.0029294, -0.0025006, 0.0023197
4: -0.0064570, -0.0032083, -0.0065994, -0.0032079, -0.0020368, 0.0021957
5: 0.0112925, 0.0125230, 0.0112385, 0.0125231, -0.0007715, 0.0008317
6: -0.0007957, 0.0039000, -0.0007962, 0.0041059, -0.0031736, 0.0029440
7: 0.9775024, 0.9807883, 0.9775021, 0.9809324, -0.0022208, 0.0020601
8: -0.0106851, -0.0071622, -0.0106855, -0.0070078, -0.0023810, 0.0022087
9: -0.0002686, 0.0020585, -0.0003706, 0.0020588, -0.0014590, 0.0015728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015818
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015818
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012216, 0.0000434, -0.0012869, 0.0000270, -0.0007941, 0.0008856
1: -0.0074106, -0.0042005, -0.0075763, -0.0042420, -0.0020150, 0.0022475
2: 0.0304325, 0.0324240, 0.0303297, 0.0323982, -0.0012501, 0.0013943
3: -0.0009813, 0.0027374, -0.0009332, 0.0029294, -0.0026036, 0.0023343
4: -0.0064308, -0.0031657, -0.0065994, -0.0032079, -0.0020496, 0.0022860
5: 0.0113024, 0.0125391, 0.0112385, 0.0125231, -0.0007763, 0.0008659
6: -0.0008573, 0.0038622, -0.0007962, 0.0041059, -0.0033043, 0.0029626
7: 0.9774593, 0.9807618, 0.9775021, 0.9809324, -0.0023122, 0.0020731
8: -0.0107313, -0.0071906, -0.0106855, -0.0070078, -0.0024790, 0.0022227
9: -0.0002498, 0.0020891, -0.0003706, 0.0020588, -0.0014682, 0.0016375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015818
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015818
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012228, 0.0000148, -0.0012888, -0.0000003, -0.0007539, 0.0008595
1: -0.0074135, -0.0042731, -0.0075812, -0.0043115, -0.0019130, 0.0021811
2: 0.0304307, 0.0323790, 0.0303266, 0.0323552, -0.0011869, 0.0013532
3: -0.0008973, 0.0027407, -0.0008528, 0.0029350, -0.0025267, 0.0022162
4: -0.0064338, -0.0032394, -0.0066043, -0.0032785, -0.0019459, 0.0022186
5: 0.0113012, 0.0125112, 0.0112366, 0.0124964, -0.0007371, 0.0008403
6: -0.0007507, 0.0038665, -0.0006942, 0.0041130, -0.0032067, 0.0028126
7: 0.9775340, 0.9807649, 0.9775735, 0.9809374, -0.0022439, 0.0019681
8: -0.0106513, -0.0071874, -0.0106090, -0.0070024, -0.0024058, 0.0021101
9: -0.0002519, 0.0020362, -0.0003741, 0.0020082, -0.0013939, 0.0015892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016330
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016330
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012128, 0.0000297, -0.0012888, -0.0000003, -0.0007586, 0.0008946
1: -0.0073881, -0.0042351, -0.0075812, -0.0043115, -0.0019251, 0.0022703
2: 0.0304464, 0.0324025, 0.0303266, 0.0323552, -0.0011943, 0.0014085
3: -0.0009412, 0.0027113, -0.0008528, 0.0029350, -0.0026300, 0.0022301
4: -0.0064080, -0.0032009, -0.0066043, -0.0032785, -0.0019581, 0.0023092
5: 0.0113110, 0.0125258, 0.0112366, 0.0124964, -0.0007417, 0.0008747
6: -0.0008064, 0.0038292, -0.0006942, 0.0041130, -0.0033378, 0.0028303
7: 0.9774950, 0.9807388, 0.9775735, 0.9809374, -0.0023356, 0.0019805
8: -0.0106931, -0.0072153, -0.0106090, -0.0070024, -0.0025042, 0.0021234
9: -0.0002335, 0.0020638, -0.0003741, 0.0020082, -0.0014026, 0.0016541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016330
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016330
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012228, 0.0000148, -0.0013051, 0.0000404, -0.0007867, 0.0008700
1: -0.0074135, -0.0042731, -0.0076225, -0.0042081, -0.0019965, 0.0022077
2: 0.0304307, 0.0323790, 0.0303010, 0.0324193, -0.0012386, 0.0013697
3: -0.0008973, 0.0027407, -0.0009726, 0.0029829, -0.0025575, 0.0023128
4: -0.0064338, -0.0032394, -0.0066464, -0.0031733, -0.0020307, 0.0022456
5: 0.0113012, 0.0125112, 0.0112207, 0.0125362, -0.0007692, 0.0008506
6: -0.0007507, 0.0038665, -0.0008462, 0.0041739, -0.0032458, 0.0029353
7: 0.9775340, 0.9807649, 0.9774671, 0.9809799, -0.0022712, 0.0020540
8: -0.0106513, -0.0071874, -0.0107230, -0.0069567, -0.0024351, 0.0022022
9: -0.0002519, 0.0020362, -0.0004043, 0.0020836, -0.0014547, 0.0016085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015937
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015937
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012128, 0.0000297, -0.0013051, 0.0000404, -0.0007915, 0.0009051
1: -0.0073881, -0.0042351, -0.0076225, -0.0042081, -0.0020085, 0.0022968
2: 0.0304464, 0.0324025, 0.0303010, 0.0324193, -0.0012461, 0.0014250
3: -0.0009412, 0.0027113, -0.0009726, 0.0029829, -0.0026608, 0.0023268
4: -0.0064080, -0.0032009, -0.0066464, -0.0031733, -0.0020430, 0.0023363
5: 0.0113110, 0.0125258, 0.0112207, 0.0125362, -0.0007738, 0.0008849
6: -0.0008064, 0.0038292, -0.0008462, 0.0041739, -0.0033768, 0.0029530
7: 0.9774950, 0.9807388, 0.9774671, 0.9809799, -0.0023630, 0.0020663
8: -0.0106931, -0.0072153, -0.0107230, -0.0069567, -0.0025335, 0.0022154
9: -0.0002335, 0.0020638, -0.0004043, 0.0020836, -0.0014634, 0.0016735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015937
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0015937
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012324, 0.0000305, -0.0012915, 0.0000003, -0.0007611, 0.0008773
1: -0.0074379, -0.0042331, -0.0075880, -0.0043099, -0.0019313, 0.0022262
2: 0.0304155, 0.0324038, 0.0303224, 0.0323562, -0.0011982, 0.0013811
3: -0.0009436, 0.0027690, -0.0008547, 0.0029429, -0.0025789, 0.0022373
4: -0.0064586, -0.0031988, -0.0066113, -0.0032769, -0.0019645, 0.0022644
5: 0.0112918, 0.0125266, 0.0112340, 0.0124970, -0.0007441, 0.0008577
6: -0.0008094, 0.0039023, -0.0006966, 0.0041230, -0.0032730, 0.0028394
7: 0.9774929, 0.9807900, 0.9775718, 0.9809444, -0.0022903, 0.0019869
8: -0.0106954, -0.0071605, -0.0106107, -0.0069949, -0.0024555, 0.0021303
9: -0.0002697, 0.0020653, -0.0003791, 0.0020094, -0.0014072, 0.0016220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016448
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016448
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012222, 0.0000473, -0.0012915, 0.0000003, -0.0007661, 0.0009126
1: -0.0074122, -0.0041905, -0.0075880, -0.0043099, -0.0019442, 0.0023159
2: 0.0304315, 0.0324302, 0.0303224, 0.0323562, -0.0012062, 0.0014368
3: -0.0009929, 0.0027392, -0.0008547, 0.0029429, -0.0026829, 0.0022522
4: -0.0064324, -0.0031555, -0.0066113, -0.0032769, -0.0019775, 0.0023557
5: 0.0113017, 0.0125430, 0.0112340, 0.0124970, -0.0007490, 0.0008923
6: -0.0008720, 0.0038646, -0.0006966, 0.0041230, -0.0034049, 0.0028584
7: 0.9774491, 0.9807636, 0.9775718, 0.9809444, -0.0023826, 0.0020001
8: -0.0107423, -0.0071888, -0.0106107, -0.0069949, -0.0025545, 0.0021445
9: -0.0002510, 0.0020963, -0.0003791, 0.0020094, -0.0014165, 0.0016874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016448
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016567, upper bound: 0.0016448
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012324, 0.0000305, -0.0013089, 0.0000411, -0.0007940, 0.0008886
1: -0.0074379, -0.0042331, -0.0076320, -0.0042063, -0.0020150, 0.0022549
2: 0.0304155, 0.0324038, 0.0302951, 0.0324204, -0.0012501, 0.0013989
3: -0.0009436, 0.0027690, -0.0009746, 0.0029939, -0.0026122, 0.0023342
4: -0.0064586, -0.0031988, -0.0066561, -0.0031715, -0.0020495, 0.0022936
5: 0.0112918, 0.0125266, 0.0112170, 0.0125369, -0.0007763, 0.0008687
6: -0.0008094, 0.0039023, -0.0008488, 0.0041878, -0.0033152, 0.0029624
7: 0.9774929, 0.9807900, 0.9774653, 0.9809897, -0.0023198, 0.0020730
8: -0.0106954, -0.0071605, -0.0107250, -0.0069463, -0.0024872, 0.0022225
9: -0.0002697, 0.0020653, -0.0004112, 0.0020849, -0.0014681, 0.0016429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0016091
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0016091
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012222, 0.0000473, -0.0013089, 0.0000411, -0.0007991, 0.0009239
1: -0.0074122, -0.0041905, -0.0076320, -0.0042063, -0.0020278, 0.0023446
2: 0.0304315, 0.0324302, 0.0302951, 0.0324204, -0.0012581, 0.0014546
3: -0.0009929, 0.0027392, -0.0009746, 0.0029939, -0.0027161, 0.0023491
4: -0.0064324, -0.0031555, -0.0066561, -0.0031715, -0.0020626, 0.0023849
5: 0.0113017, 0.0125430, 0.0112170, 0.0125369, -0.0007813, 0.0009033
6: -0.0008720, 0.0038646, -0.0008488, 0.0041878, -0.0034471, 0.0029813
7: 0.9774491, 0.9807636, 0.9774653, 0.9809897, -0.0024121, 0.0020862
8: -0.0107423, -0.0071888, -0.0107250, -0.0069463, -0.0025862, 0.0022367
9: -0.0002510, 0.0020963, -0.0004112, 0.0020849, -0.0014775, 0.0017083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0016091
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016689, upper bound: 0.0016091
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012696, 0.0000013, -0.0012931, -0.0000085, -0.0007829, 0.0008522
1: -0.0075323, -0.0043073, -0.0075920, -0.0043322, -0.0019868, 0.0021626
2: 0.0303569, 0.0323577, 0.0303199, 0.0323423, -0.0012326, 0.0013417
3: -0.0008576, 0.0028784, -0.0008288, 0.0029475, -0.0025053, 0.0023016
4: -0.0065547, -0.0032743, -0.0066153, -0.0032996, -0.0020209, 0.0021997
5: 0.0112554, 0.0124980, 0.0112325, 0.0124884, -0.0007655, 0.0008332
6: -0.0007002, 0.0040412, -0.0006637, 0.0041289, -0.0031795, 0.0029210
7: 0.9775692, 0.9808871, 0.9775949, 0.9809485, -0.0022249, 0.0020440
8: -0.0106135, -0.0070562, -0.0105861, -0.0069905, -0.0023854, 0.0021915
9: -0.0003385, 0.0020112, -0.0003820, 0.0019931, -0.0014476, 0.0015757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016274
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016327
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012644, 0.0000206, -0.0012931, -0.0000085, -0.0007869, 0.0008894
1: -0.0075191, -0.0042583, -0.0075920, -0.0043322, -0.0019968, 0.0022570
2: 0.0303651, 0.0323882, 0.0303199, 0.0323423, -0.0012388, 0.0014003
3: -0.0009144, 0.0028631, -0.0008288, 0.0029475, -0.0026147, 0.0023132
4: -0.0065412, -0.0032244, -0.0066153, -0.0032996, -0.0020310, 0.0022958
5: 0.0112605, 0.0125169, 0.0112325, 0.0124884, -0.0007693, 0.0008696
6: -0.0007724, 0.0040218, -0.0006637, 0.0041289, -0.0033183, 0.0029357
7: 0.9775188, 0.9808735, 0.9775949, 0.9809485, -0.0023220, 0.0020543
8: -0.0106676, -0.0070708, -0.0105861, -0.0069905, -0.0024896, 0.0022025
9: -0.0003289, 0.0020470, -0.0003820, 0.0019931, -0.0014549, 0.0016445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016274
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016327
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012696, 0.0000013, -0.0013081, 0.0000298, -0.0008116, 0.0008595
1: -0.0075323, -0.0043073, -0.0076302, -0.0042350, -0.0020596, 0.0021810
2: 0.0303569, 0.0323577, 0.0302962, 0.0324026, -0.0012778, 0.0013531
3: -0.0008576, 0.0028784, -0.0009414, 0.0029918, -0.0025266, 0.0023859
4: -0.0065547, -0.0032743, -0.0066542, -0.0032007, -0.0020949, 0.0022184
5: 0.0112554, 0.0124980, 0.0112178, 0.0125258, -0.0007935, 0.0008403
6: -0.0007002, 0.0040412, -0.0008066, 0.0041851, -0.0032065, 0.0030280
7: 0.9775692, 0.9808871, 0.9774948, 0.9809877, -0.0022438, 0.0021189
8: -0.0106135, -0.0070562, -0.0106933, -0.0069483, -0.0024057, 0.0022717
9: -0.0003385, 0.0020112, -0.0004098, 0.0020639, -0.0015006, 0.0015891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0015842
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0015964
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012644, 0.0000206, -0.0013081, 0.0000298, -0.0008155, 0.0008967
1: -0.0075191, -0.0042583, -0.0076302, -0.0042350, -0.0020695, 0.0022754
2: 0.0303651, 0.0323882, 0.0302962, 0.0324026, -0.0012839, 0.0014117
3: -0.0009144, 0.0028631, -0.0009414, 0.0029918, -0.0026359, 0.0023974
4: -0.0065412, -0.0032244, -0.0066542, -0.0032007, -0.0021051, 0.0023145
5: 0.0112605, 0.0125169, 0.0112178, 0.0125258, -0.0007973, 0.0008767
6: -0.0007724, 0.0040218, -0.0008066, 0.0041851, -0.0033453, 0.0030427
7: 0.9775188, 0.9808735, 0.9774948, 0.9809877, -0.0023409, 0.0021291
8: -0.0106676, -0.0070708, -0.0106933, -0.0069483, -0.0025098, 0.0022827
9: -0.0003289, 0.0020470, -0.0004098, 0.0020639, -0.0015079, 0.0016579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0015842
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0015964
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012816, 0.0000210, -0.0012963, -0.0000078, -0.0007907, 0.0008752
1: -0.0075628, -0.0042574, -0.0076000, -0.0043303, -0.0020065, 0.0022208
2: 0.0303380, 0.0323887, 0.0303149, 0.0323435, -0.0012449, 0.0013778
3: -0.0009154, 0.0029137, -0.0008310, 0.0029568, -0.0025727, 0.0023245
4: -0.0065857, -0.0032235, -0.0066235, -0.0032977, -0.0020410, 0.0022590
5: 0.0112437, 0.0125172, 0.0112294, 0.0124891, -0.0007731, 0.0008556
6: -0.0007737, 0.0040860, -0.0006665, 0.0041407, -0.0032651, 0.0029501
7: 0.9775178, 0.9809184, 0.9775929, 0.9809567, -0.0022848, 0.0020643
8: -0.0106686, -0.0070226, -0.0105882, -0.0069816, -0.0024496, 0.0022133
9: -0.0003608, 0.0020476, -0.0003879, 0.0019945, -0.0014620, 0.0016181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016477
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016526
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012758, 0.0000419, -0.0012963, -0.0000078, -0.0007954, 0.0009108
1: -0.0075480, -0.0042044, -0.0076000, -0.0043303, -0.0020184, 0.0023112
2: 0.0303472, 0.0324216, 0.0303149, 0.0323435, -0.0012522, 0.0014339
3: -0.0009769, 0.0028966, -0.0008310, 0.0029568, -0.0026775, 0.0023382
4: -0.0065706, -0.0031696, -0.0066235, -0.0032977, -0.0020530, 0.0023509
5: 0.0112494, 0.0125376, 0.0112294, 0.0124891, -0.0007776, 0.0008905
6: -0.0008516, 0.0040643, -0.0006665, 0.0041407, -0.0033980, 0.0029675
7: 0.9774633, 0.9809032, 0.9775929, 0.9809567, -0.0023778, 0.0020765
8: -0.0107271, -0.0070389, -0.0105882, -0.0069816, -0.0025494, 0.0022263
9: -0.0003500, 0.0020862, -0.0003879, 0.0019945, -0.0014706, 0.0016840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016477
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016526
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012816, 0.0000210, -0.0013120, 0.0000306, -0.0008195, 0.0008834
1: -0.0075628, -0.0042574, -0.0076400, -0.0042328, -0.0020796, 0.0022416
2: 0.0303380, 0.0323887, 0.0302901, 0.0324040, -0.0012902, 0.0013907
3: -0.0009154, 0.0029137, -0.0009439, 0.0030032, -0.0025968, 0.0024091
4: -0.0065857, -0.0032235, -0.0066642, -0.0031985, -0.0021153, 0.0022801
5: 0.0112437, 0.0125172, 0.0112139, 0.0125267, -0.0008012, 0.0008636
6: -0.0007737, 0.0040860, -0.0008098, 0.0041996, -0.0032957, 0.0030575
7: 0.9775178, 0.9809184, 0.9774926, 0.9809979, -0.0023062, 0.0021395
8: -0.0106686, -0.0070226, -0.0106957, -0.0069374, -0.0024726, 0.0022939
9: -0.0003608, 0.0020476, -0.0004170, 0.0020655, -0.0015152, 0.0016333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0015966
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016095
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012758, 0.0000419, -0.0013120, 0.0000306, -0.0008242, 0.0009190
1: -0.0075480, -0.0042044, -0.0076400, -0.0042328, -0.0020914, 0.0023320
2: 0.0303472, 0.0324216, 0.0302901, 0.0324040, -0.0012975, 0.0014468
3: -0.0009769, 0.0028966, -0.0009439, 0.0030032, -0.0027016, 0.0024228
4: -0.0065706, -0.0031696, -0.0066642, -0.0031985, -0.0021273, 0.0023721
5: 0.0112494, 0.0125376, 0.0112139, 0.0125267, -0.0008058, 0.0008985
6: -0.0008516, 0.0040643, -0.0008098, 0.0041996, -0.0034286, 0.0030749
7: 0.9774633, 0.9809032, 0.9774926, 0.9809979, -0.0023992, 0.0021516
8: -0.0107271, -0.0070389, -0.0106957, -0.0069374, -0.0025723, 0.0023069
9: -0.0003500, 0.0020862, -0.0004170, 0.0020655, -0.0015238, 0.0016992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0015966
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016095
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012703, 0.0000059, -0.0013133, 0.0000032, -0.0007875, 0.0008890
1: -0.0075341, -0.0042957, -0.0076433, -0.0043025, -0.0019985, 0.0022561
2: 0.0303559, 0.0323649, 0.0302881, 0.0323607, -0.0012399, 0.0013997
3: -0.0008710, 0.0028804, -0.0008632, 0.0030070, -0.0026136, 0.0023152
4: -0.0065564, -0.0032625, -0.0066676, -0.0032694, -0.0020328, 0.0022948
5: 0.0112548, 0.0125024, 0.0112127, 0.0124998, -0.0007700, 0.0008692
6: -0.0007173, 0.0040437, -0.0007073, 0.0042044, -0.0033169, 0.0029383
7: 0.9775574, 0.9808889, 0.9775643, 0.9810013, -0.0023210, 0.0020560
8: -0.0106263, -0.0070544, -0.0106188, -0.0069338, -0.0024885, 0.0022044
9: -0.0003398, 0.0020197, -0.0004194, 0.0020147, -0.0014561, 0.0016438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016621
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016681
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012650, 0.0000257, -0.0013133, 0.0000032, -0.0007915, 0.0009247
1: -0.0075207, -0.0042453, -0.0076433, -0.0043025, -0.0020086, 0.0023467
2: 0.0303641, 0.0323962, 0.0302881, 0.0323607, -0.0012462, 0.0014559
3: -0.0009295, 0.0028650, -0.0008632, 0.0030070, -0.0027185, 0.0023269
4: -0.0065429, -0.0032112, -0.0066676, -0.0032694, -0.0020431, 0.0023870
5: 0.0112599, 0.0125219, 0.0112127, 0.0124998, -0.0007739, 0.0009041
6: -0.0007915, 0.0040241, -0.0007073, 0.0042044, -0.0034502, 0.0029532
7: 0.9775054, 0.9808751, 0.9775643, 0.9810013, -0.0024143, 0.0020665
8: -0.0106820, -0.0070691, -0.0106188, -0.0069338, -0.0025885, 0.0022156
9: -0.0003301, 0.0020565, -0.0004194, 0.0020147, -0.0014635, 0.0017098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016621
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016681
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012703, 0.0000059, -0.0013308, 0.0000440, -0.0008189, 0.0008993
1: -0.0075341, -0.0042957, -0.0076877, -0.0041990, -0.0020782, 0.0022820
2: 0.0303559, 0.0323649, 0.0302605, 0.0324249, -0.0012893, 0.0014158
3: -0.0008710, 0.0028804, -0.0009830, 0.0030584, -0.0026436, 0.0024074
4: -0.0065564, -0.0032625, -0.0067127, -0.0031641, -0.0021138, 0.0023212
5: 0.0112548, 0.0125024, 0.0111956, 0.0125397, -0.0008007, 0.0008792
6: -0.0007173, 0.0040437, -0.0008595, 0.0042697, -0.0033551, 0.0030554
7: 0.9775574, 0.9808889, 0.9774578, 0.9810469, -0.0023477, 0.0021380
8: -0.0106263, -0.0070544, -0.0107330, -0.0068849, -0.0025171, 0.0022923
9: -0.0003398, 0.0020197, -0.0004518, 0.0020901, -0.0015142, 0.0016627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016122
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016252
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012650, 0.0000257, -0.0013308, 0.0000440, -0.0008229, 0.0009350
1: -0.0075207, -0.0042453, -0.0076877, -0.0041990, -0.0020883, 0.0023726
2: 0.0303641, 0.0323962, 0.0302605, 0.0324249, -0.0012956, 0.0014720
3: -0.0009295, 0.0028650, -0.0009830, 0.0030584, -0.0027486, 0.0024192
4: -0.0065429, -0.0032112, -0.0067127, -0.0031641, -0.0021242, 0.0024134
5: 0.0112599, 0.0125219, 0.0111956, 0.0125397, -0.0008046, 0.0009141
6: -0.0007915, 0.0040241, -0.0008595, 0.0042697, -0.0034883, 0.0030703
7: 0.9775054, 0.9808751, 0.9774578, 0.9810469, -0.0024410, 0.0021484
8: -0.0106820, -0.0070691, -0.0107330, -0.0068849, -0.0026171, 0.0023035
9: -0.0003301, 0.0020565, -0.0004518, 0.0020901, -0.0015216, 0.0017287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016122
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016253
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012822, 0.0000257, -0.0013164, 0.0000039, -0.0007955, 0.0009136
1: -0.0075645, -0.0042455, -0.0076510, -0.0043008, -0.0020186, 0.0023184
2: 0.0303370, 0.0323961, 0.0302833, 0.0323618, -0.0012523, 0.0014384
3: -0.0009293, 0.0029156, -0.0008651, 0.0030159, -0.0026858, 0.0023384
4: -0.0065874, -0.0032114, -0.0066754, -0.0032677, -0.0020532, 0.0023582
5: 0.0112431, 0.0125218, 0.0112097, 0.0125005, -0.0007777, 0.0008932
6: -0.0007912, 0.0040885, -0.0007099, 0.0042157, -0.0034086, 0.0029678
7: 0.9775056, 0.9809202, 0.9775626, 0.9810092, -0.0023852, 0.0020767
8: -0.0106818, -0.0070208, -0.0106207, -0.0069253, -0.0025573, 0.0022265
9: -0.0003620, 0.0020563, -0.0004250, 0.0020160, -0.0014708, 0.0016892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016862
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016929
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012764, 0.0000465, -0.0013164, 0.0000039, -0.0008002, 0.0009486
1: -0.0075496, -0.0041926, -0.0076510, -0.0043008, -0.0020305, 0.0024072
2: 0.0303462, 0.0324289, 0.0302833, 0.0323618, -0.0012598, 0.0014935
3: -0.0009905, 0.0028985, -0.0008651, 0.0030159, -0.0027887, 0.0023523
4: -0.0065723, -0.0031576, -0.0066754, -0.0032677, -0.0020654, 0.0024486
5: 0.0112488, 0.0125422, 0.0112097, 0.0125005, -0.0007823, 0.0009274
6: -0.0008689, 0.0040667, -0.0007099, 0.0042157, -0.0035392, 0.0029854
7: 0.9774513, 0.9809050, 0.9775626, 0.9810092, -0.0024765, 0.0020890
8: -0.0107400, -0.0070372, -0.0106207, -0.0069253, -0.0026553, 0.0022397
9: -0.0003512, 0.0020948, -0.0004250, 0.0020160, -0.0014795, 0.0017539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016881
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0016939
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012822, 0.0000257, -0.0013346, 0.0000448, -0.0008270, 0.0009239
1: -0.0075645, -0.0042455, -0.0076973, -0.0041970, -0.0020986, 0.0023446
2: 0.0303370, 0.0323961, 0.0302546, 0.0324262, -0.0013019, 0.0014546
3: -0.0009293, 0.0029156, -0.0009854, 0.0030695, -0.0027161, 0.0024311
4: -0.0065874, -0.0032114, -0.0067225, -0.0031621, -0.0021346, 0.0023848
5: 0.0112431, 0.0125218, 0.0111919, 0.0125405, -0.0008085, 0.0009033
6: -0.0007912, 0.0040885, -0.0008625, 0.0042838, -0.0034471, 0.0030853
7: 0.9775056, 0.9809202, 0.9774557, 0.9810569, -0.0024121, 0.0021590
8: -0.0106818, -0.0070208, -0.0107352, -0.0068743, -0.0025861, 0.0023148
9: -0.0003620, 0.0020563, -0.0004587, 0.0020916, -0.0015290, 0.0017083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016344
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016475
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012764, 0.0000465, -0.0013346, 0.0000448, -0.0008317, 0.0009589
1: -0.0075496, -0.0041926, -0.0076973, -0.0041970, -0.0021105, 0.0024334
2: 0.0303462, 0.0324289, 0.0302546, 0.0324262, -0.0013094, 0.0015097
3: -0.0009905, 0.0028985, -0.0009854, 0.0030695, -0.0028190, 0.0024449
4: -0.0065723, -0.0031576, -0.0067225, -0.0031621, -0.0021468, 0.0024752
5: 0.0112488, 0.0125422, 0.0111919, 0.0125405, -0.0008131, 0.0009375
6: -0.0008689, 0.0040667, -0.0008625, 0.0042838, -0.0035776, 0.0031029
7: 0.9774513, 0.9809050, 0.9774557, 0.9810569, -0.0025035, 0.0021713
8: -0.0107400, -0.0070372, -0.0107352, -0.0068743, -0.0026841, 0.0023280
9: -0.0003512, 0.0020948, -0.0004587, 0.0020916, -0.0015378, 0.0017730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016344
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016649, upper bound: 0.0016475
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012222, 0.0000112, -0.0012606, 0.0000070, -0.0007889, 0.0008292
1: -0.0074120, -0.0042822, -0.0075094, -0.0042928, -0.0020020, 0.0021042
2: 0.0304316, 0.0323734, 0.0303711, 0.0323667, -0.0012421, 0.0013055
3: -0.0008867, 0.0027390, -0.0008744, 0.0028519, -0.0024376, 0.0023192
4: -0.0064323, -0.0032487, -0.0065314, -0.0032595, -0.0020364, 0.0021403
5: 0.0113018, 0.0125077, 0.0112643, 0.0125036, -0.0007713, 0.0008107
6: -0.0007373, 0.0038643, -0.0007216, 0.0040075, -0.0030937, 0.0029434
7: 0.9775434, 0.9807633, 0.9775543, 0.9808636, -0.0021648, 0.0020597
8: -0.0106413, -0.0071890, -0.0106295, -0.0070815, -0.0023210, 0.0022083
9: -0.0002509, 0.0020296, -0.0003219, 0.0020218, -0.0014587, 0.0015332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016058
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016058
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012121, 0.0000254, -0.0012606, 0.0000070, -0.0007683, 0.0008388
1: -0.0073866, -0.0042461, -0.0075094, -0.0042928, -0.0019497, 0.0021286
2: 0.0304474, 0.0323958, 0.0303711, 0.0323667, -0.0012096, 0.0013206
3: -0.0009286, 0.0027095, -0.0008744, 0.0028519, -0.0024659, 0.0022587
4: -0.0064064, -0.0032120, -0.0065314, -0.0032595, -0.0019832, 0.0021651
5: 0.0113116, 0.0125216, 0.0112643, 0.0125036, -0.0007512, 0.0008201
6: -0.0007904, 0.0038269, -0.0007216, 0.0040075, -0.0031295, 0.0028666
7: 0.9775063, 0.9807371, 0.9775543, 0.9808636, -0.0021899, 0.0020059
8: -0.0106811, -0.0072170, -0.0106295, -0.0070815, -0.0023479, 0.0021506
9: -0.0002323, 0.0020559, -0.0003219, 0.0020218, -0.0014206, 0.0015509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016054
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016054
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012222, 0.0000112, -0.0012794, 0.0000466, -0.0008227, 0.0008390
1: -0.0074120, -0.0042822, -0.0075573, -0.0041923, -0.0020877, 0.0021291
2: 0.0304316, 0.0323734, 0.0303415, 0.0324291, -0.0012952, 0.0013209
3: -0.0008867, 0.0027390, -0.0009908, 0.0029073, -0.0024664, 0.0024185
4: -0.0064323, -0.0032487, -0.0065801, -0.0031573, -0.0021235, 0.0021656
5: 0.0113018, 0.0125077, 0.0112458, 0.0125423, -0.0008043, 0.0008203
6: -0.0007373, 0.0038643, -0.0008693, 0.0040779, -0.0031302, 0.0030694
7: 0.9775434, 0.9807633, 0.9774510, 0.9809127, -0.0021904, 0.0021478
8: -0.0106413, -0.0071890, -0.0107404, -0.0070287, -0.0023484, 0.0023028
9: -0.0002509, 0.0020296, -0.0003567, 0.0020950, -0.0015211, 0.0015513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015696
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015696
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012121, 0.0000254, -0.0012794, 0.0000466, -0.0008004, 0.0008482
1: -0.0073866, -0.0042461, -0.0075573, -0.0041923, -0.0020311, 0.0021525
2: 0.0304474, 0.0323958, 0.0303415, 0.0324291, -0.0012601, 0.0013354
3: -0.0009286, 0.0027095, -0.0009908, 0.0029073, -0.0024936, 0.0023529
4: -0.0064064, -0.0032120, -0.0065801, -0.0031573, -0.0020660, 0.0021894
5: 0.0113116, 0.0125216, 0.0112458, 0.0125423, -0.0007825, 0.0008293
6: -0.0007904, 0.0038269, -0.0008693, 0.0040779, -0.0031646, 0.0029862
7: 0.9775063, 0.9807371, 0.9774510, 0.9809127, -0.0022145, 0.0020896
8: -0.0106811, -0.0072170, -0.0107404, -0.0070287, -0.0023743, 0.0022404
9: -0.0002323, 0.0020559, -0.0003567, 0.0020950, -0.0014799, 0.0015683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015669
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015670
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012317, 0.0000269, -0.0012638, 0.0000078, -0.0007959, 0.0008482
1: -0.0074363, -0.0042424, -0.0075176, -0.0042908, -0.0020197, 0.0021524
2: 0.0304165, 0.0323980, 0.0303661, 0.0323680, -0.0012530, 0.0013354
3: -0.0009328, 0.0027671, -0.0008767, 0.0028613, -0.0024935, 0.0023397
4: -0.0064570, -0.0032083, -0.0065397, -0.0032575, -0.0020544, 0.0021894
5: 0.0112925, 0.0125230, 0.0112611, 0.0125043, -0.0007781, 0.0008293
6: -0.0007957, 0.0039000, -0.0007246, 0.0040195, -0.0031645, 0.0029694
7: 0.9775024, 0.9807883, 0.9775523, 0.9808720, -0.0022144, 0.0020778
8: -0.0106851, -0.0071622, -0.0106318, -0.0070725, -0.0023742, 0.0022278
9: -0.0002686, 0.0020585, -0.0003278, 0.0020233, -0.0014716, 0.0015683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016209
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016209
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012216, 0.0000434, -0.0012638, 0.0000078, -0.0007758, 0.0008579
1: -0.0074106, -0.0042005, -0.0075176, -0.0042908, -0.0019687, 0.0021769
2: 0.0304325, 0.0324240, 0.0303661, 0.0323680, -0.0012214, 0.0013506
3: -0.0009813, 0.0027374, -0.0008767, 0.0028613, -0.0025219, 0.0022807
4: -0.0064308, -0.0031657, -0.0065397, -0.0032575, -0.0020025, 0.0022143
5: 0.0113024, 0.0125391, 0.0112611, 0.0125043, -0.0007585, 0.0008387
6: -0.0008573, 0.0038622, -0.0007246, 0.0040195, -0.0032006, 0.0028945
7: 0.9774593, 0.9807618, 0.9775523, 0.9808720, -0.0022396, 0.0020254
8: -0.0107313, -0.0071906, -0.0106318, -0.0070725, -0.0024012, 0.0021716
9: -0.0002498, 0.0020891, -0.0003278, 0.0020233, -0.0014345, 0.0015861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016187
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016187
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012317, 0.0000269, -0.0012829, 0.0000475, -0.0008296, 0.0008580
1: -0.0074363, -0.0042424, -0.0075662, -0.0041901, -0.0021053, 0.0021772
2: 0.0304165, 0.0323980, 0.0303359, 0.0324304, -0.0013062, 0.0013508
3: -0.0009328, 0.0027671, -0.0009933, 0.0029177, -0.0025222, 0.0024389
4: -0.0064570, -0.0032083, -0.0065891, -0.0031551, -0.0021415, 0.0022146
5: 0.0112925, 0.0125230, 0.0112424, 0.0125431, -0.0008111, 0.0008388
6: -0.0007957, 0.0039000, -0.0008726, 0.0040910, -0.0032010, 0.0030953
7: 0.9775024, 0.9807883, 0.9774487, 0.9809219, -0.0022399, 0.0021660
8: -0.0106851, -0.0071622, -0.0107428, -0.0070189, -0.0024016, 0.0023223
9: -0.0002686, 0.0020585, -0.0003632, 0.0020966, -0.0015340, 0.0015864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015858
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015858
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012216, 0.0000434, -0.0012829, 0.0000475, -0.0008079, 0.0008671
1: -0.0074106, -0.0042005, -0.0075662, -0.0041901, -0.0020502, 0.0022003
2: 0.0304325, 0.0324240, 0.0303359, 0.0324304, -0.0012720, 0.0013651
3: -0.0009813, 0.0027374, -0.0009933, 0.0029177, -0.0025490, 0.0023751
4: -0.0064308, -0.0031657, -0.0065891, -0.0031551, -0.0020854, 0.0022381
5: 0.0113024, 0.0125391, 0.0112424, 0.0125431, -0.0007899, 0.0008477
6: -0.0008573, 0.0038622, -0.0008726, 0.0040910, -0.0032350, 0.0030143
7: 0.9774593, 0.9807618, 0.9774487, 0.9809219, -0.0022637, 0.0021093
8: -0.0107313, -0.0071906, -0.0107428, -0.0070189, -0.0024270, 0.0022615
9: -0.0002498, 0.0020891, -0.0003632, 0.0020966, -0.0014938, 0.0016032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015818
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015818
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012228, 0.0000148, -0.0012837, 0.0000198, -0.0007901, 0.0008677
1: -0.0074135, -0.0042731, -0.0075681, -0.0042603, -0.0020049, 0.0022018
2: 0.0304307, 0.0323790, 0.0303347, 0.0323869, -0.0012438, 0.0013660
3: -0.0008973, 0.0027407, -0.0009120, 0.0029199, -0.0025507, 0.0023226
4: -0.0064338, -0.0032394, -0.0065910, -0.0032265, -0.0020393, 0.0022396
5: 0.0113012, 0.0125112, 0.0112417, 0.0125161, -0.0007724, 0.0008483
6: -0.0007507, 0.0038665, -0.0007693, 0.0040938, -0.0032372, 0.0029476
7: 0.9775340, 0.9807649, 0.9775209, 0.9809239, -0.0022652, 0.0020626
8: -0.0106513, -0.0071874, -0.0106654, -0.0070168, -0.0024287, 0.0022114
9: -0.0002519, 0.0020362, -0.0003646, 0.0020455, -0.0014608, 0.0016043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016330
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016330
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012128, 0.0000297, -0.0012837, 0.0000198, -0.0007707, 0.0008751
1: -0.0073881, -0.0042351, -0.0075681, -0.0042603, -0.0019558, 0.0022207
2: 0.0304464, 0.0324025, 0.0303347, 0.0323869, -0.0012134, 0.0013777
3: -0.0009412, 0.0027113, -0.0009120, 0.0029199, -0.0025726, 0.0022658
4: -0.0064080, -0.0032009, -0.0065910, -0.0032265, -0.0019894, 0.0022588
5: 0.0113110, 0.0125258, 0.0112417, 0.0125161, -0.0007535, 0.0008556
6: -0.0008064, 0.0038292, -0.0007693, 0.0040938, -0.0032650, 0.0028755
7: 0.9774950, 0.9807388, 0.9775209, 0.9809239, -0.0022847, 0.0020122
8: -0.0106931, -0.0072153, -0.0106654, -0.0070168, -0.0024495, 0.0021574
9: -0.0002335, 0.0020638, -0.0003646, 0.0020455, -0.0014251, 0.0016180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016330
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016330
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012228, 0.0000148, -0.0013019, 0.0000613, -0.0008253, 0.0008785
1: -0.0074135, -0.0042731, -0.0076143, -0.0041550, -0.0020943, 0.0022294
2: 0.0304307, 0.0323790, 0.0303061, 0.0324523, -0.0012993, 0.0013831
3: -0.0008973, 0.0027407, -0.0010341, 0.0029734, -0.0025826, 0.0024261
4: -0.0064338, -0.0032394, -0.0066381, -0.0031193, -0.0021302, 0.0022677
5: 0.0113012, 0.0125112, 0.0112239, 0.0125567, -0.0008069, 0.0008589
6: -0.0007507, 0.0038665, -0.0009243, 0.0041618, -0.0032777, 0.0030791
7: 0.9775340, 0.9807649, 0.9774125, 0.9809715, -0.0022936, 0.0021546
8: -0.0106513, -0.0071874, -0.0107816, -0.0069658, -0.0024591, 0.0023100
9: -0.0002519, 0.0020362, -0.0003983, 0.0021222, -0.0015259, 0.0016244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015941
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015941
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012128, 0.0000297, -0.0013019, 0.0000613, -0.0008058, 0.0008856
1: -0.0073881, -0.0042351, -0.0076143, -0.0041550, -0.0020449, 0.0022474
2: 0.0304464, 0.0324025, 0.0303061, 0.0324523, -0.0012687, 0.0013943
3: -0.0009412, 0.0027113, -0.0010341, 0.0029734, -0.0026035, 0.0023690
4: -0.0064080, -0.0032009, -0.0066381, -0.0031193, -0.0020800, 0.0022860
5: 0.0113110, 0.0125258, 0.0112239, 0.0125567, -0.0007879, 0.0008659
6: -0.0008064, 0.0038292, -0.0009243, 0.0041618, -0.0033042, 0.0030065
7: 0.9774950, 0.9807388, 0.9774125, 0.9809715, -0.0023121, 0.0021038
8: -0.0106931, -0.0072153, -0.0107816, -0.0069658, -0.0024789, 0.0022556
9: -0.0002335, 0.0020638, -0.0003983, 0.0021222, -0.0014900, 0.0016375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015937
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0015937
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012324, 0.0000305, -0.0012867, 0.0000206, -0.0007975, 0.0008854
1: -0.0074379, -0.0042331, -0.0075758, -0.0042583, -0.0020238, 0.0022468
2: 0.0304155, 0.0324038, 0.0303300, 0.0323882, -0.0012556, 0.0013939
3: -0.0009436, 0.0027690, -0.0009144, 0.0029288, -0.0026028, 0.0023445
4: -0.0064586, -0.0031988, -0.0065989, -0.0032244, -0.0020586, 0.0022854
5: 0.0112918, 0.0125266, 0.0112387, 0.0125169, -0.0007797, 0.0008656
6: -0.0008094, 0.0039023, -0.0007724, 0.0041051, -0.0033033, 0.0029755
7: 0.9774929, 0.9807900, 0.9775187, 0.9809318, -0.0023115, 0.0020821
8: -0.0106954, -0.0071605, -0.0106676, -0.0070083, -0.0024783, 0.0022323
9: -0.0002697, 0.0020653, -0.0003702, 0.0020470, -0.0014746, 0.0016370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016448
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016448
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012222, 0.0000473, -0.0012867, 0.0000206, -0.0007785, 0.0008945
1: -0.0074122, -0.0041905, -0.0075758, -0.0042583, -0.0019754, 0.0022699
2: 0.0304315, 0.0324302, 0.0303300, 0.0323882, -0.0012256, 0.0014083
3: -0.0009929, 0.0027392, -0.0009144, 0.0029288, -0.0026296, 0.0022885
4: -0.0064324, -0.0031555, -0.0065989, -0.0032244, -0.0020094, 0.0023089
5: 0.0113017, 0.0125430, 0.0112387, 0.0125169, -0.0007611, 0.0008745
6: -0.0008720, 0.0038646, -0.0007724, 0.0041051, -0.0033373, 0.0029043
7: 0.9774491, 0.9807636, 0.9775187, 0.9809318, -0.0023352, 0.0020323
8: -0.0107423, -0.0071888, -0.0106676, -0.0070083, -0.0025038, 0.0021790
9: -0.0002510, 0.0020963, -0.0003702, 0.0020470, -0.0014393, 0.0016539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016448
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016513, upper bound: 0.0016448
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012324, 0.0000305, -0.0013055, 0.0000622, -0.0008327, 0.0008968
1: -0.0074379, -0.0042331, -0.0076235, -0.0041528, -0.0021132, 0.0022757
2: 0.0304155, 0.0324038, 0.0303004, 0.0324536, -0.0013110, 0.0014118
3: -0.0009436, 0.0027690, -0.0010366, 0.0029840, -0.0026362, 0.0024481
4: -0.0064586, -0.0031988, -0.0066474, -0.0031172, -0.0021495, 0.0023147
5: 0.0112918, 0.0125266, 0.0112203, 0.0125575, -0.0008142, 0.0008768
6: -0.0008094, 0.0039023, -0.0009274, 0.0041753, -0.0033457, 0.0031069
7: 0.9774929, 0.9807900, 0.9774103, 0.9809809, -0.0023412, 0.0021741
8: -0.0106954, -0.0071605, -0.0107839, -0.0069557, -0.0025101, 0.0023309
9: -0.0002697, 0.0020653, -0.0004050, 0.0021238, -0.0015397, 0.0016581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016093
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016093
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012222, 0.0000473, -0.0013055, 0.0000622, -0.0008136, 0.0009054
1: -0.0074122, -0.0041905, -0.0076235, -0.0041528, -0.0020645, 0.0022975
2: 0.0304315, 0.0324302, 0.0303004, 0.0324536, -0.0012808, 0.0014254
3: -0.0009929, 0.0027392, -0.0010366, 0.0029840, -0.0026616, 0.0023917
4: -0.0064324, -0.0031555, -0.0066474, -0.0031172, -0.0021000, 0.0023370
5: 0.0113017, 0.0125430, 0.0112203, 0.0125575, -0.0007954, 0.0008852
6: -0.0008720, 0.0038646, -0.0009274, 0.0041753, -0.0033779, 0.0030353
7: 0.9774491, 0.9807636, 0.9774103, 0.9809809, -0.0023637, 0.0021240
8: -0.0107423, -0.0071888, -0.0107839, -0.0069557, -0.0025342, 0.0022772
9: -0.0002510, 0.0020963, -0.0004050, 0.0021238, -0.0015042, 0.0016740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016091
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016091
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012696, 0.0000013, -0.0012862, 0.0000105, -0.0008211, 0.0008576
1: -0.0075323, -0.0043073, -0.0075745, -0.0042839, -0.0020837, 0.0021763
2: 0.0303569, 0.0323577, 0.0303308, 0.0323723, -0.0012927, 0.0013502
3: -0.0008576, 0.0028784, -0.0008848, 0.0029272, -0.0025211, 0.0024138
4: -0.0065547, -0.0032743, -0.0065975, -0.0032504, -0.0021195, 0.0022136
5: 0.0112554, 0.0124980, 0.0112392, 0.0125070, -0.0008028, 0.0008385
6: -0.0007002, 0.0040412, -0.0007348, 0.0041032, -0.0031996, 0.0030635
7: 0.9775692, 0.9808871, 0.9775451, 0.9809304, -0.0022389, 0.0021437
8: -0.0106135, -0.0070562, -0.0106394, -0.0070098, -0.0024005, 0.0022984
9: -0.0003385, 0.0020112, -0.0003692, 0.0020283, -0.0015182, 0.0015857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016448, upper bound: 0.0016274
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016448, upper bound: 0.0016327
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0012644, 0.0000206, -0.0012862, 0.0000105, -0.0007999, 0.0008709
1: -0.0075191, -0.0042583, -0.0075745, -0.0042839, -0.0020299, 0.0022101
2: 0.0303651, 0.0323882, 0.0303308, 0.0323723, -0.0012594, 0.0013711
3: -0.0009144, 0.0028631, -0.0008848, 0.0029272, -0.0025602, 0.0023516
4: -0.0065412, -0.0032244, -0.0065975, -0.0032504, -0.0020648, 0.0022480
5: 0.0112605, 0.0125169, 0.0112392, 0.0125070, -0.0007821, 0.0008515
6: -0.0007724, 0.0040218, -0.0007348, 0.0041032, -0.0032493, 0.0029845
7: 0.9775188, 0.9808735, 0.9775451, 0.9809304, -0.0022737, 0.0020884
8: -0.0106676, -0.0070708, -0.0106394, -0.0070098, -0.0024378, 0.0022391
9: -0.0003289, 0.0020470, -0.0003692, 0.0020283, -0.0014790, 0.0016103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016448, upper bound: 0.0016275
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016448, upper bound: 0.0016327
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0012696, 0.0000013, -0.0013054, 0.0000501, -0.0008540, 0.0008667
1: -0.0075323, -0.0043073, -0.0076233, -0.0041835, -0.0021672, 0.0021994
2: 0.0303569, 0.0323577, 0.0303005, 0.0324346, -0.0013445, 0.0013645
3: -0.0008576, 0.0028784, -0.0010011, 0.0029838, -0.0025479, 0.0025106
4: -0.0065547, -0.0032743, -0.0066472, -0.0031483, -0.0022044, 0.0022372
5: 0.0112554, 0.0124980, 0.0112204, 0.0125457, -0.0008350, 0.0008474
6: -0.0007002, 0.0040412, -0.0008824, 0.0041749, -0.0032337, 0.0031862
7: 0.9775692, 0.9808871, 0.9774418, 0.9809807, -0.0022628, 0.0022296
8: -0.0106135, -0.0070562, -0.0107501, -0.0069559, -0.0024260, 0.0023905
9: -0.0003385, 0.0020112, -0.0004048, 0.0021015, -0.0015790, 0.0016025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0015847
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0015984
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0012644, 0.0000206, -0.0013054, 0.0000501, -0.0008313, 0.0008794
1: -0.0075191, -0.0042583, -0.0076233, -0.0041835, -0.0021095, 0.0022316
2: 0.0303651, 0.0323882, 0.0303005, 0.0324346, -0.0013088, 0.0013845
3: -0.0009144, 0.0028631, -0.0010011, 0.0029838, -0.0025852, 0.0024438
4: -0.0065412, -0.0032244, -0.0066472, -0.0031483, -0.0021458, 0.0022699
5: 0.0112605, 0.0125169, 0.0112204, 0.0125457, -0.0008128, 0.0008598
6: -0.0007724, 0.0040218, -0.0008824, 0.0041749, -0.0032810, 0.0031015
7: 0.9775188, 0.9808735, 0.9774418, 0.9809807, -0.0022959, 0.0021703
8: -0.0106676, -0.0070708, -0.0107501, -0.0069559, -0.0024615, 0.0023269
9: -0.0003289, 0.0020470, -0.0004048, 0.0021015, -0.0015370, 0.0016260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0015842
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016590, upper bound: 0.0015964
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0012816, 0.0000210, -0.0012894, 0.0000113, -0.0008290, 0.0008818
1: -0.0075628, -0.0042574, -0.0075825, -0.0042820, -0.0021037, 0.0022377
2: 0.0303380, 0.0323887, 0.0303258, 0.0323735, -0.0013051, 0.0013882
3: -0.0009154, 0.0029137, -0.0008870, 0.0029366, -0.0025922, 0.0024370
4: -0.0065857, -0.0032235, -0.0066057, -0.0032485, -0.0021398, 0.0022761
5: 0.0112437, 0.0125172, 0.0112361, 0.0125077, -0.0008105, 0.0008621
6: -0.0007737, 0.0040860, -0.0007375, 0.0041150, -0.0032899, 0.0030929
7: 0.9775178, 0.9809184, 0.9775432, 0.9809388, -0.0023021, 0.0021642
8: -0.0106686, -0.0070226, -0.0106415, -0.0070009, -0.0024682, 0.0023204
9: -0.0003608, 0.0020476, -0.0003751, 0.0020297, -0.0015328, 0.0016304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255

Time for candidate selection: 0.24 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.53 + 596.47 = 600.00 seconds
