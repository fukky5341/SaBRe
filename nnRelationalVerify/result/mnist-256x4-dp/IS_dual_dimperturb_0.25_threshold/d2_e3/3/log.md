## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0001407


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0083696, -0.0071940, -0.0083696, -0.0071940, -0.0007115, 0.0007115)
1: (-0.0052984, -0.0049669, -0.0052984, -0.0049669, -0.0002006, 0.0002006)
2: (-0.0005326, 0.0019129, -0.0005326, 0.0019129, -0.0014800, 0.0014800)
3: (0.0015568, 0.0018804, 0.0015568, 0.0018804, -0.0001959, 0.0001959)
4: (0.0046623, 0.0064899, 0.0046623, 0.0064899, -0.0011061, 0.0011061)
5: (0.9968016, 0.9973094, 0.9968016, 0.9973094, -0.0003073, 0.0003073)
6: (0.0049804, 0.0054413, 0.0049804, 0.0054413, -0.0002789, 0.0002789)
7: (-0.0047954, -0.0030754, -0.0047954, -0.0030754, -0.0010409, 0.0010409)
8: (-0.0067993, -0.0054606, -0.0067993, -0.0054606, -0.0008102, 0.0008102)
9: (-0.0035386, -0.0034231, -0.0035386, -0.0034231, -0.0000699, 0.0000699)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.64 + 1.39 = 3.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0001842, upper bound: 0.0001842

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001794, upper bound: 0.0001787
time: 0.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001794, upper bound: 0.0001794
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 5, lower bound: -0.0001794, upper bound: 0.0001787
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 5, lower bound: -0.0001794, upper bound: 0.0001794

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0083445, -0.0071919, -0.0083617, -0.0071941, -0.0006797, 0.0006889
1: -0.0052913, -0.0049663, -0.0052961, -0.0049670, -0.0001916, 0.0001942
2: -0.0004805, 0.0019172, -0.0005161, 0.0019126, -0.0014139, 0.0014331
3: 0.0015637, 0.0018810, 0.0015590, 0.0018804, -0.0001871, 0.0001896
4: 0.0046590, 0.0064510, 0.0046625, 0.0064776, -0.0010710, 0.0010566
5: 0.9968007, 0.9972985, 0.9968016, 0.9973059, -0.0002976, 0.0002936
6: 0.0049796, 0.0054315, 0.0049805, 0.0054382, -0.0002701, 0.0002665
7: -0.0047985, -0.0031121, -0.0047952, -0.0030870, -0.0010079, 0.0009944
8: -0.0067707, -0.0054582, -0.0067903, -0.0054608, -0.0007739, 0.0007845
9: -0.0035388, -0.0034256, -0.0035386, -0.0034239, -0.0000677, 0.0000668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001688, upper bound: 0.0001623
time: 0.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001709, upper bound: 0.0001703
time: 0.55 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0083504, -0.0071944, -0.0083644, -0.0071941, -0.0006707, 0.0007074
1: -0.0052930, -0.0049670, -0.0052969, -0.0049669, -0.0001891, 0.0001994
2: -0.0004927, 0.0019120, -0.0005219, 0.0019127, -0.0013951, 0.0014714
3: 0.0015621, 0.0018803, 0.0015582, 0.0018804, -0.0001846, 0.0001947
4: 0.0046629, 0.0064601, 0.0046625, 0.0064819, -0.0010997, 0.0010426
5: 0.9968018, 0.9973010, 0.9968016, 0.9973071, -0.0003055, 0.0002897
6: 0.0049806, 0.0054338, 0.0049805, 0.0054393, -0.0002773, 0.0002629
7: -0.0047948, -0.0031035, -0.0047952, -0.0030830, -0.0010349, 0.0009812
8: -0.0067774, -0.0054611, -0.0067934, -0.0054607, -0.0007637, 0.0008055
9: -0.0035386, -0.0034250, -0.0035386, -0.0034236, -0.0000695, 0.0000659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001688, upper bound: 0.0001634
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001709, upper bound: 0.0001709
time: 0.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.66 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001688, upper bound: 0.0001623
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001709, upper bound: 0.0001703
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001688, upper bound: 0.0001634
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 5, lower bound: -0.0001709, upper bound: 0.0001709

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083445, -0.0072161, -0.0084153, -0.0072874, -0.0005550, 0.0006618
1: -0.0052913, -0.0049732, -0.0053112, -0.0049932, -0.0001565, 0.0001866
2: -0.0004805, 0.0018668, -0.0006277, 0.0017186, -0.0011545, 0.0013766
3: 0.0015637, 0.0018743, 0.0015442, 0.0018547, -0.0001528, 0.0001822
4: 0.0046967, 0.0064509, 0.0048075, 0.0065610, -0.0010288, 0.0008628
5: 0.9968111, 0.9972985, 0.9968419, 0.9973291, -0.0002858, 0.0002397
6: 0.0049891, 0.0054315, 0.0050170, 0.0054592, -0.0002594, 0.0002176
7: -0.0047630, -0.0031121, -0.0046588, -0.0030085, -0.0009682, 0.0008120
8: -0.0067707, -0.0054858, -0.0068513, -0.0055669, -0.0006320, 0.0007535
9: -0.0035364, -0.0034256, -0.0035294, -0.0034186, -0.0000650, 0.0000545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001635, upper bound: 0.0001547
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001635, upper bound: 0.0001572
time: 0.61 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083445, -0.0071919, -0.0083617, -0.0072065, -0.0005666, 0.0006889
1: -0.0052913, -0.0049663, -0.0052961, -0.0049704, -0.0001597, 0.0001942
2: -0.0004805, 0.0019172, -0.0005161, 0.0018869, -0.0011786, 0.0014331
3: 0.0015637, 0.0018810, 0.0015590, 0.0018770, -0.0001560, 0.0001896
4: 0.0046590, 0.0064510, 0.0046817, 0.0064776, -0.0010710, 0.0008808
5: 0.9968007, 0.9972985, 0.9968070, 0.9973059, -0.0002976, 0.0002447
6: 0.0049796, 0.0054315, 0.0049853, 0.0054382, -0.0002701, 0.0002221
7: -0.0047985, -0.0031121, -0.0047772, -0.0030870, -0.0010079, 0.0008289
8: -0.0067707, -0.0054582, -0.0067902, -0.0054748, -0.0006452, 0.0007845
9: -0.0035388, -0.0034256, -0.0035374, -0.0034239, -0.0000677, 0.0000557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001644, upper bound: 0.0001653
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001659, upper bound: 0.0001653
time: 0.55 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083504, -0.0072185, -0.0084181, -0.0072873, -0.0005416, 0.0006840
1: -0.0052930, -0.0049738, -0.0053120, -0.0049932, -0.0001527, 0.0001929
2: -0.0004927, 0.0018619, -0.0006336, 0.0017187, -0.0011267, 0.0014230
3: 0.0015621, 0.0018737, 0.0015435, 0.0018547, -0.0001491, 0.0001883
4: 0.0047004, 0.0064601, 0.0048074, 0.0065653, -0.0010634, 0.0008420
5: 0.9968122, 0.9973010, 0.9968419, 0.9973302, -0.0002955, 0.0002339
6: 0.0049900, 0.0054338, 0.0050170, 0.0054604, -0.0002682, 0.0002123
7: -0.0047595, -0.0031035, -0.0046589, -0.0030044, -0.0010008, 0.0007924
8: -0.0067774, -0.0054885, -0.0068545, -0.0055669, -0.0006168, 0.0007789
9: -0.0035362, -0.0034250, -0.0035295, -0.0034184, -0.0000672, 0.0000532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001636, upper bound: 0.0001556
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001635, upper bound: 0.0001579
time: 0.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083504, -0.0071944, -0.0083644, -0.0072064, -0.0005536, 0.0007073
1: -0.0052930, -0.0049670, -0.0052969, -0.0049704, -0.0001561, 0.0001994
2: -0.0004927, 0.0019120, -0.0005219, 0.0018870, -0.0011516, 0.0014714
3: 0.0015621, 0.0018803, 0.0015582, 0.0018770, -0.0001524, 0.0001947
4: 0.0046629, 0.0064601, 0.0046816, 0.0064819, -0.0010996, 0.0008606
5: 0.9968018, 0.9973010, 0.9968069, 0.9973071, -0.0003055, 0.0002391
6: 0.0049806, 0.0054338, 0.0049853, 0.0054393, -0.0002773, 0.0002170
7: -0.0047948, -0.0031035, -0.0047772, -0.0030830, -0.0010349, 0.0008099
8: -0.0067774, -0.0054611, -0.0067934, -0.0054748, -0.0006304, 0.0008055
9: -0.0035386, -0.0034250, -0.0035374, -0.0034236, -0.0000695, 0.0000544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001659, upper bound: 0.0001644
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001659, upper bound: 0.0001659
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.55 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 5, lower bound: -0.0001635, upper bound: 0.0001547
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 5, lower bound: -0.0001635, upper bound: 0.0001572
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 5, lower bound: -0.0001644, upper bound: 0.0001653
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 5, lower bound: -0.0001659, upper bound: 0.0001653
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 5, lower bound: -0.0001636, upper bound: 0.0001556
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 5, lower bound: -0.0001635, upper bound: 0.0001579
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 5, lower bound: -0.0001659, upper bound: 0.0001644
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 5, lower bound: -0.0001659, upper bound: 0.0001659

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083221, -0.0072173, -0.0084098, -0.0072876, -0.0005314, 0.0006530
1: -0.0052850, -0.0049735, -0.0053097, -0.0049933, -0.0001498, 0.0001841
2: -0.0004338, 0.0018645, -0.0006163, 0.0017181, -0.0011054, 0.0013584
3: 0.0015699, 0.0018740, 0.0015457, 0.0018547, -0.0001463, 0.0001798
4: 0.0046985, 0.0064160, 0.0048079, 0.0065525, -0.0010152, 0.0008261
5: 0.9968117, 0.9972887, 0.9968420, 0.9973267, -0.0002821, 0.0002295
6: 0.0049896, 0.0054227, 0.0050171, 0.0054571, -0.0002560, 0.0002083
7: -0.0047613, -0.0031449, -0.0046584, -0.0030165, -0.0009554, 0.0007774
8: -0.0067452, -0.0054871, -0.0068451, -0.0055672, -0.0006051, 0.0007436
9: -0.0035363, -0.0034278, -0.0035294, -0.0034192, -0.0000642, 0.0000522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001558, upper bound: 0.0001480
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001503
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083238, -0.0072076, -0.0084094, -0.0072878, -0.0005334, 0.0006697
1: -0.0052855, -0.0049707, -0.0053096, -0.0049934, -0.0001504, 0.0001888
2: -0.0004374, 0.0018847, -0.0006154, 0.0017177, -0.0011095, 0.0013932
3: 0.0015694, 0.0018767, 0.0015459, 0.0018546, -0.0001468, 0.0001844
4: 0.0046834, 0.0064188, 0.0048082, 0.0065518, -0.0010412, 0.0008292
5: 0.9968074, 0.9972896, 0.9968421, 0.9973265, -0.0002893, 0.0002304
6: 0.0049858, 0.0054234, 0.0050172, 0.0054569, -0.0002626, 0.0002091
7: -0.0047755, -0.0031423, -0.0046581, -0.0030172, -0.0009798, 0.0007804
8: -0.0067472, -0.0054761, -0.0068446, -0.0055675, -0.0006073, 0.0007626
9: -0.0035373, -0.0034276, -0.0035294, -0.0034192, -0.0000658, 0.0000524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001558, upper bound: 0.0001498
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001527
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0083393, -0.0071922, -0.0083399, -0.0072076, -0.0005577, 0.0006651
1: -0.0052898, -0.0049664, -0.0052900, -0.0049707, -0.0001572, 0.0001875
2: -0.0004697, 0.0019167, -0.0004708, 0.0018847, -0.0011602, 0.0013836
3: 0.0015651, 0.0018809, 0.0015650, 0.0018767, -0.0001535, 0.0001831
4: 0.0046594, 0.0064429, 0.0046834, 0.0064437, -0.0010340, 0.0008670
5: 0.9968008, 0.9972963, 0.9968075, 0.9972965, -0.0002873, 0.0002409
6: 0.0049797, 0.0054295, 0.0049858, 0.0054297, -0.0002608, 0.0002187
7: -0.0047981, -0.0031197, -0.0047756, -0.0031189, -0.0009731, 0.0008160
8: -0.0067648, -0.0054585, -0.0067654, -0.0054760, -0.0006351, 0.0007574
9: -0.0035388, -0.0034261, -0.0035373, -0.0034260, -0.0000653, 0.0000548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001629
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001608
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083385, -0.0071924, -0.0083410, -0.0071988, -0.0005778, 0.0006672
1: -0.0052896, -0.0049665, -0.0052903, -0.0049683, -0.0001629, 0.0001881
2: -0.0004680, 0.0019163, -0.0004732, 0.0019028, -0.0012019, 0.0013878
3: 0.0015654, 0.0018809, 0.0015647, 0.0018791, -0.0001591, 0.0001837
4: 0.0046598, 0.0064416, 0.0046698, 0.0064455, -0.0010372, 0.0008982
5: 0.9968009, 0.9972959, 0.9968036, 0.9972970, -0.0002882, 0.0002496
6: 0.0049798, 0.0054292, 0.0049823, 0.0054301, -0.0002616, 0.0002265
7: -0.0047978, -0.0031209, -0.0047883, -0.0031172, -0.0009761, 0.0008453
8: -0.0067639, -0.0054587, -0.0067667, -0.0054661, -0.0006579, 0.0007597
9: -0.0035388, -0.0034262, -0.0035381, -0.0034259, -0.0000655, 0.0000568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001579, upper bound: 0.0001629
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001579, upper bound: 0.0001607
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083281, -0.0072196, -0.0084128, -0.0072876, -0.0005179, 0.0006754
1: -0.0052867, -0.0049741, -0.0053105, -0.0049933, -0.0001460, 0.0001904
2: -0.0004464, 0.0018596, -0.0006224, 0.0017182, -0.0010774, 0.0014050
3: 0.0015682, 0.0018734, 0.0015449, 0.0018547, -0.0001426, 0.0001859
4: 0.0047021, 0.0064255, 0.0048078, 0.0065570, -0.0010500, 0.0008052
5: 0.9968126, 0.9972914, 0.9968420, 0.9973280, -0.0002917, 0.0002237
6: 0.0049905, 0.0054251, 0.0050171, 0.0054583, -0.0002648, 0.0002031
7: -0.0047580, -0.0031361, -0.0046585, -0.0030122, -0.0009882, 0.0007578
8: -0.0067521, -0.0054897, -0.0068484, -0.0055672, -0.0005898, 0.0007691
9: -0.0035361, -0.0034272, -0.0035294, -0.0034189, -0.0000664, 0.0000509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001558, upper bound: 0.0001496
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001512
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083291, -0.0072100, -0.0084120, -0.0072878, -0.0005200, 0.0006904
1: -0.0052870, -0.0049714, -0.0053103, -0.0049934, -0.0001466, 0.0001947
2: -0.0004485, 0.0018795, -0.0006209, 0.0017177, -0.0010816, 0.0014362
3: 0.0015679, 0.0018760, 0.0015451, 0.0018546, -0.0001431, 0.0001901
4: 0.0046872, 0.0064270, 0.0048081, 0.0065559, -0.0010734, 0.0008083
5: 0.9968085, 0.9972919, 0.9968421, 0.9973276, -0.0002982, 0.0002246
6: 0.0049867, 0.0054255, 0.0050172, 0.0054580, -0.0002707, 0.0002038
7: -0.0047719, -0.0031346, -0.0046581, -0.0030133, -0.0010101, 0.0007607
8: -0.0067532, -0.0054789, -0.0068476, -0.0055674, -0.0005921, 0.0007862
9: -0.0035370, -0.0034271, -0.0035294, -0.0034190, -0.0000678, 0.0000511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001558, upper bound: 0.0001513
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001537
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083282, -0.0071955, -0.0083593, -0.0072067, -0.0005259, 0.0006998
1: -0.0052867, -0.0049673, -0.0052954, -0.0049705, -0.0001483, 0.0001973
2: -0.0004464, 0.0019098, -0.0005111, 0.0018865, -0.0010940, 0.0014558
3: 0.0015682, 0.0018800, 0.0015597, 0.0018769, -0.0001448, 0.0001926
4: 0.0046646, 0.0064255, 0.0046820, 0.0064738, -0.0010879, 0.0008176
5: 0.9968022, 0.9972914, 0.9968070, 0.9973049, -0.0003023, 0.0002271
6: 0.0049810, 0.0054251, 0.0049854, 0.0054373, -0.0002744, 0.0002062
7: -0.0047932, -0.0031360, -0.0047768, -0.0030905, -0.0010239, 0.0007694
8: -0.0067521, -0.0054623, -0.0067875, -0.0054750, -0.0005988, 0.0007969
9: -0.0035385, -0.0034272, -0.0035374, -0.0034241, -0.0000688, 0.0000517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001580, upper bound: 0.0001622
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001580, upper bound: 0.0001589
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083292, -0.0071865, -0.0083584, -0.0072069, -0.0005259, 0.0007110
1: -0.0052870, -0.0049648, -0.0052952, -0.0049706, -0.0001483, 0.0002005
2: -0.0004485, 0.0019285, -0.0005093, 0.0018860, -0.0010939, 0.0014790
3: 0.0015679, 0.0018825, 0.0015599, 0.0018769, -0.0001448, 0.0001957
4: 0.0046507, 0.0064270, 0.0046824, 0.0064725, -0.0011053, 0.0008175
5: 0.9967983, 0.9972919, 0.9968072, 0.9973044, -0.0003071, 0.0002271
6: 0.0049775, 0.0054255, 0.0049855, 0.0054369, -0.0002787, 0.0002062
7: -0.0048064, -0.0031346, -0.0047765, -0.0030918, -0.0010402, 0.0007694
8: -0.0067532, -0.0054521, -0.0067865, -0.0054753, -0.0005988, 0.0008096
9: -0.0035394, -0.0034271, -0.0035374, -0.0034242, -0.0000698, 0.0000517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001578, upper bound: 0.0001636
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001578, upper bound: 0.0001614
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.65 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001558, upper bound: 0.0001480
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001503
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001558, upper bound: 0.0001498
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001527
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001629
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001608
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001579, upper bound: 0.0001629
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001579, upper bound: 0.0001607
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001558, upper bound: 0.0001496
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001512
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001558, upper bound: 0.0001513
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001593, upper bound: 0.0001537
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001580, upper bound: 0.0001622
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001580, upper bound: 0.0001589
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001578, upper bound: 0.0001636
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 5, lower bound: -0.0001578, upper bound: 0.0001614

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083121, -0.0072180, -0.0083780, -0.0072805, -0.0005093, 0.0006093
1: -0.0052822, -0.0049737, -0.0053007, -0.0049913, -0.0001436, 0.0001718
2: -0.0004131, 0.0018630, -0.0005501, 0.0017329, -0.0010594, 0.0012675
3: 0.0015726, 0.0018738, 0.0015545, 0.0018566, -0.0001402, 0.0001677
4: 0.0046996, 0.0064006, 0.0047968, 0.0065030, -0.0009473, 0.0007917
5: 0.9968120, 0.9972845, 0.9968389, 0.9973130, -0.0002632, 0.0002200
6: 0.0049898, 0.0054188, 0.0050144, 0.0054446, -0.0002389, 0.0001997
7: -0.0047603, -0.0031595, -0.0046688, -0.0030631, -0.0008915, 0.0007451
8: -0.0067338, -0.0054879, -0.0068088, -0.0055591, -0.0005799, 0.0006939
9: -0.0035363, -0.0034288, -0.0035301, -0.0034223, -0.0000599, 0.0000500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001480
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001480
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083211, -0.0072173, -0.0083984, -0.0072885, -0.0005301, 0.0006204
1: -0.0052847, -0.0049735, -0.0053065, -0.0049936, -0.0001495, 0.0001749
2: -0.0004317, 0.0018643, -0.0005926, 0.0017163, -0.0011028, 0.0012906
3: 0.0015702, 0.0018740, 0.0015489, 0.0018544, -0.0001459, 0.0001708
4: 0.0046986, 0.0064145, 0.0048092, 0.0065347, -0.0009645, 0.0008241
5: 0.9968116, 0.9972883, 0.9968424, 0.9973217, -0.0002680, 0.0002290
6: 0.0049896, 0.0054223, 0.0050175, 0.0054526, -0.0002432, 0.0002078
7: -0.0047612, -0.0031464, -0.0046572, -0.0030332, -0.0009077, 0.0007756
8: -0.0067440, -0.0054872, -0.0068321, -0.0055682, -0.0006037, 0.0007065
9: -0.0035363, -0.0034279, -0.0035293, -0.0034203, -0.0000610, 0.0000521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001502
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001502
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083138, -0.0072082, -0.0083773, -0.0072807, -0.0005113, 0.0006271
1: -0.0052826, -0.0049709, -0.0053005, -0.0049914, -0.0001441, 0.0001768
2: -0.0004165, 0.0018832, -0.0005487, 0.0017324, -0.0010635, 0.0013045
3: 0.0015722, 0.0018765, 0.0015547, 0.0018566, -0.0001407, 0.0001726
4: 0.0046845, 0.0064031, 0.0047972, 0.0065020, -0.0009749, 0.0007948
5: 0.9968078, 0.9972852, 0.9968390, 0.9973126, -0.0002709, 0.0002208
6: 0.0049860, 0.0054194, 0.0050144, 0.0054444, -0.0002459, 0.0002004
7: -0.0047745, -0.0031571, -0.0046685, -0.0030641, -0.0009175, 0.0007480
8: -0.0067357, -0.0054768, -0.0068081, -0.0055594, -0.0005822, 0.0007141
9: -0.0035372, -0.0034286, -0.0035301, -0.0034224, -0.0000616, 0.0000502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001498
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001498
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083229, -0.0072076, -0.0083977, -0.0072887, -0.0005321, 0.0006397
1: -0.0052852, -0.0049708, -0.0053063, -0.0049936, -0.0001500, 0.0001804
2: -0.0004354, 0.0018845, -0.0005912, 0.0017159, -0.0011069, 0.0013307
3: 0.0015697, 0.0018767, 0.0015491, 0.0018544, -0.0001465, 0.0001761
4: 0.0046835, 0.0064172, 0.0048095, 0.0065337, -0.0009945, 0.0008272
5: 0.9968075, 0.9972891, 0.9968424, 0.9973215, -0.0002763, 0.0002298
6: 0.0049858, 0.0054230, 0.0050176, 0.0054524, -0.0002508, 0.0002086
7: -0.0047755, -0.0031438, -0.0046569, -0.0030342, -0.0009359, 0.0007785
8: -0.0067460, -0.0054761, -0.0068313, -0.0055684, -0.0006059, 0.0007284
9: -0.0035373, -0.0034277, -0.0035293, -0.0034204, -0.0000628, 0.0000523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001527
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001527
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083927, -0.0072869, -0.0083399, -0.0072076, -0.0006740, 0.0005354
1: -0.0053049, -0.0049931, -0.0052900, -0.0049707, -0.0001900, 0.0001510
2: -0.0005806, 0.0017196, -0.0004708, 0.0018847, -0.0014021, 0.0011138
3: 0.0015505, 0.0018549, 0.0015650, 0.0018767, -0.0001855, 0.0001474
4: 0.0048068, 0.0065258, 0.0046834, 0.0064437, -0.0008324, 0.0010478
5: 0.9968417, 0.9973193, 0.9968075, 0.9972965, -0.0002313, 0.0002911
6: 0.0050169, 0.0054504, 0.0049858, 0.0054297, -0.0002099, 0.0002642
7: -0.0046594, -0.0030416, -0.0047756, -0.0031189, -0.0007834, 0.0009861
8: -0.0068255, -0.0055664, -0.0067654, -0.0054760, -0.0007675, 0.0006097
9: -0.0035295, -0.0034209, -0.0035373, -0.0034260, -0.0000526, 0.0000662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001493, upper bound: 0.0001557
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001512, upper bound: 0.0001586
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083393, -0.0072045, -0.0083399, -0.0072076, -0.0005577, 0.0005449
1: -0.0052898, -0.0049699, -0.0052900, -0.0049707, -0.0001572, 0.0001536
2: -0.0004697, 0.0018911, -0.0004708, 0.0018847, -0.0011601, 0.0011335
3: 0.0015651, 0.0018775, 0.0015650, 0.0018767, -0.0001535, 0.0001500
4: 0.0046786, 0.0064429, 0.0046834, 0.0064437, -0.0008471, 0.0008670
5: 0.9968061, 0.9972963, 0.9968075, 0.9972965, -0.0002354, 0.0002409
6: 0.0049845, 0.0054295, 0.0049858, 0.0054297, -0.0002136, 0.0002186
7: -0.0047801, -0.0031197, -0.0047756, -0.0031189, -0.0007972, 0.0008159
8: -0.0067648, -0.0054725, -0.0067654, -0.0054760, -0.0006351, 0.0006205
9: -0.0035376, -0.0034261, -0.0035373, -0.0034260, -0.0000535, 0.0000548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001493, upper bound: 0.0001547
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001512, upper bound: 0.0001563
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083922, -0.0072871, -0.0083410, -0.0071988, -0.0006876, 0.0005374
1: -0.0053047, -0.0049932, -0.0052903, -0.0049683, -0.0001939, 0.0001515
2: -0.0005795, 0.0017192, -0.0004732, 0.0019028, -0.0014304, 0.0011179
3: 0.0015506, 0.0018548, 0.0015647, 0.0018791, -0.0001893, 0.0001479
4: 0.0048071, 0.0065250, 0.0046698, 0.0064455, -0.0008355, 0.0010690
5: 0.9968418, 0.9973192, 0.9968036, 0.9972970, -0.0002321, 0.0002970
6: 0.0050169, 0.0054502, 0.0049823, 0.0054301, -0.0002107, 0.0002696
7: -0.0046592, -0.0030424, -0.0047883, -0.0031172, -0.0007863, 0.0010060
8: -0.0068250, -0.0055666, -0.0067667, -0.0054661, -0.0007830, 0.0006119
9: -0.0035295, -0.0034209, -0.0035381, -0.0034259, -0.0000528, 0.0000676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001512, upper bound: 0.0001557
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001537, upper bound: 0.0001586
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083385, -0.0072047, -0.0083410, -0.0071988, -0.0005778, 0.0005455
1: -0.0052896, -0.0049699, -0.0052903, -0.0049683, -0.0001629, 0.0001538
2: -0.0004680, 0.0018906, -0.0004732, 0.0019028, -0.0012019, 0.0011348
3: 0.0015654, 0.0018775, 0.0015647, 0.0018791, -0.0001590, 0.0001502
4: 0.0046789, 0.0064416, 0.0046698, 0.0064455, -0.0008480, 0.0008982
5: 0.9968062, 0.9972959, 0.9968036, 0.9972970, -0.0002356, 0.0002495
6: 0.0049846, 0.0054291, 0.0049823, 0.0054301, -0.0002139, 0.0002265
7: -0.0047798, -0.0031209, -0.0047883, -0.0031172, -0.0007981, 0.0008453
8: -0.0067639, -0.0054728, -0.0067667, -0.0054661, -0.0006579, 0.0006212
9: -0.0035376, -0.0034262, -0.0035381, -0.0034259, -0.0000536, 0.0000568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001512, upper bound: 0.0001547
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001537, upper bound: 0.0001563
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083181, -0.0072203, -0.0083805, -0.0072805, -0.0004965, 0.0006317
1: -0.0052838, -0.0049743, -0.0053014, -0.0049913, -0.0001400, 0.0001781
2: -0.0004255, 0.0018581, -0.0005552, 0.0017330, -0.0010329, 0.0013141
3: 0.0015710, 0.0018732, 0.0015538, 0.0018566, -0.0001367, 0.0001739
4: 0.0047032, 0.0064098, 0.0047967, 0.0065068, -0.0009821, 0.0007719
5: 0.9968129, 0.9972871, 0.9968389, 0.9973140, -0.0002729, 0.0002145
6: 0.0049908, 0.0054211, 0.0050143, 0.0054456, -0.0002477, 0.0001947
7: -0.0047569, -0.0031508, -0.0046689, -0.0030595, -0.0009243, 0.0007265
8: -0.0067406, -0.0054906, -0.0068116, -0.0055591, -0.0005654, 0.0007193
9: -0.0035360, -0.0034282, -0.0035301, -0.0034221, -0.0000621, 0.0000488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001496
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001496
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083271, -0.0072197, -0.0084014, -0.0072884, -0.0005168, 0.0006424
1: -0.0052864, -0.0049741, -0.0053073, -0.0049935, -0.0001457, 0.0001811
2: -0.0004442, 0.0018595, -0.0005987, 0.0017165, -0.0010751, 0.0013363
3: 0.0015685, 0.0018734, 0.0015481, 0.0018544, -0.0001423, 0.0001768
4: 0.0047022, 0.0064239, 0.0048091, 0.0065393, -0.0009987, 0.0008034
5: 0.9968127, 0.9972910, 0.9968424, 0.9973230, -0.0002775, 0.0002232
6: 0.0049905, 0.0054247, 0.0050175, 0.0054538, -0.0002518, 0.0002026
7: -0.0047578, -0.0031376, -0.0046572, -0.0030289, -0.0009399, 0.0007561
8: -0.0067509, -0.0054898, -0.0068355, -0.0055681, -0.0005885, 0.0007315
9: -0.0035361, -0.0034273, -0.0035293, -0.0034200, -0.0000631, 0.0000508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001512
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001512
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083191, -0.0072107, -0.0083796, -0.0072807, -0.0004985, 0.0006472
1: -0.0052841, -0.0049716, -0.0053012, -0.0049914, -0.0001406, 0.0001825
2: -0.0004275, 0.0018781, -0.0005533, 0.0017325, -0.0010370, 0.0013463
3: 0.0015707, 0.0018758, 0.0015541, 0.0018566, -0.0001372, 0.0001782
4: 0.0046883, 0.0064114, 0.0047971, 0.0065054, -0.0010061, 0.0007750
5: 0.9968088, 0.9972876, 0.9968390, 0.9973136, -0.0002795, 0.0002153
6: 0.0049870, 0.0054215, 0.0050144, 0.0054452, -0.0002537, 0.0001954
7: -0.0047709, -0.0031493, -0.0046686, -0.0030608, -0.0009469, 0.0007294
8: -0.0067417, -0.0054797, -0.0068106, -0.0055593, -0.0005677, 0.0007370
9: -0.0035370, -0.0034281, -0.0035301, -0.0034221, -0.0000636, 0.0000490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001513
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001513
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083281, -0.0072101, -0.0084006, -0.0072886, -0.0005188, 0.0006598
1: -0.0052867, -0.0049715, -0.0053071, -0.0049936, -0.0001463, 0.0001860
2: -0.0004464, 0.0018794, -0.0005971, 0.0017160, -0.0010793, 0.0013726
3: 0.0015682, 0.0018760, 0.0015483, 0.0018544, -0.0001428, 0.0001816
4: 0.0046873, 0.0064254, 0.0048094, 0.0065381, -0.0010258, 0.0008066
5: 0.9968085, 0.9972914, 0.9968424, 0.9973227, -0.0002850, 0.0002241
6: 0.0049867, 0.0054251, 0.0050175, 0.0054535, -0.0002587, 0.0002034
7: -0.0047718, -0.0031361, -0.0046569, -0.0030301, -0.0009654, 0.0007591
8: -0.0067521, -0.0054789, -0.0068346, -0.0055684, -0.0005908, 0.0007513
9: -0.0035370, -0.0034272, -0.0035293, -0.0034201, -0.0000648, 0.0000510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001537
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001537
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083804, -0.0072888, -0.0083593, -0.0072067, -0.0006471, 0.0005749
1: -0.0053014, -0.0049936, -0.0052954, -0.0049705, -0.0001824, 0.0001621
2: -0.0005552, 0.0017157, -0.0005111, 0.0018865, -0.0013460, 0.0011959
3: 0.0015538, 0.0018543, 0.0015597, 0.0018769, -0.0001781, 0.0001583
4: 0.0048096, 0.0065068, 0.0046820, 0.0064738, -0.0008938, 0.0010059
5: 0.9968426, 0.9973140, 0.9968070, 0.9973049, -0.0002483, 0.0002795
6: 0.0050176, 0.0054456, 0.0049854, 0.0054373, -0.0002254, 0.0002537
7: -0.0046567, -0.0030596, -0.0047768, -0.0030905, -0.0008411, 0.0009467
8: -0.0068116, -0.0055685, -0.0067875, -0.0054750, -0.0007368, 0.0006546
9: -0.0035293, -0.0034221, -0.0035374, -0.0034241, -0.0000565, 0.0000636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001513, upper bound: 0.0001535
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001579
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083281, -0.0072078, -0.0083593, -0.0072067, -0.0005259, 0.0005885
1: -0.0052867, -0.0049708, -0.0052954, -0.0049705, -0.0001483, 0.0001659
2: -0.0004464, 0.0018841, -0.0005111, 0.0018865, -0.0010939, 0.0012242
3: 0.0015682, 0.0018766, 0.0015597, 0.0018769, -0.0001448, 0.0001620
4: 0.0046838, 0.0064255, 0.0046820, 0.0064738, -0.0009149, 0.0008175
5: 0.9968075, 0.9972914, 0.9968070, 0.9973049, -0.0002542, 0.0002271
6: 0.0049859, 0.0054251, 0.0049854, 0.0054373, -0.0002307, 0.0002062
7: -0.0047752, -0.0031361, -0.0047768, -0.0030905, -0.0008610, 0.0007694
8: -0.0067521, -0.0054763, -0.0067875, -0.0054750, -0.0005988, 0.0006701
9: -0.0035373, -0.0034272, -0.0035374, -0.0034241, -0.0000578, 0.0000517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001547
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001546
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0083828, -0.0072788, -0.0083584, -0.0072069, -0.0006491, 0.0005913
1: -0.0053021, -0.0049908, -0.0052952, -0.0049706, -0.0001830, 0.0001667
2: -0.0005601, 0.0017365, -0.0005093, 0.0018860, -0.0013503, 0.0012300
3: 0.0015532, 0.0018571, 0.0015599, 0.0018769, -0.0001787, 0.0001628
4: 0.0047941, 0.0065105, 0.0046824, 0.0064725, -0.0009192, 0.0010091
5: 0.9968382, 0.9973150, 0.9968072, 0.9973044, -0.0002554, 0.0002804
6: 0.0050137, 0.0054465, 0.0049855, 0.0054369, -0.0002318, 0.0002545
7: -0.0046713, -0.0030561, -0.0047765, -0.0030918, -0.0008651, 0.0009497
8: -0.0068143, -0.0055572, -0.0067865, -0.0054753, -0.0007391, 0.0006733
9: -0.0035303, -0.0034218, -0.0035374, -0.0034242, -0.0000581, 0.0000638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001513, upper bound: 0.0001558
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001537, upper bound: 0.0001593
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0083291, -0.0071991, -0.0083584, -0.0072069, -0.0005259, 0.0006084
1: -0.0052870, -0.0049683, -0.0052952, -0.0049706, -0.0001483, 0.0001715
2: -0.0004485, 0.0019023, -0.0005093, 0.0018860, -0.0010939, 0.0012656
3: 0.0015679, 0.0018790, 0.0015599, 0.0018769, -0.0001448, 0.0001675
4: 0.0046702, 0.0064270, 0.0046824, 0.0064725, -0.0009458, 0.0008175
5: 0.9968037, 0.9972919, 0.9968072, 0.9973044, -0.0002628, 0.0002271
6: 0.0049824, 0.0054255, 0.0049855, 0.0054369, -0.0002385, 0.0002062
7: -0.0047880, -0.0031346, -0.0047765, -0.0030918, -0.0008901, 0.0007694
8: -0.0067532, -0.0054664, -0.0067865, -0.0054753, -0.0005988, 0.0006928
9: -0.0035381, -0.0034271, -0.0035374, -0.0034242, -0.0000598, 0.0000517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001573
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001537, upper bound: 0.0001572
time: 0.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.80 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001480
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001480
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001502
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001502
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001498
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001498
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001527
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001527
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001493, upper bound: 0.0001557
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001512, upper bound: 0.0001586
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001493, upper bound: 0.0001547
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001512, upper bound: 0.0001563
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001512, upper bound: 0.0001557
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001537, upper bound: 0.0001586
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001512, upper bound: 0.0001547
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001537, upper bound: 0.0001563
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001496
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001496
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001512
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001512
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001513
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001513
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001537
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001537
IS_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001513, upper bound: 0.0001535
IS_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001579
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001547
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001538, upper bound: 0.0001546
IS_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001513, upper bound: 0.0001558
IS_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001537, upper bound: 0.0001593
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001573
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 5, lower bound: -0.0001537, upper bound: 0.0001572

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083648, -0.0072885, -0.0083780, -0.0072805, -0.0005108, 0.0005154
1: -0.0052970, -0.0049936, -0.0053007, -0.0049913, -0.0001440, 0.0001453
2: -0.0005226, 0.0017162, -0.0005501, 0.0017329, -0.0010626, 0.0010721
3: 0.0015581, 0.0018544, 0.0015545, 0.0018566, -0.0001406, 0.0001419
4: 0.0048093, 0.0064824, 0.0047968, 0.0065030, -0.0008012, 0.0007941
5: 0.9968424, 0.9973072, 0.9968389, 0.9973130, -0.0002226, 0.0002206
6: 0.0050175, 0.0054394, 0.0050144, 0.0054446, -0.0002021, 0.0002003
7: -0.0046571, -0.0030825, -0.0046688, -0.0030631, -0.0007540, 0.0007474
8: -0.0067938, -0.0055683, -0.0068088, -0.0055591, -0.0005817, 0.0005869
9: -0.0035293, -0.0034236, -0.0035301, -0.0034223, -0.0000506, 0.0000502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001157, upper bound: 0.0001345
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001449
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083121, -0.0072061, -0.0083780, -0.0072805, -0.0005093, 0.0006417
1: -0.0052822, -0.0049703, -0.0053007, -0.0049913, -0.0001436, 0.0001809
2: -0.0004131, 0.0018878, -0.0005501, 0.0017329, -0.0010594, 0.0013350
3: 0.0015726, 0.0018771, 0.0015545, 0.0018566, -0.0001402, 0.0001767
4: 0.0046811, 0.0064006, 0.0047968, 0.0065030, -0.0009977, 0.0007917
5: 0.9968068, 0.9972845, 0.9968389, 0.9973130, -0.0002772, 0.0002200
6: 0.0049852, 0.0054188, 0.0050144, 0.0054446, -0.0002516, 0.0001997
7: -0.0047777, -0.0031595, -0.0046688, -0.0030631, -0.0009389, 0.0007451
8: -0.0067338, -0.0054744, -0.0068088, -0.0055591, -0.0005799, 0.0007308
9: -0.0035374, -0.0034288, -0.0035301, -0.0034223, -0.0000630, 0.0000500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001157, upper bound: 0.0001346
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001449
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083736, -0.0072878, -0.0083984, -0.0072885, -0.0005336, 0.0005240
1: -0.0052995, -0.0049934, -0.0053065, -0.0049936, -0.0001504, 0.0001477
2: -0.0005410, 0.0017177, -0.0005926, 0.0017163, -0.0011100, 0.0010899
3: 0.0015557, 0.0018546, 0.0015489, 0.0018544, -0.0001469, 0.0001442
4: 0.0048082, 0.0064961, 0.0048092, 0.0065347, -0.0008145, 0.0008295
5: 0.9968421, 0.9973111, 0.9968424, 0.9973217, -0.0002263, 0.0002305
6: 0.0050172, 0.0054429, 0.0050175, 0.0054526, -0.0002054, 0.0002092
7: -0.0046581, -0.0030695, -0.0046572, -0.0030332, -0.0007666, 0.0007807
8: -0.0068038, -0.0055675, -0.0068321, -0.0055682, -0.0006076, 0.0005966
9: -0.0035294, -0.0034227, -0.0035293, -0.0034203, -0.0000515, 0.0000524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001436
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001503
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083211, -0.0072054, -0.0083984, -0.0072885, -0.0005301, 0.0006545
1: -0.0052847, -0.0049701, -0.0053065, -0.0049936, -0.0001495, 0.0001845
2: -0.0004317, 0.0018891, -0.0005926, 0.0017163, -0.0011028, 0.0013614
3: 0.0015702, 0.0018773, 0.0015489, 0.0018544, -0.0001459, 0.0001802
4: 0.0046800, 0.0064145, 0.0048092, 0.0065347, -0.0010174, 0.0008241
5: 0.9968064, 0.9972883, 0.9968424, 0.9973217, -0.0002827, 0.0002290
6: 0.0049849, 0.0054223, 0.0050175, 0.0054526, -0.0002566, 0.0002078
7: -0.0047787, -0.0031464, -0.0046572, -0.0030332, -0.0009575, 0.0007756
8: -0.0067440, -0.0054736, -0.0068321, -0.0055682, -0.0006037, 0.0007452
9: -0.0035375, -0.0034279, -0.0035293, -0.0034203, -0.0000643, 0.0000521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001435
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001502
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083681, -0.0072785, -0.0083773, -0.0072807, -0.0005127, 0.0005405
1: -0.0052979, -0.0049907, -0.0053005, -0.0049914, -0.0001446, 0.0001524
2: -0.0005295, 0.0017370, -0.0005487, 0.0017324, -0.0010666, 0.0011243
3: 0.0015572, 0.0018572, 0.0015547, 0.0018566, -0.0001411, 0.0001488
4: 0.0047937, 0.0064876, 0.0047972, 0.0065020, -0.0008402, 0.0007971
5: 0.9968380, 0.9973086, 0.9968390, 0.9973126, -0.0002334, 0.0002215
6: 0.0050136, 0.0054407, 0.0050144, 0.0054444, -0.0002119, 0.0002010
7: -0.0046717, -0.0030776, -0.0046685, -0.0030641, -0.0007908, 0.0007502
8: -0.0067976, -0.0055569, -0.0068081, -0.0055594, -0.0005839, 0.0006155
9: -0.0035303, -0.0034233, -0.0035301, -0.0034224, -0.0000531, 0.0000504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001145, upper bound: 0.0001360
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001471
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083138, -0.0071963, -0.0083773, -0.0072807, -0.0005113, 0.0006562
1: -0.0052826, -0.0049676, -0.0053005, -0.0049914, -0.0001441, 0.0001850
2: -0.0004165, 0.0019081, -0.0005487, 0.0017324, -0.0010635, 0.0013650
3: 0.0015722, 0.0018798, 0.0015547, 0.0018566, -0.0001407, 0.0001806
4: 0.0046658, 0.0064031, 0.0047972, 0.0065020, -0.0010201, 0.0007948
5: 0.9968026, 0.9972852, 0.9968390, 0.9973126, -0.0002834, 0.0002208
6: 0.0049813, 0.0054194, 0.0050144, 0.0054444, -0.0002573, 0.0002004
7: -0.0047921, -0.0031571, -0.0046685, -0.0030641, -0.0009600, 0.0007480
8: -0.0067357, -0.0054632, -0.0068081, -0.0055594, -0.0005822, 0.0007472
9: -0.0035384, -0.0034286, -0.0035301, -0.0034224, -0.0000645, 0.0000502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001145, upper bound: 0.0001362
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001471
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083766, -0.0072778, -0.0083977, -0.0072887, -0.0005355, 0.0005514
1: -0.0053003, -0.0049905, -0.0053063, -0.0049936, -0.0001510, 0.0001554
2: -0.0005472, 0.0017385, -0.0005912, 0.0017159, -0.0011140, 0.0011469
3: 0.0015549, 0.0018574, 0.0015491, 0.0018544, -0.0001474, 0.0001518
4: 0.0047926, 0.0065008, 0.0048095, 0.0065337, -0.0008572, 0.0008325
5: 0.9968378, 0.9973123, 0.9968424, 0.9973215, -0.0002381, 0.0002313
6: 0.0050133, 0.0054441, 0.0050176, 0.0054524, -0.0002162, 0.0002100
7: -0.0046728, -0.0030652, -0.0046569, -0.0030342, -0.0008067, 0.0007835
8: -0.0068072, -0.0055561, -0.0068313, -0.0055684, -0.0006098, 0.0006278
9: -0.0035304, -0.0034224, -0.0035293, -0.0034204, -0.0000542, 0.0000526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001461
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001527
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083229, -0.0071956, -0.0083977, -0.0072887, -0.0005321, 0.0006702
1: -0.0052852, -0.0049674, -0.0053063, -0.0049936, -0.0001500, 0.0001890
2: -0.0004354, 0.0019094, -0.0005912, 0.0017159, -0.0011069, 0.0013942
3: 0.0015697, 0.0018800, 0.0015491, 0.0018544, -0.0001465, 0.0001845
4: 0.0046649, 0.0064172, 0.0048095, 0.0065337, -0.0010419, 0.0008272
5: 0.9968023, 0.9972891, 0.9968424, 0.9973215, -0.0002895, 0.0002298
6: 0.0049811, 0.0054230, 0.0050176, 0.0054524, -0.0002628, 0.0002086
7: -0.0047930, -0.0031438, -0.0046569, -0.0030342, -0.0009806, 0.0007785
8: -0.0067460, -0.0054625, -0.0068313, -0.0055684, -0.0006059, 0.0007632
9: -0.0035385, -0.0034277, -0.0035293, -0.0034204, -0.0000658, 0.0000523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001461
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001527
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083611, -0.0072786, -0.0083296, -0.0072083, -0.0006316, 0.0005146
1: -0.0052960, -0.0049908, -0.0052871, -0.0049709, -0.0001781, 0.0001451
2: -0.0005150, 0.0017370, -0.0004494, 0.0018832, -0.0013139, 0.0010705
3: 0.0015591, 0.0018572, 0.0015678, 0.0018765, -0.0001739, 0.0001417
4: 0.0047938, 0.0064768, 0.0046845, 0.0064277, -0.0008000, 0.0009819
5: 0.9968381, 0.9973057, 0.9968078, 0.9972920, -0.0002223, 0.0002728
6: 0.0050136, 0.0054380, 0.0049860, 0.0054256, -0.0002018, 0.0002476
7: -0.0046717, -0.0030878, -0.0047745, -0.0031340, -0.0007529, 0.0009241
8: -0.0067896, -0.0055569, -0.0067537, -0.0054769, -0.0007192, 0.0005860
9: -0.0035303, -0.0034240, -0.0035372, -0.0034271, -0.0000506, 0.0000620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001358, upper bound: 0.0001363
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001531
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083810, -0.0072878, -0.0083389, -0.0072076, -0.0006433, 0.0005343
1: -0.0053016, -0.0049934, -0.0052897, -0.0049708, -0.0001814, 0.0001506
2: -0.0005564, 0.0017177, -0.0004687, 0.0018845, -0.0013382, 0.0011115
3: 0.0015537, 0.0018546, 0.0015653, 0.0018767, -0.0001771, 0.0001471
4: 0.0048081, 0.0065077, 0.0046835, 0.0064421, -0.0008306, 0.0010001
5: 0.9968421, 0.9973143, 0.9968075, 0.9972960, -0.0002308, 0.0002778
6: 0.0050172, 0.0054458, 0.0049858, 0.0054293, -0.0002095, 0.0002522
7: -0.0046582, -0.0030587, -0.0047755, -0.0031204, -0.0007817, 0.0009412
8: -0.0068123, -0.0055674, -0.0067643, -0.0054761, -0.0007325, 0.0006084
9: -0.0035294, -0.0034220, -0.0035373, -0.0034261, -0.0000525, 0.0000632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001447, upper bound: 0.0001580
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001447, upper bound: 0.0001586
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0083074, -0.0071962, -0.0083296, -0.0072083, -0.0005153, 0.0005221
1: -0.0052808, -0.0049675, -0.0052871, -0.0049709, -0.0001453, 0.0001472
2: -0.0004033, 0.0019083, -0.0004494, 0.0018832, -0.0010720, 0.0010861
3: 0.0015739, 0.0018798, 0.0015678, 0.0018765, -0.0001419, 0.0001437
4: 0.0046658, 0.0063933, 0.0046845, 0.0064277, -0.0008117, 0.0008011
5: 0.9968026, 0.9972826, 0.9968078, 0.9972920, -0.0002255, 0.0002226
6: 0.0049813, 0.0054170, 0.0049860, 0.0054256, -0.0002047, 0.0002020
7: -0.0047921, -0.0031663, -0.0047745, -0.0031340, -0.0007639, 0.0007540
8: -0.0067285, -0.0054631, -0.0067537, -0.0054769, -0.0005868, 0.0005945
9: -0.0035384, -0.0034292, -0.0035372, -0.0034271, -0.0000513, 0.0000506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001573, upper bound: 0.0001493
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001574, upper bound: 0.0001522
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0083276, -0.0072053, -0.0083389, -0.0072076, -0.0005235, 0.0005441
1: -0.0052865, -0.0049701, -0.0052897, -0.0049708, -0.0001476, 0.0001534
2: -0.0004453, 0.0018893, -0.0004687, 0.0018845, -0.0010890, 0.0011319
3: 0.0015684, 0.0018773, 0.0015653, 0.0018767, -0.0001441, 0.0001498
4: 0.0046799, 0.0064247, 0.0046835, 0.0064421, -0.0008459, 0.0008139
5: 0.9968066, 0.9972911, 0.9968075, 0.9972960, -0.0002350, 0.0002261
6: 0.0049849, 0.0054249, 0.0049858, 0.0054293, -0.0002133, 0.0002052
7: -0.0047788, -0.0031368, -0.0047755, -0.0031204, -0.0007961, 0.0007659
8: -0.0067515, -0.0054735, -0.0067643, -0.0054761, -0.0005961, 0.0006196
9: -0.0035375, -0.0034273, -0.0035373, -0.0034261, -0.0000535, 0.0000514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001569, upper bound: 0.0001563
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001569, upper bound: 0.0001563
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083609, -0.0072787, -0.0083308, -0.0071995, -0.0006466, 0.0005164
1: -0.0052959, -0.0049908, -0.0052874, -0.0049685, -0.0001823, 0.0001456
2: -0.0005146, 0.0017366, -0.0004520, 0.0019014, -0.0013451, 0.0010743
3: 0.0015592, 0.0018571, 0.0015675, 0.0018789, -0.0001780, 0.0001422
4: 0.0047940, 0.0064764, 0.0046709, 0.0064296, -0.0008029, 0.0010052
5: 0.9968382, 0.9973056, 0.9968040, 0.9972926, -0.0002231, 0.0002793
6: 0.0050137, 0.0054379, 0.0049826, 0.0054261, -0.0002025, 0.0002535
7: -0.0046714, -0.0030881, -0.0047873, -0.0031321, -0.0007556, 0.0009460
8: -0.0067894, -0.0055571, -0.0067551, -0.0054669, -0.0007363, 0.0005881
9: -0.0035303, -0.0034240, -0.0035381, -0.0034269, -0.0000507, 0.0000635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001370, upper bound: 0.0001362
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001531
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083805, -0.0072880, -0.0083400, -0.0071989, -0.0006593, 0.0005363
1: -0.0053014, -0.0049934, -0.0052900, -0.0049683, -0.0001859, 0.0001512
2: -0.0005552, 0.0017174, -0.0004711, 0.0019027, -0.0013714, 0.0011156
3: 0.0015538, 0.0018546, 0.0015650, 0.0018791, -0.0001815, 0.0001476
4: 0.0048084, 0.0065068, 0.0046699, 0.0064439, -0.0008337, 0.0010249
5: 0.9968421, 0.9973140, 0.9968036, 0.9972965, -0.0002316, 0.0002848
6: 0.0050173, 0.0054456, 0.0049824, 0.0054297, -0.0002102, 0.0002585
7: -0.0046579, -0.0030595, -0.0047882, -0.0031187, -0.0007846, 0.0009646
8: -0.0068116, -0.0055676, -0.0067656, -0.0054662, -0.0007507, 0.0006107
9: -0.0035294, -0.0034221, -0.0035381, -0.0034260, -0.0000527, 0.0000648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001580
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001586
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0083067, -0.0071964, -0.0083308, -0.0071995, -0.0005368, 0.0005226
1: -0.0052806, -0.0049676, -0.0052874, -0.0049685, -0.0001513, 0.0001473
2: -0.0004018, 0.0019078, -0.0004520, 0.0019014, -0.0011167, 0.0010871
3: 0.0015741, 0.0018798, 0.0015675, 0.0018789, -0.0001478, 0.0001439
4: 0.0046661, 0.0063921, 0.0046709, 0.0064296, -0.0008125, 0.0008345
5: 0.9968026, 0.9972821, 0.9968040, 0.9972926, -0.0002257, 0.0002319
6: 0.0049814, 0.0054167, 0.0049826, 0.0054261, -0.0002049, 0.0002105
7: -0.0047919, -0.0031675, -0.0047873, -0.0031321, -0.0007646, 0.0007854
8: -0.0067276, -0.0054634, -0.0067551, -0.0054669, -0.0006113, 0.0005951
9: -0.0035384, -0.0034293, -0.0035381, -0.0034269, -0.0000513, 0.0000527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001589, upper bound: 0.0001493
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001590, upper bound: 0.0001522
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0083272, -0.0072055, -0.0083400, -0.0071989, -0.0005463, 0.0005447
1: -0.0052864, -0.0049702, -0.0052900, -0.0049683, -0.0001540, 0.0001536
2: -0.0004444, 0.0018889, -0.0004711, 0.0019027, -0.0011363, 0.0011331
3: 0.0015685, 0.0018773, 0.0015650, 0.0018791, -0.0001504, 0.0001499
4: 0.0046802, 0.0064240, 0.0046699, 0.0064439, -0.0008468, 0.0008492
5: 0.9968066, 0.9972910, 0.9968036, 0.9972965, -0.0002353, 0.0002359
6: 0.0049849, 0.0054247, 0.0049824, 0.0054297, -0.0002136, 0.0002142
7: -0.0047785, -0.0031374, -0.0047882, -0.0031187, -0.0007969, 0.0007992
8: -0.0067510, -0.0054737, -0.0067656, -0.0054662, -0.0006220, 0.0006203
9: -0.0035375, -0.0034273, -0.0035381, -0.0034260, -0.0000535, 0.0000537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001563
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001563
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083703, -0.0072895, -0.0083805, -0.0072805, -0.0004982, 0.0005407
1: -0.0052986, -0.0049939, -0.0053014, -0.0049913, -0.0001405, 0.0001524
2: -0.0005340, 0.0017141, -0.0005552, 0.0017330, -0.0010364, 0.0011247
3: 0.0015566, 0.0018541, 0.0015538, 0.0018566, -0.0001372, 0.0001488
4: 0.0048108, 0.0064910, 0.0047967, 0.0065068, -0.0008405, 0.0007745
5: 0.9968429, 0.9973096, 0.9968389, 0.9973140, -0.0002335, 0.0002152
6: 0.0050179, 0.0054416, 0.0050143, 0.0054456, -0.0002120, 0.0001953
7: -0.0046556, -0.0030744, -0.0046689, -0.0030595, -0.0007910, 0.0007289
8: -0.0068000, -0.0055694, -0.0068116, -0.0055591, -0.0005673, 0.0006157
9: -0.0035292, -0.0034231, -0.0035301, -0.0034221, -0.0000531, 0.0000489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001493
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001496
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083181, -0.0072085, -0.0083805, -0.0072805, -0.0004965, 0.0006612
1: -0.0052838, -0.0049710, -0.0053014, -0.0049913, -0.0001400, 0.0001864
2: -0.0004255, 0.0018826, -0.0005552, 0.0017330, -0.0010329, 0.0013753
3: 0.0015710, 0.0018764, 0.0015538, 0.0018566, -0.0001367, 0.0001820
4: 0.0046849, 0.0064098, 0.0047967, 0.0065068, -0.0010278, 0.0007719
5: 0.9968079, 0.9972871, 0.9968389, 0.9973140, -0.0002856, 0.0002145
6: 0.0049861, 0.0054211, 0.0050143, 0.0054456, -0.0002592, 0.0001947
7: -0.0047741, -0.0031508, -0.0046689, -0.0030595, -0.0009673, 0.0007264
8: -0.0067406, -0.0054772, -0.0068116, -0.0055591, -0.0005654, 0.0007529
9: -0.0035372, -0.0034282, -0.0035301, -0.0034221, -0.0000650, 0.0000488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001493
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001496
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083794, -0.0072888, -0.0084014, -0.0072884, -0.0005206, 0.0005499
1: -0.0053011, -0.0049937, -0.0053073, -0.0049935, -0.0001468, 0.0001550
2: -0.0005531, 0.0017156, -0.0005987, 0.0017165, -0.0010830, 0.0011440
3: 0.0015541, 0.0018543, 0.0015481, 0.0018544, -0.0001433, 0.0001514
4: 0.0048097, 0.0065052, 0.0048091, 0.0065393, -0.0008549, 0.0008094
5: 0.9968426, 0.9973136, 0.9968424, 0.9973230, -0.0002375, 0.0002249
6: 0.0050176, 0.0054452, 0.0050175, 0.0054538, -0.0002156, 0.0002041
7: -0.0046566, -0.0030610, -0.0046572, -0.0030289, -0.0008046, 0.0007617
8: -0.0068105, -0.0055686, -0.0068355, -0.0055681, -0.0005928, 0.0006262
9: -0.0035293, -0.0034222, -0.0035293, -0.0034200, -0.0000540, 0.0000511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001450
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001512
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083271, -0.0072079, -0.0084014, -0.0072884, -0.0005168, 0.0006740
1: -0.0052864, -0.0049708, -0.0053073, -0.0049935, -0.0001457, 0.0001900
2: -0.0004442, 0.0018840, -0.0005987, 0.0017165, -0.0010751, 0.0014021
3: 0.0015685, 0.0018766, 0.0015481, 0.0018544, -0.0001423, 0.0001855
4: 0.0046839, 0.0064239, 0.0048091, 0.0065393, -0.0010478, 0.0008034
5: 0.9968076, 0.9972910, 0.9968424, 0.9973230, -0.0002911, 0.0002232
6: 0.0049859, 0.0054247, 0.0050175, 0.0054538, -0.0002642, 0.0002026
7: -0.0047751, -0.0031376, -0.0046572, -0.0030289, -0.0009861, 0.0007561
8: -0.0067509, -0.0054764, -0.0068355, -0.0055681, -0.0005885, 0.0007675
9: -0.0035373, -0.0034273, -0.0035293, -0.0034200, -0.0000662, 0.0000508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001450
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001512
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083726, -0.0072796, -0.0083796, -0.0072807, -0.0005002, 0.0005606
1: -0.0052992, -0.0049910, -0.0053012, -0.0049914, -0.0001410, 0.0001580
2: -0.0005389, 0.0017349, -0.0005533, 0.0017325, -0.0010404, 0.0011661
3: 0.0015560, 0.0018569, 0.0015541, 0.0018566, -0.0001377, 0.0001543
4: 0.0047953, 0.0064946, 0.0047971, 0.0065054, -0.0008715, 0.0007775
5: 0.9968385, 0.9973106, 0.9968390, 0.9973136, -0.0002421, 0.0002160
6: 0.0050140, 0.0054425, 0.0050144, 0.0054452, -0.0002198, 0.0001961
7: -0.0046702, -0.0030710, -0.0046686, -0.0030608, -0.0008201, 0.0007318
8: -0.0068027, -0.0055580, -0.0068106, -0.0055593, -0.0005695, 0.0006383
9: -0.0035302, -0.0034228, -0.0035301, -0.0034221, -0.0000551, 0.0000491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001511
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001513
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083191, -0.0071998, -0.0083796, -0.0072807, -0.0004985, 0.0006750
1: -0.0052841, -0.0049685, -0.0053012, -0.0049914, -0.0001406, 0.0001903
2: -0.0004275, 0.0019009, -0.0005533, 0.0017325, -0.0010370, 0.0014042
3: 0.0015707, 0.0018788, 0.0015541, 0.0018566, -0.0001372, 0.0001858
4: 0.0046713, 0.0064114, 0.0047971, 0.0065054, -0.0010494, 0.0007750
5: 0.9968041, 0.9972876, 0.9968390, 0.9973136, -0.0002916, 0.0002153
6: 0.0049827, 0.0054215, 0.0050144, 0.0054452, -0.0002646, 0.0001954
7: -0.0047870, -0.0031493, -0.0046686, -0.0030608, -0.0009876, 0.0007294
8: -0.0067417, -0.0054672, -0.0068106, -0.0055593, -0.0005677, 0.0007687
9: -0.0035381, -0.0034281, -0.0035301, -0.0034221, -0.0000663, 0.0000490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001512
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001513
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083818, -0.0072789, -0.0084006, -0.0072886, -0.0005226, 0.0005714
1: -0.0053018, -0.0049908, -0.0053071, -0.0049936, -0.0001473, 0.0001611
2: -0.0005581, 0.0017363, -0.0005971, 0.0017160, -0.0010872, 0.0011887
3: 0.0015534, 0.0018571, 0.0015483, 0.0018544, -0.0001439, 0.0001573
4: 0.0047942, 0.0065089, 0.0048094, 0.0065381, -0.0008883, 0.0008125
5: 0.9968383, 0.9973146, 0.9968424, 0.9973227, -0.0002468, 0.0002257
6: 0.0050137, 0.0054461, 0.0050175, 0.0054535, -0.0002240, 0.0002049
7: -0.0046712, -0.0030575, -0.0046569, -0.0030301, -0.0008360, 0.0007646
8: -0.0068132, -0.0055572, -0.0068346, -0.0055684, -0.0005951, 0.0006507
9: -0.0035303, -0.0034219, -0.0035293, -0.0034201, -0.0000561, 0.0000513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001464
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001537
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083281, -0.0071991, -0.0084006, -0.0072886, -0.0005188, 0.0006889
1: -0.0052867, -0.0049684, -0.0053071, -0.0049936, -0.0001463, 0.0001942
2: -0.0004464, 0.0019022, -0.0005971, 0.0017160, -0.0010793, 0.0014330
3: 0.0015682, 0.0018790, 0.0015483, 0.0018544, -0.0001428, 0.0001896
4: 0.0046703, 0.0064254, 0.0048094, 0.0065381, -0.0010709, 0.0008066
5: 0.9968037, 0.9972914, 0.9968424, 0.9973227, -0.0002975, 0.0002241
6: 0.0049825, 0.0054251, 0.0050175, 0.0054535, -0.0002701, 0.0002034
7: -0.0047879, -0.0031361, -0.0046569, -0.0030301, -0.0010079, 0.0007591
8: -0.0067521, -0.0054665, -0.0068346, -0.0055684, -0.0005908, 0.0007844
9: -0.0035381, -0.0034272, -0.0035293, -0.0034201, -0.0000677, 0.0000510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001464
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001536
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083480, -0.0072816, -0.0083491, -0.0072074, -0.0006030, 0.0005529
1: -0.0052923, -0.0049916, -0.0052926, -0.0049707, -0.0001700, 0.0001559
2: -0.0004877, 0.0017307, -0.0004901, 0.0018850, -0.0012544, 0.0011502
3: 0.0015628, 0.0018563, 0.0015624, 0.0018767, -0.0001660, 0.0001522
4: 0.0047985, 0.0064563, 0.0046831, 0.0064581, -0.0008596, 0.0009375
5: 0.9968394, 0.9973000, 0.9968075, 0.9973005, -0.0002388, 0.0002605
6: 0.0050148, 0.0054329, 0.0049857, 0.0054333, -0.0002168, 0.0002364
7: -0.0046672, -0.0031070, -0.0047758, -0.0031053, -0.0008090, 0.0008823
8: -0.0067747, -0.0055603, -0.0067760, -0.0054759, -0.0006867, 0.0006296
9: -0.0035300, -0.0034253, -0.0035373, -0.0034251, -0.0000543, 0.0000592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001498, upper bound: 0.0001535
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001498, upper bound: 0.0001535
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083583, -0.0072067, -0.0006161, 0.0005737
1: -0.0052982, -0.0049939, -0.0052952, -0.0049705, -0.0001737, 0.0001617
2: -0.0005318, 0.0017140, -0.0005091, 0.0018864, -0.0012816, 0.0011934
3: 0.0015569, 0.0018541, 0.0015599, 0.0018769, -0.0001696, 0.0001579
4: 0.0048109, 0.0064893, 0.0046821, 0.0064723, -0.0008918, 0.0009578
5: 0.9968429, 0.9973091, 0.9968071, 0.9973044, -0.0002478, 0.0002661
6: 0.0050179, 0.0054412, 0.0049854, 0.0054369, -0.0002249, 0.0002415
7: -0.0046555, -0.0030760, -0.0047767, -0.0030920, -0.0008393, 0.0009014
8: -0.0067988, -0.0055695, -0.0067864, -0.0054751, -0.0007016, 0.0006533
9: -0.0035292, -0.0034232, -0.0035374, -0.0034242, -0.0000564, 0.0000605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001573
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001579
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083181, -0.0072085, -0.0083269, -0.0071995, -0.0005026, 0.0005448
1: -0.0052838, -0.0049710, -0.0052863, -0.0049685, -0.0001417, 0.0001536
2: -0.0004255, 0.0018826, -0.0004438, 0.0019013, -0.0010456, 0.0011333
3: 0.0015710, 0.0018764, 0.0015686, 0.0018789, -0.0001384, 0.0001500
4: 0.0046849, 0.0064098, 0.0046709, 0.0064235, -0.0008470, 0.0007814
5: 0.9968079, 0.9972871, 0.9968041, 0.9972908, -0.0002353, 0.0002171
6: 0.0049861, 0.0054211, 0.0049826, 0.0054246, -0.0002136, 0.0001971
7: -0.0047741, -0.0031508, -0.0047873, -0.0031379, -0.0007971, 0.0007354
8: -0.0067406, -0.0054772, -0.0067507, -0.0054669, -0.0005723, 0.0006204
9: -0.0035372, -0.0034282, -0.0035381, -0.0034273, -0.0000535, 0.0000494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001546
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001546
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083271, -0.0072079, -0.0083481, -0.0072075, -0.0005251, 0.0005542
1: -0.0052864, -0.0049708, -0.0052923, -0.0049707, -0.0001480, 0.0001563
2: -0.0004442, 0.0018840, -0.0004879, 0.0018848, -0.0010922, 0.0011529
3: 0.0015685, 0.0018766, 0.0015627, 0.0018767, -0.0001445, 0.0001526
4: 0.0046839, 0.0064239, 0.0046833, 0.0064565, -0.0008616, 0.0008163
5: 0.9968076, 0.9972910, 0.9968075, 0.9973000, -0.0002394, 0.0002268
6: 0.0049859, 0.0054247, 0.0049857, 0.0054329, -0.0002173, 0.0002058
7: -0.0047751, -0.0031376, -0.0047757, -0.0031068, -0.0008109, 0.0007682
8: -0.0067509, -0.0054764, -0.0067748, -0.0054760, -0.0005979, 0.0006311
9: -0.0035373, -0.0034273, -0.0035373, -0.0034252, -0.0000544, 0.0000516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001526
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001546
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083500, -0.0072718, -0.0083484, -0.0072076, -0.0006050, 0.0005730
1: -0.0052928, -0.0049889, -0.0052924, -0.0049708, -0.0001706, 0.0001616
2: -0.0004919, 0.0017510, -0.0004885, 0.0018845, -0.0012585, 0.0011920
3: 0.0015622, 0.0018590, 0.0015627, 0.0018767, -0.0001665, 0.0001577
4: 0.0047833, 0.0064595, 0.0046835, 0.0064569, -0.0008908, 0.0009405
5: 0.9968352, 0.9973009, 0.9968075, 0.9973001, -0.0002475, 0.0002613
6: 0.0050109, 0.0054337, 0.0049858, 0.0054330, -0.0002247, 0.0002372
7: -0.0046815, -0.0031041, -0.0047755, -0.0031065, -0.0008384, 0.0008851
8: -0.0067770, -0.0055492, -0.0067751, -0.0054761, -0.0006889, 0.0006525
9: -0.0035310, -0.0034251, -0.0035373, -0.0034252, -0.0000563, 0.0000594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001498, upper bound: 0.0001558
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001498, upper bound: 0.0001558
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083717, -0.0072796, -0.0083574, -0.0072070, -0.0006183, 0.0005901
1: -0.0052989, -0.0049911, -0.0052949, -0.0049706, -0.0001743, 0.0001664
2: -0.0005369, 0.0017347, -0.0005072, 0.0018859, -0.0012862, 0.0012275
3: 0.0015562, 0.0018569, 0.0015602, 0.0018769, -0.0001702, 0.0001624
4: 0.0047954, 0.0064931, 0.0046825, 0.0064709, -0.0009173, 0.0009613
5: 0.9968385, 0.9973102, 0.9968072, 0.9973041, -0.0002549, 0.0002671
6: 0.0050140, 0.0054421, 0.0049855, 0.0054365, -0.0002313, 0.0002424
7: -0.0046701, -0.0030724, -0.0047764, -0.0030933, -0.0008633, 0.0009046
8: -0.0068016, -0.0055581, -0.0067854, -0.0054754, -0.0007041, 0.0006719
9: -0.0035302, -0.0034229, -0.0035373, -0.0034243, -0.0000580, 0.0000607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001588
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001593
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083191, -0.0071998, -0.0083263, -0.0071998, -0.0005026, 0.0005651
1: -0.0052841, -0.0049685, -0.0052862, -0.0049685, -0.0001417, 0.0001593
2: -0.0004275, 0.0019009, -0.0004426, 0.0019009, -0.0010455, 0.0011756
3: 0.0015707, 0.0018788, 0.0015687, 0.0018788, -0.0001384, 0.0001556
4: 0.0046713, 0.0064114, 0.0046713, 0.0064226, -0.0008786, 0.0007814
5: 0.9968041, 0.9972876, 0.9968041, 0.9972907, -0.0002441, 0.0002171
6: 0.0049827, 0.0054215, 0.0049827, 0.0054244, -0.0002216, 0.0001970
7: -0.0047870, -0.0031493, -0.0047870, -0.0031387, -0.0008268, 0.0007353
8: -0.0067417, -0.0054672, -0.0067500, -0.0054672, -0.0005723, 0.0006435
9: -0.0035381, -0.0034281, -0.0035381, -0.0034274, -0.0000555, 0.0000494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001572
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001572
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083281, -0.0071991, -0.0083471, -0.0072077, -0.0005251, 0.0005759
1: -0.0052867, -0.0049684, -0.0052920, -0.0049708, -0.0001480, 0.0001624
2: -0.0004464, 0.0019022, -0.0004858, 0.0018843, -0.0010923, 0.0011979
3: 0.0015682, 0.0018790, 0.0015630, 0.0018767, -0.0001445, 0.0001585
4: 0.0046703, 0.0064254, 0.0046836, 0.0064549, -0.0008952, 0.0008163
5: 0.9968037, 0.9972914, 0.9968076, 0.9972996, -0.0002487, 0.0002268
6: 0.0049825, 0.0054251, 0.0049858, 0.0054325, -0.0002258, 0.0002059
7: -0.0047879, -0.0031361, -0.0047753, -0.0031083, -0.0008425, 0.0007682
8: -0.0067521, -0.0054665, -0.0067736, -0.0054762, -0.0005979, 0.0006557
9: -0.0035381, -0.0034272, -0.0035373, -0.0034253, -0.0000566, 0.0000516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001546
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001572
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.59 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001157, upper bound: 0.0001345
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001449
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001157, upper bound: 0.0001346
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001449
IS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001436
IS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001503
IS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001435
IS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001502
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001145, upper bound: 0.0001360
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001471
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001145, upper bound: 0.0001362
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001471
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001461
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001527
IS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001461
IS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001527
IS_A1_B2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001358, upper bound: 0.0001363
IS_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001531
IS_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001447, upper bound: 0.0001580
IS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001447, upper bound: 0.0001586
IS_A1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001573, upper bound: 0.0001493
IS_A1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001574, upper bound: 0.0001522
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001569, upper bound: 0.0001563
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001569, upper bound: 0.0001563
IS_A1_B2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001370, upper bound: 0.0001362
IS_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001531
IS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001580
IS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001463, upper bound: 0.0001586
IS_A1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001589, upper bound: 0.0001493
IS_A1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001590, upper bound: 0.0001522
IS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001563
IS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001563
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001493
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001496
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001493
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001496
IS_A2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001450
IS_A2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001512
IS_A2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001450
IS_A2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001512
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001511
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001513
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001512
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001513
IS_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001464
IS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001537
IS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001464
IS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001536
IS_A2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001498, upper bound: 0.0001535
IS_A2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001498, upper bound: 0.0001535
IS_A2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001573
IS_A2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001579
IS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001546
IS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001546
IS_A2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001526
IS_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001546
IS_A2_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001498, upper bound: 0.0001558
IS_A2_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001498, upper bound: 0.0001558
IS_A2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001588
IS_A2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001464, upper bound: 0.0001593
IS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001572
IS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001572
IS_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001546
IS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 5, lower bound: -0.0001616, upper bound: 0.0001572

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083630, -0.0072886, -0.0083688, -0.0072812, -0.0005089, 0.0005003
1: -0.0052965, -0.0049936, -0.0052981, -0.0049915, -0.0001435, 0.0001411
2: -0.0005190, 0.0017160, -0.0005310, 0.0017315, -0.0010586, 0.0010408
3: 0.0015586, 0.0018544, 0.0015570, 0.0018564, -0.0001401, 0.0001377
4: 0.0048095, 0.0064797, 0.0047979, 0.0064887, -0.0007778, 0.0007912
5: 0.9968425, 0.9973065, 0.9968393, 0.9973090, -0.0002161, 0.0002198
6: 0.0050175, 0.0054388, 0.0050146, 0.0054410, -0.0001962, 0.0001995
7: -0.0046569, -0.0030850, -0.0046678, -0.0030765, -0.0007320, 0.0007446
8: -0.0067918, -0.0055684, -0.0067984, -0.0055599, -0.0005795, 0.0005697
9: -0.0035293, -0.0034238, -0.0035301, -0.0034232, -0.0000492, 0.0000500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001425, upper bound: 0.0001449
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001425, upper bound: 0.0001449
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083106, -0.0072062, -0.0083688, -0.0072812, -0.0005071, 0.0006262
1: -0.0052817, -0.0049703, -0.0052981, -0.0049915, -0.0001430, 0.0001766
2: -0.0004099, 0.0018875, -0.0005310, 0.0017315, -0.0010550, 0.0013027
3: 0.0015731, 0.0018771, 0.0015570, 0.0018564, -0.0001396, 0.0001724
4: 0.0046812, 0.0063982, 0.0047979, 0.0064887, -0.0009735, 0.0007884
5: 0.9968069, 0.9972838, 0.9968393, 0.9973090, -0.0002705, 0.0002190
6: 0.0049852, 0.0054182, 0.0050146, 0.0054410, -0.0002455, 0.0001988
7: -0.0047776, -0.0031617, -0.0046678, -0.0030765, -0.0009162, 0.0007420
8: -0.0067321, -0.0054745, -0.0067984, -0.0055599, -0.0005775, 0.0007131
9: -0.0035374, -0.0034289, -0.0035301, -0.0034232, -0.0000615, 0.0000498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001518, upper bound: 0.0001449
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001518, upper bound: 0.0001449
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083429, -0.0072793, -0.0083984, -0.0072885, -0.0004903, 0.0005466
1: -0.0052908, -0.0049910, -0.0053065, -0.0049936, -0.0001382, 0.0001541
2: -0.0004771, 0.0017354, -0.0005926, 0.0017163, -0.0010199, 0.0011371
3: 0.0015642, 0.0018570, 0.0015489, 0.0018544, -0.0001350, 0.0001505
4: 0.0047949, 0.0064484, 0.0048092, 0.0065347, -0.0008498, 0.0007622
5: 0.9968384, 0.9972979, 0.9968424, 0.9973217, -0.0002361, 0.0002118
6: 0.0050139, 0.0054309, 0.0050175, 0.0054526, -0.0002143, 0.0001922
7: -0.0046706, -0.0031144, -0.0046572, -0.0030332, -0.0007998, 0.0007174
8: -0.0067689, -0.0055577, -0.0068321, -0.0055682, -0.0005583, 0.0006224
9: -0.0035302, -0.0034257, -0.0035293, -0.0034203, -0.0000537, 0.0000482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001309, upper bound: 0.0001155
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001405
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083631, -0.0072886, -0.0083984, -0.0072885, -0.0004996, 0.0005235
1: -0.0052965, -0.0049936, -0.0053065, -0.0049936, -0.0001408, 0.0001476
2: -0.0005190, 0.0017161, -0.0005926, 0.0017163, -0.0010392, 0.0010890
3: 0.0015586, 0.0018544, 0.0015489, 0.0018544, -0.0001375, 0.0001441
4: 0.0048094, 0.0064797, 0.0048092, 0.0065347, -0.0008139, 0.0007766
5: 0.9968424, 0.9973065, 0.9968424, 0.9973217, -0.0002261, 0.0002158
6: 0.0050175, 0.0054388, 0.0050175, 0.0054526, -0.0002052, 0.0001959
7: -0.0046570, -0.0030850, -0.0046572, -0.0030332, -0.0007660, 0.0007309
8: -0.0067918, -0.0055683, -0.0068321, -0.0055682, -0.0005688, 0.0005961
9: -0.0035293, -0.0034238, -0.0035293, -0.0034203, -0.0000514, 0.0000491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001151, upper bound: 0.0001439
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001477
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0082901, -0.0071970, -0.0083984, -0.0072885, -0.0004923, 0.0006790
1: -0.0052759, -0.0049678, -0.0053065, -0.0049936, -0.0001388, 0.0001914
2: -0.0003671, 0.0019067, -0.0005926, 0.0017163, -0.0010240, 0.0014124
3: 0.0015787, 0.0018796, 0.0015489, 0.0018544, -0.0001355, 0.0001869
4: 0.0046670, 0.0063662, 0.0048092, 0.0065347, -0.0010555, 0.0007653
5: 0.9968029, 0.9972749, 0.9968424, 0.9973217, -0.0002933, 0.0002126
6: 0.0049816, 0.0054101, 0.0050175, 0.0054526, -0.0002662, 0.0001930
7: -0.0047910, -0.0031918, -0.0046572, -0.0030332, -0.0009934, 0.0007202
8: -0.0067087, -0.0054640, -0.0068321, -0.0055682, -0.0005606, 0.0007731
9: -0.0035383, -0.0034309, -0.0035293, -0.0034203, -0.0000667, 0.0000484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001472, upper bound: 0.0001215
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001534, upper bound: 0.0001405
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0083105, -0.0072061, -0.0083984, -0.0072885, -0.0004984, 0.0006540
1: -0.0052817, -0.0049703, -0.0053065, -0.0049936, -0.0001405, 0.0001844
2: -0.0004098, 0.0018876, -0.0005926, 0.0017163, -0.0010369, 0.0013604
3: 0.0015731, 0.0018771, 0.0015489, 0.0018544, -0.0001372, 0.0001800
4: 0.0046812, 0.0063981, 0.0048092, 0.0065347, -0.0010167, 0.0007749
5: 0.9968069, 0.9972839, 0.9968424, 0.9973217, -0.0002825, 0.0002153
6: 0.0049852, 0.0054182, 0.0050175, 0.0054526, -0.0002564, 0.0001954
7: -0.0047776, -0.0031618, -0.0046572, -0.0030332, -0.0009568, 0.0007292
8: -0.0067320, -0.0054744, -0.0068321, -0.0055682, -0.0005676, 0.0007447
9: -0.0035374, -0.0034289, -0.0035293, -0.0034203, -0.0000642, 0.0000490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001366, upper bound: 0.0001439
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001534, upper bound: 0.0001472
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083665, -0.0072786, -0.0083685, -0.0072814, -0.0005108, 0.0005214
1: -0.0052975, -0.0049908, -0.0052981, -0.0049916, -0.0001440, 0.0001470
2: -0.0005263, 0.0017368, -0.0005304, 0.0017310, -0.0010626, 0.0010846
3: 0.0015577, 0.0018571, 0.0015571, 0.0018564, -0.0001406, 0.0001435
4: 0.0047939, 0.0064852, 0.0047982, 0.0064883, -0.0008106, 0.0007941
5: 0.9968381, 0.9973080, 0.9968393, 0.9973089, -0.0002252, 0.0002206
6: 0.0050136, 0.0054401, 0.0050147, 0.0054409, -0.0002044, 0.0002003
7: -0.0046716, -0.0030799, -0.0046675, -0.0030770, -0.0007629, 0.0007473
8: -0.0067958, -0.0055570, -0.0067981, -0.0055601, -0.0005817, 0.0005937
9: -0.0035303, -0.0034234, -0.0035300, -0.0034232, -0.0000512, 0.0000502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001471
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001471
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083122, -0.0071964, -0.0083685, -0.0072814, -0.0005091, 0.0006391
1: -0.0052822, -0.0049676, -0.0052981, -0.0049916, -0.0001435, 0.0001802
2: -0.0004133, 0.0019079, -0.0005304, 0.0017310, -0.0010591, 0.0013294
3: 0.0015726, 0.0018798, 0.0015571, 0.0018564, -0.0001402, 0.0001759
4: 0.0046660, 0.0064007, 0.0047982, 0.0064883, -0.0009935, 0.0007915
5: 0.9968026, 0.9972845, 0.9968393, 0.9973089, -0.0002760, 0.0002199
6: 0.0049814, 0.0054188, 0.0050147, 0.0054409, -0.0002506, 0.0001996
7: -0.0047919, -0.0031594, -0.0046675, -0.0030770, -0.0009350, 0.0007449
8: -0.0067339, -0.0054633, -0.0067981, -0.0055601, -0.0005797, 0.0007277
9: -0.0035384, -0.0034288, -0.0035300, -0.0034232, -0.0000628, 0.0000500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001471
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001471
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083466, -0.0072674, -0.0083977, -0.0072887, -0.0004921, 0.0005747
1: -0.0052919, -0.0049876, -0.0053063, -0.0049936, -0.0001388, 0.0001620
2: -0.0004848, 0.0017601, -0.0005912, 0.0017159, -0.0010238, 0.0011956
3: 0.0015631, 0.0018602, 0.0015491, 0.0018544, -0.0001355, 0.0001582
4: 0.0047765, 0.0064541, 0.0048095, 0.0065337, -0.0008935, 0.0007651
5: 0.9968333, 0.9972994, 0.9968424, 0.9973215, -0.0002482, 0.0002126
6: 0.0050092, 0.0054323, 0.0050176, 0.0054524, -0.0002253, 0.0001929
7: -0.0046879, -0.0031091, -0.0046569, -0.0030342, -0.0008409, 0.0007200
8: -0.0067731, -0.0055442, -0.0068313, -0.0055684, -0.0005604, 0.0006544
9: -0.0035314, -0.0034254, -0.0035293, -0.0034204, -0.0000565, 0.0000483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001275, upper bound: 0.0001143
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001434
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083656, -0.0072786, -0.0083977, -0.0072887, -0.0005012, 0.0005509
1: -0.0052972, -0.0049908, -0.0053063, -0.0049936, -0.0001413, 0.0001553
2: -0.0005242, 0.0017369, -0.0005912, 0.0017159, -0.0010426, 0.0011460
3: 0.0015579, 0.0018571, 0.0015491, 0.0018544, -0.0001380, 0.0001517
4: 0.0047938, 0.0064836, 0.0048095, 0.0065337, -0.0008564, 0.0007792
5: 0.9968381, 0.9973075, 0.9968424, 0.9973215, -0.0002379, 0.0002165
6: 0.0050136, 0.0054397, 0.0050176, 0.0054524, -0.0002160, 0.0001965
7: -0.0046716, -0.0030813, -0.0046569, -0.0030342, -0.0008060, 0.0007333
8: -0.0067947, -0.0055569, -0.0068313, -0.0055684, -0.0005707, 0.0006273
9: -0.0035303, -0.0034235, -0.0035293, -0.0034204, -0.0000541, 0.0000492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001141, upper bound: 0.0001463
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001502
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0082921, -0.0071875, -0.0083977, -0.0072887, -0.0004943, 0.0006960
1: -0.0052765, -0.0049651, -0.0053063, -0.0049936, -0.0001394, 0.0001962
2: -0.0003714, 0.0019263, -0.0005912, 0.0017159, -0.0010282, 0.0014479
3: 0.0015781, 0.0018822, 0.0015491, 0.0018544, -0.0001361, 0.0001916
4: 0.0046522, 0.0063694, 0.0048095, 0.0065337, -0.0010821, 0.0007684
5: 0.9967988, 0.9972758, 0.9968424, 0.9973215, -0.0003006, 0.0002135
6: 0.0049779, 0.0054109, 0.0050176, 0.0054524, -0.0002729, 0.0001938
7: -0.0048049, -0.0031888, -0.0046569, -0.0030342, -0.0010184, 0.0007232
8: -0.0067110, -0.0054532, -0.0068313, -0.0055684, -0.0005628, 0.0007926
9: -0.0035393, -0.0034307, -0.0035293, -0.0034204, -0.0000684, 0.0000486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001469, upper bound: 0.0001208
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001534, upper bound: 0.0001434
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0083126, -0.0071964, -0.0083977, -0.0072887, -0.0005002, 0.0006697
1: -0.0052823, -0.0049676, -0.0053063, -0.0049936, -0.0001410, 0.0001888
2: -0.0004139, 0.0019080, -0.0005912, 0.0017159, -0.0010404, 0.0013931
3: 0.0015725, 0.0018798, 0.0015491, 0.0018544, -0.0001377, 0.0001844
4: 0.0046660, 0.0064012, 0.0048095, 0.0065337, -0.0010411, 0.0007776
5: 0.9968026, 0.9972847, 0.9968424, 0.9973215, -0.0002893, 0.0002160
6: 0.0049814, 0.0054190, 0.0050176, 0.0054524, -0.0002626, 0.0001961
7: -0.0047919, -0.0031589, -0.0046569, -0.0030342, -0.0009798, 0.0007318
8: -0.0067343, -0.0054633, -0.0068313, -0.0055684, -0.0005695, 0.0007626
9: -0.0035384, -0.0034287, -0.0035293, -0.0034204, -0.0000658, 0.0000491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001364, upper bound: 0.0001463
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001534, upper bound: 0.0001500
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083523, -0.0072792, -0.0083279, -0.0072084, -0.0006158, 0.0005027
1: -0.0052935, -0.0049909, -0.0052866, -0.0049710, -0.0001736, 0.0001417
2: -0.0004966, 0.0017356, -0.0004459, 0.0018830, -0.0012809, 0.0010456
3: 0.0015616, 0.0018570, 0.0015683, 0.0018765, -0.0001695, 0.0001384
4: 0.0047948, 0.0064630, 0.0046847, 0.0064251, -0.0007814, 0.0009573
5: 0.9968384, 0.9973019, 0.9968079, 0.9972913, -0.0002171, 0.0002660
6: 0.0050138, 0.0054345, 0.0049861, 0.0054250, -0.0001971, 0.0002414
7: -0.0046707, -0.0031007, -0.0047743, -0.0031364, -0.0007354, 0.0009009
8: -0.0067795, -0.0055576, -0.0067518, -0.0054770, -0.0007012, 0.0005724
9: -0.0035302, -0.0034248, -0.0035372, -0.0034272, -0.0000494, 0.0000605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001517
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001531
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083810, -0.0072878, -0.0083070, -0.0072003, -0.0006670, 0.0004967
1: -0.0053016, -0.0049934, -0.0052807, -0.0049687, -0.0001880, 0.0001400
2: -0.0005564, 0.0017177, -0.0004023, 0.0018997, -0.0013874, 0.0010333
3: 0.0015537, 0.0018546, 0.0015741, 0.0018787, -0.0001836, 0.0001367
4: 0.0048081, 0.0065077, 0.0046722, 0.0063925, -0.0007722, 0.0010369
5: 0.9968421, 0.9973143, 0.9968044, 0.9972823, -0.0002145, 0.0002881
6: 0.0050172, 0.0054458, 0.0049829, 0.0054168, -0.0001947, 0.0002615
7: -0.0046582, -0.0030587, -0.0047861, -0.0031670, -0.0007267, 0.0009758
8: -0.0068123, -0.0055674, -0.0067280, -0.0054678, -0.0007595, 0.0005656
9: -0.0035294, -0.0034220, -0.0035380, -0.0034293, -0.0000488, 0.0000655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001280, upper bound: 0.0001453
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001417, upper bound: 0.0001555
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083810, -0.0072878, -0.0083285, -0.0072083, -0.0006427, 0.0005022
1: -0.0053016, -0.0049934, -0.0052868, -0.0049710, -0.0001812, 0.0001416
2: -0.0005564, 0.0017177, -0.0004471, 0.0018830, -0.0013370, 0.0010447
3: 0.0015537, 0.0018546, 0.0015681, 0.0018765, -0.0001769, 0.0001382
4: 0.0048081, 0.0065077, 0.0046846, 0.0064260, -0.0007807, 0.0009992
5: 0.9968421, 0.9973143, 0.9968078, 0.9972915, -0.0002169, 0.0002776
6: 0.0050172, 0.0054458, 0.0049861, 0.0054252, -0.0001969, 0.0002520
7: -0.0046582, -0.0030587, -0.0047744, -0.0031356, -0.0007347, 0.0009404
8: -0.0068123, -0.0055674, -0.0067524, -0.0054770, -0.0007319, 0.0005719
9: -0.0035294, -0.0034220, -0.0035372, -0.0034272, -0.0000493, 0.0000631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001280, upper bound: 0.0001506
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001417, upper bound: 0.0001561
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0082879, -0.0071966, -0.0083236, -0.0072086, -0.0004952, 0.0005008
1: -0.0052753, -0.0049677, -0.0052854, -0.0049710, -0.0001396, 0.0001412
2: -0.0003627, 0.0019074, -0.0004370, 0.0018825, -0.0010302, 0.0010419
3: 0.0015793, 0.0018797, 0.0015695, 0.0018764, -0.0001363, 0.0001379
4: 0.0046664, 0.0063630, 0.0046850, 0.0064185, -0.0007786, 0.0007699
5: 0.9968027, 0.9972741, 0.9968079, 0.9972895, -0.0002163, 0.0002139
6: 0.0049815, 0.0054093, 0.0049862, 0.0054233, -0.0001964, 0.0001942
7: -0.0047916, -0.0031949, -0.0047740, -0.0031426, -0.0007328, 0.0007245
8: -0.0067063, -0.0054636, -0.0067469, -0.0054773, -0.0005639, 0.0005703
9: -0.0035384, -0.0034312, -0.0035372, -0.0034276, -0.0000492, 0.0000487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001573, upper bound: 0.0001485
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001573, upper bound: 0.0001493
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0082989, -0.0071965, -0.0083279, -0.0072084, -0.0005011, 0.0005064
1: -0.0052784, -0.0049676, -0.0052866, -0.0049710, -0.0001413, 0.0001428
2: -0.0003855, 0.0019076, -0.0004459, 0.0018830, -0.0010425, 0.0010535
3: 0.0015763, 0.0018797, 0.0015683, 0.0018765, -0.0001380, 0.0001394
4: 0.0046663, 0.0063800, 0.0046847, 0.0064251, -0.0007873, 0.0007791
5: 0.9968027, 0.9972787, 0.9968079, 0.9972913, -0.0002187, 0.0002164
6: 0.0049814, 0.0054136, 0.0049861, 0.0054250, -0.0001986, 0.0001965
7: -0.0047917, -0.0031789, -0.0047743, -0.0031364, -0.0007410, 0.0007332
8: -0.0067187, -0.0054635, -0.0067518, -0.0054770, -0.0005706, 0.0005767
9: -0.0035384, -0.0034301, -0.0035372, -0.0034272, -0.0000498, 0.0000492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001574, upper bound: 0.0001501
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001574, upper bound: 0.0001522
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083276, -0.0072053, -0.0083070, -0.0072003, -0.0005443, 0.0005004
1: -0.0052865, -0.0049701, -0.0052807, -0.0049687, -0.0001535, 0.0001411
2: -0.0004453, 0.0018893, -0.0004023, 0.0018997, -0.0011323, 0.0010409
3: 0.0015684, 0.0018773, 0.0015741, 0.0018787, -0.0001498, 0.0001377
4: 0.0046799, 0.0064247, 0.0046722, 0.0063925, -0.0007779, 0.0008462
5: 0.9968066, 0.9972911, 0.9968044, 0.9972823, -0.0002161, 0.0002351
6: 0.0049849, 0.0054249, 0.0049829, 0.0054168, -0.0001962, 0.0002134
7: -0.0047788, -0.0031368, -0.0047861, -0.0031670, -0.0007321, 0.0007964
8: -0.0067515, -0.0054735, -0.0067280, -0.0054678, -0.0006198, 0.0005698
9: -0.0035375, -0.0034273, -0.0035380, -0.0034293, -0.0000492, 0.0000535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001524, upper bound: 0.0001535
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001540, upper bound: 0.0001537
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083276, -0.0072053, -0.0083285, -0.0072083, -0.0005231, 0.0005090
1: -0.0052865, -0.0049701, -0.0052868, -0.0049710, -0.0001475, 0.0001435
2: -0.0004453, 0.0018893, -0.0004471, 0.0018830, -0.0010882, 0.0010588
3: 0.0015684, 0.0018773, 0.0015681, 0.0018765, -0.0001440, 0.0001401
4: 0.0046799, 0.0064247, 0.0046846, 0.0064260, -0.0007913, 0.0008132
5: 0.9968066, 0.9972911, 0.9968078, 0.9972915, -0.0002198, 0.0002259
6: 0.0049849, 0.0054249, 0.0049861, 0.0054252, -0.0001996, 0.0002051
7: -0.0047788, -0.0031368, -0.0047744, -0.0031356, -0.0007447, 0.0007653
8: -0.0067515, -0.0054735, -0.0067524, -0.0054770, -0.0005957, 0.0005796
9: -0.0035375, -0.0034273, -0.0035372, -0.0034272, -0.0000500, 0.0000514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001513
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001540, upper bound: 0.0001537
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083524, -0.0072794, -0.0083293, -0.0071996, -0.0006291, 0.0005049
1: -0.0052935, -0.0049910, -0.0052870, -0.0049685, -0.0001774, 0.0001424
2: -0.0004969, 0.0017352, -0.0004488, 0.0019012, -0.0013086, 0.0010503
3: 0.0015615, 0.0018569, 0.0015679, 0.0018789, -0.0001732, 0.0001390
4: 0.0047950, 0.0064632, 0.0046710, 0.0064272, -0.0007850, 0.0009780
5: 0.9968385, 0.9973019, 0.9968040, 0.9972919, -0.0002181, 0.0002717
6: 0.0050139, 0.0054346, 0.0049826, 0.0054255, -0.0001980, 0.0002466
7: -0.0046705, -0.0031005, -0.0047872, -0.0031344, -0.0007387, 0.0009204
8: -0.0067797, -0.0055578, -0.0067534, -0.0054670, -0.0007163, 0.0005750
9: -0.0035302, -0.0034248, -0.0035381, -0.0034271, -0.0000496, 0.0000618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001531
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001531
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083805, -0.0072880, -0.0083089, -0.0071909, -0.0006827, 0.0004986
1: -0.0053014, -0.0049934, -0.0052812, -0.0049660, -0.0001925, 0.0001406
2: -0.0005552, 0.0017174, -0.0004063, 0.0019194, -0.0014202, 0.0010372
3: 0.0015538, 0.0018546, 0.0015735, 0.0018813, -0.0001879, 0.0001373
4: 0.0048084, 0.0065068, 0.0046574, 0.0063955, -0.0007751, 0.0010614
5: 0.9968421, 0.9973140, 0.9968002, 0.9972831, -0.0002154, 0.0002949
6: 0.0050173, 0.0054456, 0.0049792, 0.0054175, -0.0001955, 0.0002677
7: -0.0046579, -0.0030595, -0.0048000, -0.0031642, -0.0007295, 0.0009989
8: -0.0068116, -0.0055676, -0.0067301, -0.0054570, -0.0007774, 0.0005678
9: -0.0035294, -0.0034221, -0.0035389, -0.0034291, -0.0000490, 0.0000671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001292, upper bound: 0.0001451
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001555
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083805, -0.0072880, -0.0083298, -0.0071996, -0.0006588, 0.0005038
1: -0.0053014, -0.0049934, -0.0052871, -0.0049685, -0.0001857, 0.0001420
2: -0.0005552, 0.0017174, -0.0004498, 0.0019012, -0.0013704, 0.0010480
3: 0.0015538, 0.0018546, 0.0015678, 0.0018789, -0.0001814, 0.0001387
4: 0.0048084, 0.0065068, 0.0046710, 0.0064280, -0.0007832, 0.0010241
5: 0.9968421, 0.9973140, 0.9968039, 0.9972921, -0.0002176, 0.0002845
6: 0.0050173, 0.0054456, 0.0049826, 0.0054257, -0.0001975, 0.0002583
7: -0.0046579, -0.0030595, -0.0047872, -0.0031337, -0.0007371, 0.0009638
8: -0.0068116, -0.0055676, -0.0067539, -0.0054670, -0.0007502, 0.0005737
9: -0.0035294, -0.0034221, -0.0035381, -0.0034270, -0.0000495, 0.0000647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001292, upper bound: 0.0001506
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001562
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0082872, -0.0071968, -0.0083246, -0.0071998, -0.0005159, 0.0005028
1: -0.0052751, -0.0049677, -0.0052857, -0.0049686, -0.0001455, 0.0001418
2: -0.0003613, 0.0019071, -0.0004391, 0.0019007, -0.0010732, 0.0010459
3: 0.0015795, 0.0018797, 0.0015692, 0.0018788, -0.0001420, 0.0001384
4: 0.0046667, 0.0063619, 0.0046714, 0.0064200, -0.0007817, 0.0008021
5: 0.9968027, 0.9972737, 0.9968041, 0.9972899, -0.0002172, 0.0002228
6: 0.0049815, 0.0054090, 0.0049827, 0.0054237, -0.0001971, 0.0002023
7: -0.0047913, -0.0031959, -0.0047868, -0.0031412, -0.0007356, 0.0007548
8: -0.0067055, -0.0054638, -0.0067481, -0.0054673, -0.0005875, 0.0005725
9: -0.0035383, -0.0034312, -0.0035380, -0.0034275, -0.0000494, 0.0000507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001584, upper bound: 0.0001493
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001584, upper bound: 0.0001493
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0082980, -0.0071967, -0.0083293, -0.0071996, -0.0005208, 0.0005082
1: -0.0052782, -0.0049677, -0.0052870, -0.0049685, -0.0001468, 0.0001433
2: -0.0003836, 0.0019072, -0.0004488, 0.0019012, -0.0010833, 0.0010573
3: 0.0015765, 0.0018797, 0.0015679, 0.0018789, -0.0001434, 0.0001399
4: 0.0046666, 0.0063786, 0.0046710, 0.0064272, -0.0007901, 0.0008096
5: 0.9968028, 0.9972785, 0.9968040, 0.9972919, -0.0002195, 0.0002249
6: 0.0049815, 0.0054133, 0.0049826, 0.0054255, -0.0001993, 0.0002042
7: -0.0047914, -0.0031802, -0.0047872, -0.0031344, -0.0007436, 0.0007619
8: -0.0067177, -0.0054637, -0.0067534, -0.0054670, -0.0005930, 0.0005787
9: -0.0035384, -0.0034302, -0.0035381, -0.0034271, -0.0000499, 0.0000512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001586, upper bound: 0.0001522
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001586, upper bound: 0.0001522
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083272, -0.0072055, -0.0083089, -0.0071909, -0.0005671, 0.0005009
1: -0.0052864, -0.0049702, -0.0052812, -0.0049660, -0.0001599, 0.0001412
2: -0.0004444, 0.0018889, -0.0004063, 0.0019194, -0.0011796, 0.0010419
3: 0.0015685, 0.0018773, 0.0015735, 0.0018813, -0.0001561, 0.0001379
4: 0.0046802, 0.0064240, 0.0046574, 0.0063955, -0.0007787, 0.0008816
5: 0.9968066, 0.9972910, 0.9968002, 0.9972831, -0.0002163, 0.0002449
6: 0.0049849, 0.0054247, 0.0049792, 0.0054175, -0.0001964, 0.0002223
7: -0.0047785, -0.0031374, -0.0048000, -0.0031642, -0.0007328, 0.0008297
8: -0.0067510, -0.0054737, -0.0067301, -0.0054570, -0.0006457, 0.0005704
9: -0.0035375, -0.0034273, -0.0035389, -0.0034291, -0.0000492, 0.0000557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001532, upper bound: 0.0001535
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001568, upper bound: 0.0001537
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083272, -0.0072055, -0.0083298, -0.0071996, -0.0005458, 0.0005094
1: -0.0052864, -0.0049702, -0.0052871, -0.0049685, -0.0001539, 0.0001436
2: -0.0004444, 0.0018889, -0.0004498, 0.0019012, -0.0011354, 0.0010597
3: 0.0015685, 0.0018773, 0.0015678, 0.0018789, -0.0001503, 0.0001402
4: 0.0046802, 0.0064240, 0.0046710, 0.0064280, -0.0007920, 0.0008486
5: 0.9968066, 0.9972910, 0.9968039, 0.9972921, -0.0002200, 0.0002358
6: 0.0049849, 0.0054247, 0.0049826, 0.0054257, -0.0001997, 0.0002140
7: -0.0047785, -0.0031374, -0.0047872, -0.0031337, -0.0007453, 0.0007986
8: -0.0067510, -0.0054737, -0.0067539, -0.0054670, -0.0006215, 0.0005801
9: -0.0035375, -0.0034273, -0.0035381, -0.0034270, -0.0000500, 0.0000536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001513
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001568, upper bound: 0.0001537
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083703, -0.0072895, -0.0083611, -0.0072786, -0.0005207, 0.0005109
1: -0.0052986, -0.0049939, -0.0052960, -0.0049908, -0.0001468, 0.0001440
2: -0.0005340, 0.0017141, -0.0005150, 0.0017370, -0.0010831, 0.0010628
3: 0.0015566, 0.0018541, 0.0015591, 0.0018572, -0.0001433, 0.0001406
4: 0.0048108, 0.0064910, 0.0047938, 0.0064768, -0.0007943, 0.0008094
5: 0.9968429, 0.9973096, 0.9968381, 0.9973057, -0.0002207, 0.0002249
6: 0.0050179, 0.0054416, 0.0050136, 0.0054380, -0.0002003, 0.0002041
7: -0.0046556, -0.0030744, -0.0046717, -0.0030878, -0.0007475, 0.0007618
8: -0.0068000, -0.0055694, -0.0067896, -0.0055569, -0.0005929, 0.0005818
9: -0.0035292, -0.0034231, -0.0035303, -0.0034240, -0.0000502, 0.0000512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001155, upper bound: 0.0001357
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001462
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083703, -0.0072895, -0.0083659, -0.0072808, -0.0004980, 0.0004975
1: -0.0052986, -0.0049939, -0.0052973, -0.0049914, -0.0001404, 0.0001403
2: -0.0005340, 0.0017141, -0.0005249, 0.0017322, -0.0010359, 0.0010348
3: 0.0015566, 0.0018541, 0.0015578, 0.0018565, -0.0001371, 0.0001369
4: 0.0048108, 0.0064910, 0.0047973, 0.0064841, -0.0007734, 0.0007742
5: 0.9968429, 0.9973096, 0.9968391, 0.9973077, -0.0002149, 0.0002151
6: 0.0050179, 0.0054416, 0.0050145, 0.0054399, -0.0001950, 0.0001952
7: -0.0046556, -0.0030744, -0.0046683, -0.0030809, -0.0007278, 0.0007286
8: -0.0068000, -0.0055694, -0.0067950, -0.0055595, -0.0005670, 0.0005665
9: -0.0035292, -0.0034231, -0.0035301, -0.0034235, -0.0000489, 0.0000489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001155, upper bound: 0.0001357
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001465
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083181, -0.0072085, -0.0083611, -0.0072786, -0.0005116, 0.0006314
1: -0.0052838, -0.0049710, -0.0052960, -0.0049908, -0.0001442, 0.0001780
2: -0.0004255, 0.0018826, -0.0005150, 0.0017370, -0.0010642, 0.0013134
3: 0.0015710, 0.0018764, 0.0015591, 0.0018572, -0.0001408, 0.0001738
4: 0.0046849, 0.0064098, 0.0047938, 0.0064768, -0.0009816, 0.0007953
5: 0.9968079, 0.9972871, 0.9968381, 0.9973057, -0.0002727, 0.0002210
6: 0.0049861, 0.0054211, 0.0050136, 0.0054380, -0.0002475, 0.0002006
7: -0.0047741, -0.0031508, -0.0046717, -0.0030878, -0.0009238, 0.0007485
8: -0.0067406, -0.0054772, -0.0067896, -0.0055569, -0.0005825, 0.0007190
9: -0.0035372, -0.0034282, -0.0035303, -0.0034240, -0.0000620, 0.0000503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001357
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001462
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083181, -0.0072085, -0.0083659, -0.0072808, -0.0004963, 0.0006230
1: -0.0052838, -0.0049710, -0.0052973, -0.0049914, -0.0001399, 0.0001756
2: -0.0004255, 0.0018826, -0.0005249, 0.0017322, -0.0010324, 0.0012959
3: 0.0015710, 0.0018764, 0.0015578, 0.0018565, -0.0001366, 0.0001715
4: 0.0046849, 0.0064098, 0.0047973, 0.0064841, -0.0009685, 0.0007715
5: 0.9968079, 0.9972871, 0.9968391, 0.9973077, -0.0002691, 0.0002144
6: 0.0049861, 0.0054211, 0.0050145, 0.0054399, -0.0002442, 0.0001946
7: -0.0047741, -0.0031508, -0.0046683, -0.0030809, -0.0009115, 0.0007261
8: -0.0067406, -0.0054772, -0.0067950, -0.0055595, -0.0005651, 0.0007094
9: -0.0035372, -0.0034282, -0.0035301, -0.0034235, -0.0000612, 0.0000488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001357
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001465
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083480, -0.0072816, -0.0084014, -0.0072884, -0.0004775, 0.0005707
1: -0.0052923, -0.0049916, -0.0053073, -0.0049935, -0.0001346, 0.0001609
2: -0.0004877, 0.0017307, -0.0005987, 0.0017165, -0.0009933, 0.0011872
3: 0.0015628, 0.0018563, 0.0015481, 0.0018544, -0.0001314, 0.0001571
4: 0.0047985, 0.0064563, 0.0048091, 0.0065393, -0.0008872, 0.0007423
5: 0.9968394, 0.9973000, 0.9968424, 0.9973230, -0.0002465, 0.0002062
6: 0.0050148, 0.0054329, 0.0050175, 0.0054538, -0.0002237, 0.0001872
7: -0.0046672, -0.0031070, -0.0046572, -0.0030289, -0.0008350, 0.0006986
8: -0.0067747, -0.0055603, -0.0068355, -0.0055681, -0.0005437, 0.0006499
9: -0.0035300, -0.0034253, -0.0035293, -0.0034200, -0.0000561, 0.0000469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001447
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001450
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0084014, -0.0072884, -0.0004856, 0.0005494
1: -0.0052982, -0.0049939, -0.0053073, -0.0049935, -0.0001369, 0.0001549
2: -0.0005318, 0.0017140, -0.0005987, 0.0017165, -0.0010102, 0.0011430
3: 0.0015569, 0.0018541, 0.0015481, 0.0018544, -0.0001337, 0.0001513
4: 0.0048109, 0.0064893, 0.0048091, 0.0065393, -0.0008542, 0.0007550
5: 0.9968429, 0.9973091, 0.9968424, 0.9973230, -0.0002373, 0.0002098
6: 0.0050179, 0.0054412, 0.0050175, 0.0054538, -0.0002154, 0.0001904
7: -0.0046555, -0.0030760, -0.0046572, -0.0030289, -0.0008039, 0.0007105
8: -0.0067988, -0.0055695, -0.0068355, -0.0055681, -0.0005530, 0.0006257
9: -0.0035292, -0.0034232, -0.0035293, -0.0034200, -0.0000540, 0.0000477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001515
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001516
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0082955, -0.0072006, -0.0084014, -0.0072884, -0.0004790, 0.0006981
1: -0.0052775, -0.0049688, -0.0053073, -0.0049935, -0.0001350, 0.0001968
2: -0.0003786, 0.0018991, -0.0005987, 0.0017165, -0.0009964, 0.0014522
3: 0.0015772, 0.0018786, 0.0015481, 0.0018544, -0.0001319, 0.0001922
4: 0.0046726, 0.0063748, 0.0048091, 0.0065393, -0.0010853, 0.0007446
5: 0.9968045, 0.9972773, 0.9968424, 0.9973230, -0.0003015, 0.0002069
6: 0.0049830, 0.0054123, 0.0050175, 0.0054538, -0.0002737, 0.0001878
7: -0.0047857, -0.0031838, -0.0046572, -0.0030289, -0.0010214, 0.0007008
8: -0.0067149, -0.0054682, -0.0068355, -0.0055681, -0.0005454, 0.0007949
9: -0.0035380, -0.0034304, -0.0035293, -0.0034200, -0.0000686, 0.0000471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001447
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001450
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0083166, -0.0072086, -0.0084014, -0.0072884, -0.0004844, 0.0006735
1: -0.0052834, -0.0049710, -0.0053073, -0.0049935, -0.0001366, 0.0001899
2: -0.0004223, 0.0018824, -0.0005987, 0.0017165, -0.0010076, 0.0014009
3: 0.0015714, 0.0018764, 0.0015481, 0.0018544, -0.0001333, 0.0001854
4: 0.0046850, 0.0064075, 0.0048091, 0.0065393, -0.0010470, 0.0007531
5: 0.9968079, 0.9972864, 0.9968424, 0.9973230, -0.0002909, 0.0002092
6: 0.0049862, 0.0054205, 0.0050175, 0.0054538, -0.0002640, 0.0001899
7: -0.0047740, -0.0031530, -0.0046572, -0.0030289, -0.0009853, 0.0007087
8: -0.0067389, -0.0054773, -0.0068355, -0.0055681, -0.0005516, 0.0007669
9: -0.0035372, -0.0034283, -0.0035293, -0.0034200, -0.0000662, 0.0000476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001511
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001513
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083726, -0.0072796, -0.0083609, -0.0072787, -0.0005225, 0.0005319
1: -0.0052992, -0.0049910, -0.0052959, -0.0049908, -0.0001473, 0.0001500
2: -0.0005389, 0.0017349, -0.0005146, 0.0017366, -0.0010869, 0.0011065
3: 0.0015560, 0.0018569, 0.0015592, 0.0018571, -0.0001438, 0.0001464
4: 0.0047953, 0.0064946, 0.0047940, 0.0064764, -0.0008270, 0.0008123
5: 0.9968385, 0.9973106, 0.9968382, 0.9973056, -0.0002298, 0.0002257
6: 0.0050140, 0.0054425, 0.0050137, 0.0054379, -0.0002085, 0.0002049
7: -0.0046702, -0.0030710, -0.0046714, -0.0030881, -0.0007783, 0.0007645
8: -0.0068027, -0.0055580, -0.0067894, -0.0055571, -0.0005950, 0.0006057
9: -0.0035302, -0.0034228, -0.0035303, -0.0034240, -0.0000523, 0.0000513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001142, upper bound: 0.0001369
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001484
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083726, -0.0072796, -0.0083653, -0.0072811, -0.0004999, 0.0005219
1: -0.0052992, -0.0049910, -0.0052971, -0.0049915, -0.0001409, 0.0001471
2: -0.0005389, 0.0017349, -0.0005236, 0.0017318, -0.0010399, 0.0010856
3: 0.0015560, 0.0018569, 0.0015580, 0.0018565, -0.0001376, 0.0001437
4: 0.0047953, 0.0064946, 0.0047976, 0.0064832, -0.0008113, 0.0007772
5: 0.9968385, 0.9973106, 0.9968392, 0.9973074, -0.0002254, 0.0002159
6: 0.0050140, 0.0054425, 0.0050146, 0.0054396, -0.0002046, 0.0001960
7: -0.0046702, -0.0030710, -0.0046680, -0.0030818, -0.0007635, 0.0007314
8: -0.0068027, -0.0055580, -0.0067943, -0.0055597, -0.0005692, 0.0005943
9: -0.0035302, -0.0034228, -0.0035301, -0.0034236, -0.0000513, 0.0000491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001142, upper bound: 0.0001369
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001485
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083191, -0.0071998, -0.0083609, -0.0072787, -0.0005135, 0.0006464
1: -0.0052841, -0.0049685, -0.0052959, -0.0049908, -0.0001448, 0.0001822
2: -0.0004275, 0.0019009, -0.0005146, 0.0017366, -0.0010681, 0.0013447
3: 0.0015707, 0.0018788, 0.0015592, 0.0018571, -0.0001414, 0.0001779
4: 0.0046713, 0.0064114, 0.0047940, 0.0064764, -0.0010049, 0.0007983
5: 0.9968041, 0.9972876, 0.9968382, 0.9973056, -0.0002792, 0.0002218
6: 0.0049827, 0.0054215, 0.0050137, 0.0054379, -0.0002534, 0.0002013
7: -0.0047870, -0.0031493, -0.0046714, -0.0030881, -0.0009457, 0.0007512
8: -0.0067417, -0.0054672, -0.0067894, -0.0055571, -0.0005847, 0.0007361
9: -0.0035381, -0.0034281, -0.0035303, -0.0034240, -0.0000635, 0.0000504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001370
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001484
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083191, -0.0071998, -0.0083653, -0.0072811, -0.0004983, 0.0006376
1: -0.0052841, -0.0049685, -0.0052971, -0.0049915, -0.0001405, 0.0001798
2: -0.0004275, 0.0019009, -0.0005236, 0.0017318, -0.0010365, 0.0013263
3: 0.0015707, 0.0018788, 0.0015580, 0.0018565, -0.0001372, 0.0001755
4: 0.0046713, 0.0064114, 0.0047976, 0.0064832, -0.0009912, 0.0007746
5: 0.9968041, 0.9972876, 0.9968392, 0.9973074, -0.0002754, 0.0002152
6: 0.0049827, 0.0054215, 0.0050146, 0.0054396, -0.0002500, 0.0001953
7: -0.0047870, -0.0031493, -0.0046680, -0.0030818, -0.0009329, 0.0007290
8: -0.0067417, -0.0054672, -0.0067943, -0.0055597, -0.0005674, 0.0007260
9: -0.0035381, -0.0034281, -0.0035301, -0.0034236, -0.0000626, 0.0000490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001370
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001485
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0083500, -0.0072718, -0.0084006, -0.0072886, -0.0004793, 0.0005933
1: -0.0052928, -0.0049889, -0.0053071, -0.0049936, -0.0001351, 0.0001673
2: -0.0004919, 0.0017510, -0.0005971, 0.0017160, -0.0009971, 0.0012342
3: 0.0015622, 0.0018590, 0.0015483, 0.0018544, -0.0001320, 0.0001633
4: 0.0047833, 0.0064595, 0.0048094, 0.0065381, -0.0009223, 0.0007452
5: 0.9968352, 0.9973009, 0.9968424, 0.9973227, -0.0002563, 0.0002070
6: 0.0050109, 0.0054337, 0.0050175, 0.0054535, -0.0002326, 0.0001879
7: -0.0046815, -0.0031041, -0.0046569, -0.0030301, -0.0008680, 0.0007013
8: -0.0067770, -0.0055492, -0.0068346, -0.0055684, -0.0005458, 0.0006756
9: -0.0035310, -0.0034251, -0.0035293, -0.0034201, -0.0000583, 0.0000471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001463
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001464
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083717, -0.0072796, -0.0084006, -0.0072886, -0.0004878, 0.0005709
1: -0.0052989, -0.0049911, -0.0053071, -0.0049936, -0.0001375, 0.0001610
2: -0.0005369, 0.0017347, -0.0005971, 0.0017160, -0.0010147, 0.0011877
3: 0.0015562, 0.0018569, 0.0015483, 0.0018544, -0.0001343, 0.0001572
4: 0.0047954, 0.0064931, 0.0048094, 0.0065381, -0.0008876, 0.0007583
5: 0.9968385, 0.9973102, 0.9968424, 0.9973227, -0.0002466, 0.0002107
6: 0.0050140, 0.0054421, 0.0050175, 0.0054535, -0.0002238, 0.0001912
7: -0.0046701, -0.0030724, -0.0046569, -0.0030301, -0.0008353, 0.0007136
8: -0.0068016, -0.0055581, -0.0068346, -0.0055684, -0.0005554, 0.0006501
9: -0.0035302, -0.0034229, -0.0035293, -0.0034201, -0.0000561, 0.0000479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001538
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001539
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0082969, -0.0071912, -0.0084006, -0.0072886, -0.0004810, 0.0007137
1: -0.0052779, -0.0049661, -0.0053071, -0.0049936, -0.0001356, 0.0002012
2: -0.0003814, 0.0019188, -0.0005971, 0.0017160, -0.0010006, 0.0014846
3: 0.0015768, 0.0018812, 0.0015483, 0.0018544, -0.0001324, 0.0001965
4: 0.0046579, 0.0063769, 0.0048094, 0.0065381, -0.0011095, 0.0007478
5: 0.9968003, 0.9972779, 0.9968424, 0.9973227, -0.0003082, 0.0002078
6: 0.0049793, 0.0054128, 0.0050175, 0.0054535, -0.0002798, 0.0001886
7: -0.0047995, -0.0031818, -0.0046569, -0.0030301, -0.0010441, 0.0007037
8: -0.0067165, -0.0054574, -0.0068346, -0.0055684, -0.0005477, 0.0008127
9: -0.0035389, -0.0034303, -0.0035293, -0.0034201, -0.0000701, 0.0000473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001463
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001464
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0083177, -0.0071999, -0.0084006, -0.0072886, -0.0004866, 0.0006884
1: -0.0052837, -0.0049686, -0.0053071, -0.0049936, -0.0001372, 0.0001941
2: -0.0004247, 0.0019007, -0.0005971, 0.0017160, -0.0010122, 0.0014320
3: 0.0015711, 0.0018788, 0.0015483, 0.0018544, -0.0001339, 0.0001895
4: 0.0046714, 0.0064092, 0.0048094, 0.0065381, -0.0010702, 0.0007564
5: 0.9968041, 0.9972869, 0.9968424, 0.9973227, -0.0002973, 0.0002102
6: 0.0049827, 0.0054210, 0.0050175, 0.0054535, -0.0002699, 0.0001908
7: -0.0047868, -0.0031513, -0.0046569, -0.0030301, -0.0010071, 0.0007119
8: -0.0067402, -0.0054673, -0.0068346, -0.0055684, -0.0005541, 0.0007839
9: -0.0035380, -0.0034282, -0.0035293, -0.0034201, -0.0000676, 0.0000478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001537
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001537
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083480, -0.0072816, -0.0083294, -0.0072052, -0.0006262, 0.0005252
1: -0.0052923, -0.0049916, -0.0052870, -0.0049701, -0.0001765, 0.0001481
2: -0.0004877, 0.0017307, -0.0004491, 0.0018895, -0.0013025, 0.0010926
3: 0.0015628, 0.0018563, 0.0015679, 0.0018773, -0.0001724, 0.0001446
4: 0.0047985, 0.0064563, 0.0046798, 0.0064275, -0.0008165, 0.0009734
5: 0.9968394, 0.9973000, 0.9968064, 0.9972920, -0.0002269, 0.0002704
6: 0.0050148, 0.0054329, 0.0049848, 0.0054256, -0.0002059, 0.0002455
7: -0.0046672, -0.0031070, -0.0047790, -0.0031342, -0.0007685, 0.0009161
8: -0.0067747, -0.0055603, -0.0067535, -0.0054734, -0.0007130, 0.0005981
9: -0.0035300, -0.0034253, -0.0035375, -0.0034271, -0.0000516, 0.0000615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001375, upper bound: 0.0001368
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001508
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083480, -0.0072816, -0.0083351, -0.0072077, -0.0006028, 0.0005125
1: -0.0052923, -0.0049916, -0.0052886, -0.0049708, -0.0001700, 0.0001445
2: -0.0004877, 0.0017307, -0.0004608, 0.0018843, -0.0012539, 0.0010661
3: 0.0015628, 0.0018563, 0.0015663, 0.0018767, -0.0001659, 0.0001411
4: 0.0047985, 0.0064563, 0.0046836, 0.0064363, -0.0007967, 0.0009371
5: 0.9968394, 0.9973000, 0.9968075, 0.9972945, -0.0002214, 0.0002604
6: 0.0050148, 0.0054329, 0.0049858, 0.0054278, -0.0002009, 0.0002363
7: -0.0046672, -0.0031070, -0.0047753, -0.0031259, -0.0007498, 0.0008819
8: -0.0067747, -0.0055603, -0.0067600, -0.0054762, -0.0006864, 0.0005836
9: -0.0035300, -0.0034253, -0.0035373, -0.0034265, -0.0000503, 0.0000592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001375, upper bound: 0.0001368
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001508
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083269, -0.0071995, -0.0006415, 0.0005368
1: -0.0052982, -0.0049939, -0.0052863, -0.0049685, -0.0001809, 0.0001513
2: -0.0005318, 0.0017140, -0.0004438, 0.0019013, -0.0013344, 0.0011166
3: 0.0015569, 0.0018541, 0.0015686, 0.0018789, -0.0001766, 0.0001478
4: 0.0048109, 0.0064893, 0.0046709, 0.0064235, -0.0008345, 0.0009972
5: 0.9968429, 0.9973091, 0.9968041, 0.9972908, -0.0002318, 0.0002771
6: 0.0050179, 0.0054412, 0.0049826, 0.0054246, -0.0002104, 0.0002515
7: -0.0046555, -0.0030760, -0.0047873, -0.0031379, -0.0007853, 0.0009385
8: -0.0067988, -0.0055695, -0.0067507, -0.0054669, -0.0007304, 0.0006112
9: -0.0035292, -0.0034232, -0.0035381, -0.0034273, -0.0000527, 0.0000630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001572
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001572
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083481, -0.0072075, -0.0006156, 0.0005421
1: -0.0052982, -0.0049939, -0.0052923, -0.0049707, -0.0001736, 0.0001528
2: -0.0005318, 0.0017140, -0.0004879, 0.0018848, -0.0012806, 0.0011276
3: 0.0015569, 0.0018541, 0.0015627, 0.0018767, -0.0001695, 0.0001492
4: 0.0048109, 0.0064893, 0.0046833, 0.0064565, -0.0008427, 0.0009570
5: 0.9968429, 0.9973091, 0.9968075, 0.9973000, -0.0002341, 0.0002659
6: 0.0050179, 0.0054412, 0.0049857, 0.0054329, -0.0002125, 0.0002413
7: -0.0046555, -0.0030760, -0.0047757, -0.0031068, -0.0007931, 0.0009007
8: -0.0067988, -0.0055695, -0.0067748, -0.0054760, -0.0007010, 0.0006173
9: -0.0035292, -0.0034232, -0.0035373, -0.0034252, -0.0000533, 0.0000605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001578
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001579
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0083181, -0.0072085, -0.0083074, -0.0071962, -0.0005246, 0.0005151
1: -0.0052838, -0.0049710, -0.0052808, -0.0049675, -0.0001479, 0.0001452
2: -0.0004255, 0.0018826, -0.0004033, 0.0019083, -0.0010912, 0.0010716
3: 0.0015710, 0.0018764, 0.0015739, 0.0018798, -0.0001444, 0.0001418
4: 0.0046849, 0.0064098, 0.0046658, 0.0063933, -0.0008008, 0.0008155
5: 0.9968079, 0.9972871, 0.9968026, 0.9972826, -0.0002225, 0.0002266
6: 0.0049861, 0.0054211, 0.0049813, 0.0054170, -0.0002020, 0.0002057
7: -0.0047741, -0.0031508, -0.0047921, -0.0031663, -0.0007537, 0.0007675
8: -0.0067406, -0.0054772, -0.0067285, -0.0054631, -0.0005973, 0.0005866
9: -0.0035372, -0.0034282, -0.0035384, -0.0034292, -0.0000506, 0.0000515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001514
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001516
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083181, -0.0072085, -0.0083126, -0.0071999, -0.0005024, 0.0005020
1: -0.0052838, -0.0049710, -0.0052823, -0.0049686, -0.0001416, 0.0001415
2: -0.0004255, 0.0018826, -0.0004141, 0.0019006, -0.0010451, 0.0010442
3: 0.0015710, 0.0018764, 0.0015725, 0.0018788, -0.0001383, 0.0001382
4: 0.0046849, 0.0064098, 0.0046715, 0.0064013, -0.0007804, 0.0007810
5: 0.9968079, 0.9972871, 0.9968042, 0.9972847, -0.0002168, 0.0002170
6: 0.0049861, 0.0054211, 0.0049827, 0.0054190, -0.0001968, 0.0001970
7: -0.0047741, -0.0031508, -0.0047868, -0.0031588, -0.0007344, 0.0007350
8: -0.0067406, -0.0054772, -0.0067344, -0.0054673, -0.0005721, 0.0005716
9: -0.0035372, -0.0034282, -0.0035380, -0.0034287, -0.0000493, 0.0000494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001514
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001516
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0082955, -0.0072006, -0.0083481, -0.0072075, -0.0004819, 0.0005754
1: -0.0052775, -0.0049688, -0.0052923, -0.0049707, -0.0001359, 0.0001622
2: -0.0003786, 0.0018991, -0.0004879, 0.0018848, -0.0010024, 0.0011969
3: 0.0015772, 0.0018786, 0.0015627, 0.0018767, -0.0001327, 0.0001584
4: 0.0046726, 0.0063748, 0.0046833, 0.0064565, -0.0008945, 0.0007492
5: 0.9968045, 0.9972773, 0.9968075, 0.9973000, -0.0002485, 0.0002081
6: 0.0049830, 0.0054123, 0.0049857, 0.0054329, -0.0002256, 0.0001889
7: -0.0047857, -0.0031838, -0.0047757, -0.0031068, -0.0008418, 0.0007050
8: -0.0067149, -0.0054682, -0.0067748, -0.0054760, -0.0005487, 0.0006552
9: -0.0035380, -0.0034304, -0.0035373, -0.0034252, -0.0000565, 0.0000473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001526
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001526
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083166, -0.0072086, -0.0083481, -0.0072075, -0.0004907, 0.0005538
1: -0.0052834, -0.0049710, -0.0052923, -0.0049707, -0.0001383, 0.0001561
2: -0.0004223, 0.0018824, -0.0004879, 0.0018848, -0.0010208, 0.0011520
3: 0.0015714, 0.0018764, 0.0015627, 0.0018767, -0.0001351, 0.0001524
4: 0.0046850, 0.0064075, 0.0046833, 0.0064565, -0.0008609, 0.0007629
5: 0.9968079, 0.9972864, 0.9968075, 0.9973000, -0.0002392, 0.0002119
6: 0.0049862, 0.0054205, 0.0049857, 0.0054329, -0.0002171, 0.0001924
7: -0.0047740, -0.0031530, -0.0047757, -0.0031068, -0.0008102, 0.0007179
8: -0.0067389, -0.0054773, -0.0067748, -0.0054760, -0.0005588, 0.0006306
9: -0.0035372, -0.0034283, -0.0035373, -0.0034252, -0.0000544, 0.0000482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001546
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001547
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083500, -0.0072718, -0.0083285, -0.0072054, -0.0006281, 0.0005453
1: -0.0052928, -0.0049889, -0.0052868, -0.0049701, -0.0001771, 0.0001537
2: -0.0004919, 0.0017510, -0.0004472, 0.0018891, -0.0013067, 0.0011343
3: 0.0015622, 0.0018590, 0.0015681, 0.0018773, -0.0001729, 0.0001501
4: 0.0047833, 0.0064595, 0.0046801, 0.0064261, -0.0008477, 0.0009765
5: 0.9968352, 0.9973009, 0.9968065, 0.9972916, -0.0002355, 0.0002713
6: 0.0050109, 0.0054337, 0.0049849, 0.0054252, -0.0002138, 0.0002463
7: -0.0046815, -0.0031041, -0.0047787, -0.0031355, -0.0007978, 0.0009190
8: -0.0067770, -0.0055492, -0.0067525, -0.0054736, -0.0007153, 0.0006209
9: -0.0035310, -0.0034251, -0.0035375, -0.0034272, -0.0000536, 0.0000617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001364
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001534
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083500, -0.0072718, -0.0083339, -0.0072079, -0.0006047, 0.0005372
1: -0.0052928, -0.0049889, -0.0052883, -0.0049708, -0.0001705, 0.0001514
2: -0.0004919, 0.0017510, -0.0004584, 0.0018839, -0.0012580, 0.0011174
3: 0.0015622, 0.0018590, 0.0015666, 0.0018766, -0.0001665, 0.0001479
4: 0.0047833, 0.0064595, 0.0046840, 0.0064345, -0.0008351, 0.0009401
5: 0.9968352, 0.9973009, 0.9968076, 0.9972939, -0.0002320, 0.0002612
6: 0.0050109, 0.0054337, 0.0049859, 0.0054273, -0.0002106, 0.0002371
7: -0.0046815, -0.0031041, -0.0047750, -0.0031276, -0.0007859, 0.0008848
8: -0.0067770, -0.0055492, -0.0067587, -0.0054765, -0.0006886, 0.0006117
9: -0.0035310, -0.0034251, -0.0035373, -0.0034266, -0.0000528, 0.0000594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001364
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001534
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083717, -0.0072796, -0.0083263, -0.0071998, -0.0006435, 0.0005540
1: -0.0052989, -0.0049911, -0.0052862, -0.0049685, -0.0001814, 0.0001562
2: -0.0005369, 0.0017347, -0.0004426, 0.0019009, -0.0013386, 0.0011524
3: 0.0015562, 0.0018569, 0.0015687, 0.0018788, -0.0001771, 0.0001525
4: 0.0047954, 0.0064931, 0.0046713, 0.0064226, -0.0008612, 0.0010004
5: 0.9968385, 0.9973102, 0.9968041, 0.9972907, -0.0002393, 0.0002779
6: 0.0050140, 0.0054421, 0.0049827, 0.0054244, -0.0002172, 0.0002523
7: -0.0046701, -0.0030724, -0.0047870, -0.0031387, -0.0008105, 0.0009415
8: -0.0068016, -0.0055581, -0.0067500, -0.0054672, -0.0007328, 0.0006308
9: -0.0035302, -0.0034229, -0.0035381, -0.0034274, -0.0000544, 0.0000632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001588
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001587
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083717, -0.0072796, -0.0083471, -0.0072077, -0.0006178, 0.0005604
1: -0.0052989, -0.0049911, -0.0052920, -0.0049708, -0.0001742, 0.0001580
2: -0.0005369, 0.0017347, -0.0004858, 0.0018843, -0.0012852, 0.0011657
3: 0.0015562, 0.0018569, 0.0015630, 0.0018767, -0.0001701, 0.0001543
4: 0.0047954, 0.0064931, 0.0046836, 0.0064549, -0.0008711, 0.0009605
5: 0.9968385, 0.9973102, 0.9968076, 0.9972996, -0.0002420, 0.0002668
6: 0.0050140, 0.0054421, 0.0049858, 0.0054325, -0.0002197, 0.0002422
7: -0.0046701, -0.0030724, -0.0047753, -0.0031083, -0.0008198, 0.0009039
8: -0.0068016, -0.0055581, -0.0067736, -0.0054762, -0.0007035, 0.0006381
9: -0.0035302, -0.0034229, -0.0035373, -0.0034253, -0.0000551, 0.0000607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001593
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001593
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0083191, -0.0071998, -0.0083067, -0.0071964, -0.0005251, 0.0005366
1: -0.0052841, -0.0049685, -0.0052806, -0.0049676, -0.0001480, 0.0001513
2: -0.0004275, 0.0019009, -0.0004018, 0.0019078, -0.0010923, 0.0011163
3: 0.0015707, 0.0018788, 0.0015741, 0.0018798, -0.0001446, 0.0001477
4: 0.0046713, 0.0064114, 0.0046661, 0.0063921, -0.0008342, 0.0008163
5: 0.9968041, 0.9972876, 0.9968026, 0.9972821, -0.0002318, 0.0002268
6: 0.0049827, 0.0054215, 0.0049814, 0.0054167, -0.0002104, 0.0002059
7: -0.0047870, -0.0031493, -0.0047919, -0.0031675, -0.0007851, 0.0007683
8: -0.0067417, -0.0054672, -0.0067276, -0.0054634, -0.0005979, 0.0006110
9: -0.0035381, -0.0034281, -0.0035384, -0.0034293, -0.0000527, 0.0000516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001544
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001545
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083191, -0.0071998, -0.0083117, -0.0072001, -0.0005024, 0.0005267
1: -0.0052841, -0.0049685, -0.0052820, -0.0049686, -0.0001416, 0.0001485
2: -0.0004275, 0.0019009, -0.0004122, 0.0019002, -0.0010450, 0.0010956
3: 0.0015707, 0.0018788, 0.0015727, 0.0018788, -0.0001383, 0.0001450
4: 0.0046713, 0.0064114, 0.0046718, 0.0064000, -0.0008188, 0.0007810
5: 0.9968041, 0.9972876, 0.9968042, 0.9972843, -0.0002275, 0.0002170
6: 0.0049827, 0.0054215, 0.0049828, 0.0054186, -0.0002065, 0.0001970
7: -0.0047870, -0.0031493, -0.0047864, -0.0031601, -0.0007706, 0.0007350
8: -0.0067417, -0.0054672, -0.0067334, -0.0054676, -0.0005721, 0.0005997
9: -0.0035381, -0.0034281, -0.0035380, -0.0034288, -0.0000517, 0.0000494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001545
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001546
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0082969, -0.0071912, -0.0083471, -0.0072077, -0.0004818, 0.0005980
1: -0.0052779, -0.0049661, -0.0052920, -0.0049708, -0.0001358, 0.0001686
2: -0.0003814, 0.0019188, -0.0004858, 0.0018843, -0.0010022, 0.0012439
3: 0.0015768, 0.0018812, 0.0015630, 0.0018767, -0.0001326, 0.0001646
4: 0.0046579, 0.0063769, 0.0046836, 0.0064549, -0.0009296, 0.0007490
5: 0.9968003, 0.9972779, 0.9968076, 0.9972996, -0.0002583, 0.0002081
6: 0.0049793, 0.0054128, 0.0049858, 0.0054325, -0.0002344, 0.0001889
7: -0.0047995, -0.0031818, -0.0047753, -0.0031083, -0.0008749, 0.0007049
8: -0.0067165, -0.0054574, -0.0067736, -0.0054762, -0.0005486, 0.0006809
9: -0.0035389, -0.0034303, -0.0035373, -0.0034253, -0.0000587, 0.0000473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001547
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001546
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083177, -0.0071999, -0.0083471, -0.0072077, -0.0004907, 0.0005754
1: -0.0052837, -0.0049686, -0.0052920, -0.0049708, -0.0001384, 0.0001622
2: -0.0004247, 0.0019007, -0.0004858, 0.0018843, -0.0010208, 0.0011970
3: 0.0015711, 0.0018788, 0.0015630, 0.0018767, -0.0001351, 0.0001584
4: 0.0046714, 0.0064092, 0.0046836, 0.0064549, -0.0008946, 0.0007629
5: 0.9968041, 0.9972869, 0.9968076, 0.9972996, -0.0002485, 0.0002119
6: 0.0049827, 0.0054210, 0.0049858, 0.0054325, -0.0002256, 0.0001924
7: -0.0047868, -0.0031513, -0.0047753, -0.0031083, -0.0008419, 0.0007179
8: -0.0067402, -0.0054673, -0.0067736, -0.0054762, -0.0005588, 0.0006552
9: -0.0035380, -0.0034282, -0.0035373, -0.0034253, -0.0000565, 0.0000482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001572
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001572
time: 0.62 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.84 seconds
IS_A1_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001425, upper bound: 0.0001449
IS_A1_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001425, upper bound: 0.0001449
IS_A1_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001518, upper bound: 0.0001449
IS_A1_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001518, upper bound: 0.0001449
IS_A1_B1_A1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001309, upper bound: 0.0001155
IS_A1_B1_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001405
IS_A1_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001151, upper bound: 0.0001439
IS_A1_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001477
IS_A1_B1_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001472, upper bound: 0.0001215
IS_A1_B1_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001534, upper bound: 0.0001405
IS_A1_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001366, upper bound: 0.0001439
IS_A1_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001534, upper bound: 0.0001472
IS_A1_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001471
IS_A1_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001471
IS_A1_B1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001471
IS_A1_B1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001471
IS_A1_B1_A2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001275, upper bound: 0.0001143
IS_A1_B1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001434
IS_A1_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001141, upper bound: 0.0001463
IS_A1_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001502
IS_A1_B1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001469, upper bound: 0.0001208
IS_A1_B1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001534, upper bound: 0.0001434
IS_A1_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001364, upper bound: 0.0001463
IS_A1_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001534, upper bound: 0.0001500
IS_A1_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001517
IS_A1_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001531
IS_A1_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001280, upper bound: 0.0001453
IS_A1_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001417, upper bound: 0.0001555
IS_A1_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001280, upper bound: 0.0001506
IS_A1_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001417, upper bound: 0.0001561
IS_A1_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001573, upper bound: 0.0001485
IS_A1_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001573, upper bound: 0.0001493
IS_A1_B2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001574, upper bound: 0.0001501
IS_A1_B2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001574, upper bound: 0.0001522
IS_A1_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001524, upper bound: 0.0001535
IS_A1_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001540, upper bound: 0.0001537
IS_A1_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001513
IS_A1_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001540, upper bound: 0.0001537
IS_A1_B2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001531
IS_A1_B2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001531
IS_A1_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001292, upper bound: 0.0001451
IS_A1_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001555
IS_A1_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001292, upper bound: 0.0001506
IS_A1_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001436, upper bound: 0.0001562
IS_A1_B2_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001584, upper bound: 0.0001493
IS_A1_B2_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001584, upper bound: 0.0001493
IS_A1_B2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001586, upper bound: 0.0001522
IS_A1_B2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001586, upper bound: 0.0001522
IS_A1_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001532, upper bound: 0.0001535
IS_A1_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001568, upper bound: 0.0001537
IS_A1_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001513
IS_A1_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001568, upper bound: 0.0001537
IS_A2_B1_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001155, upper bound: 0.0001357
IS_A2_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001462
IS_A2_B1_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001155, upper bound: 0.0001357
IS_A2_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001465
IS_A2_B1_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001357
IS_A2_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001462
IS_A2_B1_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001357
IS_A2_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001465
IS_A2_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001447
IS_A2_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001450
IS_A2_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001515
IS_A2_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001516
IS_A2_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001447
IS_A2_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001450
IS_A2_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001511
IS_A2_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001513
IS_A2_B1_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001142, upper bound: 0.0001369
IS_A2_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001484
IS_A2_B1_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001142, upper bound: 0.0001369
IS_A2_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001485
IS_A2_B1_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001370
IS_A2_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001484
IS_A2_B1_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001370
IS_A2_B1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001531, upper bound: 0.0001485
IS_A2_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001463
IS_A2_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001464
IS_A2_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001538
IS_A2_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001539
IS_A2_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001463
IS_A2_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001464
IS_A2_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001537
IS_A2_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001556, upper bound: 0.0001537
IS_A2_B2_A1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001375, upper bound: 0.0001368
IS_A2_B2_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001508
IS_A2_B2_A1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001375, upper bound: 0.0001368
IS_A2_B2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001508
IS_A2_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001572
IS_A2_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001572
IS_A2_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001578
IS_A2_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001579
IS_A2_B2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001514
IS_A2_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001516
IS_A2_B2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001514
IS_A2_B2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001516
IS_A2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001526
IS_A2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001526
IS_A2_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001546
IS_A2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001591, upper bound: 0.0001547
IS_A2_B2_A2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001364
IS_A2_B2_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001534
IS_A2_B2_A2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001362, upper bound: 0.0001364
IS_A2_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001534
IS_A2_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001588
IS_A2_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001587
IS_A2_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001593
IS_A2_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001593
IS_A2_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001544
IS_A2_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001545
IS_A2_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001545
IS_A2_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001546
IS_A2_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001547
IS_A2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001546
IS_A2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001572
IS_A2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001572

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0083630, -0.0072886, -0.0083511, -0.0072820, -0.0005073, 0.0004794
1: -0.0052965, -0.0049936, -0.0052931, -0.0049917, -0.0001430, 0.0001352
2: -0.0005190, 0.0017160, -0.0004941, 0.0017299, -0.0010553, 0.0009973
3: 0.0015586, 0.0018544, 0.0015619, 0.0018562, -0.0001396, 0.0001320
4: 0.0048095, 0.0064797, 0.0047990, 0.0064612, -0.0007453, 0.0007887
5: 0.9968425, 0.9973065, 0.9968395, 0.9973013, -0.0002071, 0.0002191
6: 0.0050175, 0.0054388, 0.0050149, 0.0054341, -0.0001880, 0.0001989
7: -0.0046569, -0.0030850, -0.0046667, -0.0031025, -0.0007014, 0.0007422
8: -0.0067918, -0.0055684, -0.0067782, -0.0055608, -0.0005777, 0.0005459
9: -0.0035293, -0.0034238, -0.0035300, -0.0034249, -0.0000471, 0.0000498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001449
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001449
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083630, -0.0072886, -0.0083543, -0.0072722, -0.0005279, 0.0004966
1: -0.0052965, -0.0049936, -0.0052941, -0.0049890, -0.0001488, 0.0001400
2: -0.0005190, 0.0017160, -0.0005009, 0.0017503, -0.0010982, 0.0010331
3: 0.0015586, 0.0018544, 0.0015610, 0.0018589, -0.0001453, 0.0001367
4: 0.0048095, 0.0064797, 0.0047838, 0.0064662, -0.0007721, 0.0008207
5: 0.9968425, 0.9973065, 0.9968354, 0.9973028, -0.0002145, 0.0002280
6: 0.0050175, 0.0054388, 0.0050111, 0.0054353, -0.0001947, 0.0002070
7: -0.0046569, -0.0030850, -0.0046810, -0.0030977, -0.0007266, 0.0007724
8: -0.0067918, -0.0055684, -0.0067819, -0.0055496, -0.0006011, 0.0005655
9: -0.0035293, -0.0034238, -0.0035309, -0.0034246, -0.0000488, 0.0000519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001449
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001449
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0083106, -0.0072062, -0.0083511, -0.0072820, -0.0005055, 0.0006053
1: -0.0052817, -0.0049703, -0.0052931, -0.0049917, -0.0001425, 0.0001707
2: -0.0004099, 0.0018875, -0.0004941, 0.0017299, -0.0010516, 0.0012592
3: 0.0015731, 0.0018771, 0.0015619, 0.0018562, -0.0001392, 0.0001666
4: 0.0046812, 0.0063982, 0.0047990, 0.0064612, -0.0009410, 0.0007859
5: 0.9968069, 0.9972838, 0.9968395, 0.9973013, -0.0002614, 0.0002184
6: 0.0049852, 0.0054182, 0.0050149, 0.0054341, -0.0002373, 0.0001982
7: -0.0047776, -0.0031617, -0.0046667, -0.0031025, -0.0008856, 0.0007396
8: -0.0067321, -0.0054745, -0.0067782, -0.0055608, -0.0005757, 0.0006893
9: -0.0035374, -0.0034289, -0.0035300, -0.0034249, -0.0000595, 0.0000497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001449
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001449
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083106, -0.0072062, -0.0083543, -0.0072722, -0.0005262, 0.0006225
1: -0.0052817, -0.0049703, -0.0052941, -0.0049890, -0.0001483, 0.0001755
2: -0.0004099, 0.0018875, -0.0005009, 0.0017503, -0.0010945, 0.0012949
3: 0.0015731, 0.0018771, 0.0015610, 0.0018589, -0.0001448, 0.0001714
4: 0.0046812, 0.0063982, 0.0047838, 0.0064662, -0.0009678, 0.0008180
5: 0.9968069, 0.9972838, 0.9968354, 0.9973028, -0.0002689, 0.0002273
6: 0.0049852, 0.0054182, 0.0050111, 0.0054353, -0.0002441, 0.0002063
7: -0.0047776, -0.0031617, -0.0046810, -0.0030977, -0.0009108, 0.0007698
8: -0.0067321, -0.0054745, -0.0067819, -0.0055496, -0.0005991, 0.0007089
9: -0.0035374, -0.0034289, -0.0035309, -0.0034246, -0.0000612, 0.0000517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001449
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001449
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083340, -0.0072799, -0.0083967, -0.0072886, -0.0004769, 0.0005315
1: -0.0052883, -0.0049911, -0.0053060, -0.0049936, -0.0001345, 0.0001498
2: -0.0004585, 0.0017342, -0.0005891, 0.0017161, -0.0009921, 0.0011055
3: 0.0015666, 0.0018568, 0.0015493, 0.0018544, -0.0001313, 0.0001463
4: 0.0047958, 0.0064345, 0.0048093, 0.0065321, -0.0008262, 0.0007415
5: 0.9968386, 0.9972939, 0.9968424, 0.9973210, -0.0002295, 0.0002060
6: 0.0050141, 0.0054274, 0.0050175, 0.0054520, -0.0002084, 0.0001870
7: -0.0046698, -0.0031275, -0.0046570, -0.0030357, -0.0007776, 0.0006978
8: -0.0067587, -0.0055584, -0.0068302, -0.0055683, -0.0005431, 0.0006052
9: -0.0035302, -0.0034266, -0.0035293, -0.0034205, -0.0000522, 0.0000469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001405
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001405
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083574, -0.0072890, -0.0083799, -0.0072879, -0.0004954, 0.0005039
1: -0.0052949, -0.0049937, -0.0053013, -0.0049934, -0.0001397, 0.0001421
2: -0.0005073, 0.0017153, -0.0005540, 0.0017176, -0.0010306, 0.0010483
3: 0.0015602, 0.0018543, 0.0015540, 0.0018546, -0.0001364, 0.0001387
4: 0.0048100, 0.0064710, 0.0048083, 0.0065059, -0.0007834, 0.0007702
5: 0.9968426, 0.9973041, 0.9968421, 0.9973138, -0.0002177, 0.0002140
6: 0.0050177, 0.0054366, 0.0050172, 0.0054454, -0.0001976, 0.0001942
7: -0.0046564, -0.0030932, -0.0046580, -0.0030604, -0.0007373, 0.0007249
8: -0.0067854, -0.0055688, -0.0068110, -0.0055675, -0.0005642, 0.0005738
9: -0.0035293, -0.0034243, -0.0035294, -0.0034221, -0.0000495, 0.0000487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001411, upper bound: 0.0001439
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001411, upper bound: 0.0001439
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083613, -0.0072887, -0.0083895, -0.0072890, -0.0004977, 0.0005066
1: -0.0052960, -0.0049936, -0.0053040, -0.0049937, -0.0001403, 0.0001428
2: -0.0005154, 0.0017158, -0.0005741, 0.0017152, -0.0010354, 0.0010538
3: 0.0015591, 0.0018544, 0.0015513, 0.0018543, -0.0001370, 0.0001395
4: 0.0048096, 0.0064770, 0.0048101, 0.0065209, -0.0007875, 0.0007738
5: 0.9968424, 0.9973058, 0.9968426, 0.9973179, -0.0002188, 0.0002150
6: 0.0050176, 0.0054381, 0.0050177, 0.0054491, -0.0001986, 0.0001951
7: -0.0046568, -0.0030875, -0.0046563, -0.0030462, -0.0007412, 0.0007282
8: -0.0067898, -0.0055685, -0.0068220, -0.0055688, -0.0005668, 0.0005769
9: -0.0035293, -0.0034239, -0.0035293, -0.0034212, -0.0000498, 0.0000489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001472, upper bound: 0.0001477
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001472, upper bound: 0.0001477
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0082702, -0.0071973, -0.0083926, -0.0072889, -0.0004731, 0.0006627
1: -0.0052703, -0.0049678, -0.0053048, -0.0049937, -0.0001334, 0.0001868
2: -0.0003258, 0.0019060, -0.0005804, 0.0017155, -0.0009842, 0.0013785
3: 0.0015842, 0.0018795, 0.0015505, 0.0018543, -0.0001302, 0.0001824
4: 0.0046674, 0.0063353, 0.0048098, 0.0065256, -0.0010302, 0.0007355
5: 0.9968030, 0.9972664, 0.9968426, 0.9973192, -0.0002862, 0.0002043
6: 0.0049817, 0.0054023, 0.0050176, 0.0054503, -0.0002598, 0.0001855
7: -0.0047906, -0.0032209, -0.0046566, -0.0030418, -0.0009695, 0.0006922
8: -0.0066860, -0.0054644, -0.0068254, -0.0055686, -0.0005387, 0.0007546
9: -0.0035383, -0.0034329, -0.0035293, -0.0034209, -0.0000651, 0.0000465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001526, upper bound: 0.0001215
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001526, upper bound: 0.0001215
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0082816, -0.0071972, -0.0083967, -0.0072886, -0.0004788, 0.0006669
1: -0.0052735, -0.0049678, -0.0053060, -0.0049936, -0.0001350, 0.0001880
2: -0.0003495, 0.0019061, -0.0005891, 0.0017161, -0.0009959, 0.0013873
3: 0.0015810, 0.0018795, 0.0015493, 0.0018544, -0.0001318, 0.0001836
4: 0.0046673, 0.0063530, 0.0048093, 0.0065321, -0.0010368, 0.0007443
5: 0.9968030, 0.9972713, 0.9968424, 0.9973210, -0.0002881, 0.0002068
6: 0.0049817, 0.0054068, 0.0050175, 0.0054520, -0.0002615, 0.0001877
7: -0.0047906, -0.0032042, -0.0046570, -0.0030357, -0.0009758, 0.0007005
8: -0.0066990, -0.0054643, -0.0068302, -0.0055683, -0.0005452, 0.0007594
9: -0.0035383, -0.0034318, -0.0035293, -0.0034205, -0.0000655, 0.0000470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001547, upper bound: 0.0001405
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001547, upper bound: 0.0001405
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083046, -0.0072065, -0.0083799, -0.0072879, -0.0004946, 0.0006344
1: -0.0052800, -0.0049704, -0.0053013, -0.0049934, -0.0001394, 0.0001788
2: -0.0003973, 0.0018868, -0.0005540, 0.0017176, -0.0010289, 0.0013196
3: 0.0015747, 0.0018770, 0.0015540, 0.0018546, -0.0001362, 0.0001746
4: 0.0046818, 0.0063888, 0.0048083, 0.0065059, -0.0009862, 0.0007689
5: 0.9968070, 0.9972813, 0.9968421, 0.9973138, -0.0002740, 0.0002136
6: 0.0049853, 0.0054158, 0.0050172, 0.0054454, -0.0002487, 0.0001939
7: -0.0047771, -0.0031706, -0.0046580, -0.0030604, -0.0009281, 0.0007236
8: -0.0067252, -0.0054749, -0.0068110, -0.0055675, -0.0005632, 0.0007223
9: -0.0035374, -0.0034295, -0.0035294, -0.0034221, -0.0000623, 0.0000486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001439
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001439
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083089, -0.0072062, -0.0083895, -0.0072890, -0.0004964, 0.0006390
1: -0.0052812, -0.0049704, -0.0053040, -0.0049937, -0.0001400, 0.0001801
2: -0.0004063, 0.0018874, -0.0005741, 0.0017152, -0.0010326, 0.0013292
3: 0.0015735, 0.0018771, 0.0015513, 0.0018543, -0.0001366, 0.0001759
4: 0.0046814, 0.0063955, 0.0048101, 0.0065209, -0.0009933, 0.0007717
5: 0.9968069, 0.9972832, 0.9968426, 0.9973179, -0.0002760, 0.0002144
6: 0.0049852, 0.0054175, 0.0050177, 0.0054491, -0.0002505, 0.0001946
7: -0.0047775, -0.0031642, -0.0046563, -0.0030462, -0.0009348, 0.0007263
8: -0.0067301, -0.0054746, -0.0068220, -0.0055688, -0.0005652, 0.0007276
9: -0.0035374, -0.0034291, -0.0035293, -0.0034212, -0.0000628, 0.0000488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001529, upper bound: 0.0001473
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001529, upper bound: 0.0001472
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0083665, -0.0072786, -0.0083524, -0.0072794, -0.0004808, 0.0005001
1: -0.0052975, -0.0049908, -0.0052935, -0.0049910, -0.0001356, 0.0001410
2: -0.0005263, 0.0017368, -0.0004969, 0.0017352, -0.0010002, 0.0010403
3: 0.0015577, 0.0018571, 0.0015615, 0.0018569, -0.0001324, 0.0001377
4: 0.0047939, 0.0064852, 0.0047950, 0.0064632, -0.0007775, 0.0007475
5: 0.9968381, 0.9973080, 0.9968385, 0.9973019, -0.0002160, 0.0002077
6: 0.0050136, 0.0054401, 0.0050139, 0.0054346, -0.0001961, 0.0001885
7: -0.0046716, -0.0030799, -0.0046705, -0.0031005, -0.0007317, 0.0007035
8: -0.0067958, -0.0055570, -0.0067797, -0.0055578, -0.0005475, 0.0005695
9: -0.0035303, -0.0034234, -0.0035302, -0.0034248, -0.0000491, 0.0000472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001471
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001471
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083665, -0.0072786, -0.0083571, -0.0072817, -0.0005106, 0.0005243
1: -0.0052975, -0.0049908, -0.0052948, -0.0049916, -0.0001439, 0.0001478
2: -0.0005263, 0.0017368, -0.0005065, 0.0017304, -0.0010621, 0.0010907
3: 0.0015577, 0.0018571, 0.0015603, 0.0018563, -0.0001405, 0.0001443
4: 0.0047939, 0.0064852, 0.0047987, 0.0064704, -0.0008151, 0.0007937
5: 0.9968381, 0.9973080, 0.9968395, 0.9973039, -0.0002265, 0.0002205
6: 0.0050136, 0.0054401, 0.0050148, 0.0054364, -0.0002056, 0.0002002
7: -0.0046716, -0.0030799, -0.0046670, -0.0030938, -0.0007671, 0.0007470
8: -0.0067958, -0.0055570, -0.0067850, -0.0055605, -0.0005814, 0.0005971
9: -0.0035303, -0.0034234, -0.0035300, -0.0034244, -0.0000515, 0.0000502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001471
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001471
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0083122, -0.0071964, -0.0083524, -0.0072794, -0.0004830, 0.0006178
1: -0.0052822, -0.0049676, -0.0052935, -0.0049910, -0.0001362, 0.0001742
2: -0.0004133, 0.0019079, -0.0004969, 0.0017352, -0.0010047, 0.0012851
3: 0.0015726, 0.0018798, 0.0015615, 0.0018569, -0.0001330, 0.0001701
4: 0.0046660, 0.0064007, 0.0047950, 0.0064632, -0.0009604, 0.0007508
5: 0.9968026, 0.9972845, 0.9968385, 0.9973019, -0.0002668, 0.0002086
6: 0.0049814, 0.0054188, 0.0050139, 0.0054346, -0.0002422, 0.0001893
7: -0.0047919, -0.0031594, -0.0046705, -0.0031005, -0.0009039, 0.0007066
8: -0.0067339, -0.0054633, -0.0067797, -0.0055578, -0.0005500, 0.0007035
9: -0.0035384, -0.0034288, -0.0035302, -0.0034248, -0.0000607, 0.0000474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001471
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001471
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083122, -0.0071964, -0.0083571, -0.0072817, -0.0005089, 0.0006420
1: -0.0052822, -0.0049676, -0.0052948, -0.0049916, -0.0001435, 0.0001810
2: -0.0004133, 0.0019079, -0.0005065, 0.0017304, -0.0010586, 0.0013355
3: 0.0015726, 0.0018798, 0.0015603, 0.0018563, -0.0001401, 0.0001767
4: 0.0046660, 0.0064007, 0.0047987, 0.0064704, -0.0009981, 0.0007911
5: 0.9968026, 0.9972845, 0.9968395, 0.9973039, -0.0002773, 0.0002198
6: 0.0049814, 0.0054188, 0.0050148, 0.0054364, -0.0002517, 0.0001995
7: -0.0047919, -0.0031594, -0.0046670, -0.0030938, -0.0009393, 0.0007445
8: -0.0067339, -0.0054633, -0.0067850, -0.0055605, -0.0005795, 0.0007311
9: -0.0035384, -0.0034288, -0.0035300, -0.0034244, -0.0000631, 0.0000500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001471
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001471
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0083383, -0.0072680, -0.0083961, -0.0072888, -0.0004797, 0.0005579
1: -0.0052895, -0.0049878, -0.0053058, -0.0049936, -0.0001352, 0.0001573
2: -0.0004676, 0.0017589, -0.0005877, 0.0017157, -0.0009978, 0.0011605
3: 0.0015654, 0.0018601, 0.0015495, 0.0018543, -0.0001320, 0.0001536
4: 0.0047774, 0.0064413, 0.0048097, 0.0065310, -0.0008673, 0.0007457
5: 0.9968335, 0.9972957, 0.9968425, 0.9973208, -0.0002410, 0.0002072
6: 0.0050095, 0.0054291, 0.0050176, 0.0054517, -0.0002187, 0.0001881
7: -0.0046871, -0.0031211, -0.0046567, -0.0030367, -0.0008162, 0.0007018
8: -0.0067637, -0.0055449, -0.0068294, -0.0055686, -0.0005462, 0.0006353
9: -0.0035313, -0.0034262, -0.0035293, -0.0034205, -0.0000548, 0.0000471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001434
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001434
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083595, -0.0072790, -0.0083793, -0.0072881, -0.0004973, 0.0005310
1: -0.0052955, -0.0049909, -0.0053011, -0.0049934, -0.0001402, 0.0001497
2: -0.0005115, 0.0017361, -0.0005529, 0.0017172, -0.0010344, 0.0011046
3: 0.0015596, 0.0018570, 0.0015541, 0.0018545, -0.0001369, 0.0001462
4: 0.0047944, 0.0064742, 0.0048086, 0.0065051, -0.0008255, 0.0007730
5: 0.9968383, 0.9973049, 0.9968423, 0.9973136, -0.0002293, 0.0002148
6: 0.0050137, 0.0054374, 0.0050173, 0.0054451, -0.0002082, 0.0001949
7: -0.0046711, -0.0030902, -0.0046578, -0.0030612, -0.0007769, 0.0007275
8: -0.0067877, -0.0055574, -0.0068104, -0.0055677, -0.0005662, 0.0006046
9: -0.0035303, -0.0034241, -0.0035294, -0.0034222, -0.0000522, 0.0000489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001391, upper bound: 0.0001463
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001391, upper bound: 0.0001463
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083640, -0.0072787, -0.0083888, -0.0072893, -0.0004994, 0.0005322
1: -0.0052968, -0.0049908, -0.0053038, -0.0049938, -0.0001408, 0.0001501
2: -0.0005209, 0.0017367, -0.0005725, 0.0017147, -0.0010388, 0.0011071
3: 0.0015584, 0.0018571, 0.0015515, 0.0018542, -0.0001375, 0.0001465
4: 0.0047940, 0.0064812, 0.0048104, 0.0065197, -0.0008274, 0.0007763
5: 0.9968382, 0.9973069, 0.9968427, 0.9973176, -0.0002299, 0.0002157
6: 0.0050136, 0.0054391, 0.0050178, 0.0054488, -0.0002087, 0.0001958
7: -0.0046715, -0.0030836, -0.0046560, -0.0030474, -0.0007787, 0.0007306
8: -0.0067929, -0.0055570, -0.0068211, -0.0055691, -0.0005686, 0.0006060
9: -0.0035303, -0.0034237, -0.0035293, -0.0034212, -0.0000523, 0.0000491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001502
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001503
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0082724, -0.0071876, -0.0083920, -0.0072891, -0.0004752, 0.0006741
1: -0.0052710, -0.0049651, -0.0053047, -0.0049937, -0.0001340, 0.0001901
2: -0.0003304, 0.0019261, -0.0005792, 0.0017151, -0.0009884, 0.0014024
3: 0.0015836, 0.0018822, 0.0015506, 0.0018543, -0.0001308, 0.0001856
4: 0.0046524, 0.0063388, 0.0048101, 0.0065247, -0.0010480, 0.0007387
5: 0.9967988, 0.9972673, 0.9968426, 0.9973189, -0.0002912, 0.0002052
6: 0.0049779, 0.0054032, 0.0050177, 0.0054501, -0.0002643, 0.0001863
7: -0.0048047, -0.0032176, -0.0046563, -0.0030426, -0.0009863, 0.0006952
8: -0.0066886, -0.0054534, -0.0068248, -0.0055689, -0.0005411, 0.0007677
9: -0.0035392, -0.0034327, -0.0035293, -0.0034209, -0.0000662, 0.0000467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001521, upper bound: 0.0001208
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001521, upper bound: 0.0001208
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0082835, -0.0071878, -0.0083961, -0.0072888, -0.0004816, 0.0006823
1: -0.0052741, -0.0049652, -0.0053058, -0.0049936, -0.0001358, 0.0001924
2: -0.0003535, 0.0019258, -0.0005877, 0.0017157, -0.0010019, 0.0014193
3: 0.0015805, 0.0018821, 0.0015495, 0.0018543, -0.0001326, 0.0001878
4: 0.0046527, 0.0063560, 0.0048097, 0.0065310, -0.0010607, 0.0007487
5: 0.9967989, 0.9972721, 0.9968425, 0.9973208, -0.0002947, 0.0002080
6: 0.0049780, 0.0054076, 0.0050176, 0.0054517, -0.0002675, 0.0001888
7: -0.0048045, -0.0032014, -0.0046567, -0.0030367, -0.0009983, 0.0007046
8: -0.0067012, -0.0054535, -0.0068294, -0.0055686, -0.0005484, 0.0007769
9: -0.0035392, -0.0034316, -0.0035293, -0.0034205, -0.0000670, 0.0000473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001434
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001434
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083064, -0.0071967, -0.0083793, -0.0072881, -0.0004964, 0.0006498
1: -0.0052805, -0.0049677, -0.0053011, -0.0049934, -0.0001400, 0.0001832
2: -0.0004011, 0.0019072, -0.0005529, 0.0017172, -0.0010327, 0.0013517
3: 0.0015742, 0.0018797, 0.0015541, 0.0018545, -0.0001367, 0.0001789
4: 0.0046665, 0.0063916, 0.0048086, 0.0065051, -0.0010102, 0.0007718
5: 0.9968027, 0.9972820, 0.9968423, 0.9973136, -0.0002807, 0.0002144
6: 0.0049815, 0.0054165, 0.0050173, 0.0054451, -0.0002547, 0.0001946
7: -0.0047914, -0.0031679, -0.0046578, -0.0030612, -0.0009507, 0.0007263
8: -0.0067273, -0.0054637, -0.0068104, -0.0055677, -0.0005653, 0.0007399
9: -0.0035384, -0.0034293, -0.0035294, -0.0034222, -0.0000638, 0.0000488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001463
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001463
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083110, -0.0071965, -0.0083888, -0.0072893, -0.0004981, 0.0006521
1: -0.0052818, -0.0049676, -0.0053038, -0.0049938, -0.0001404, 0.0001838
2: -0.0004106, 0.0019078, -0.0005725, 0.0017147, -0.0010362, 0.0013565
3: 0.0015730, 0.0018798, 0.0015515, 0.0018542, -0.0001371, 0.0001795
4: 0.0046661, 0.0063987, 0.0048104, 0.0065197, -0.0010138, 0.0007744
5: 0.9968026, 0.9972840, 0.9968427, 0.9973176, -0.0002817, 0.0002151
6: 0.0049814, 0.0054183, 0.0050178, 0.0054488, -0.0002557, 0.0001953
7: -0.0047918, -0.0031612, -0.0046560, -0.0030474, -0.0009541, 0.0007288
8: -0.0067325, -0.0054634, -0.0068211, -0.0055691, -0.0005672, 0.0007425
9: -0.0035384, -0.0034289, -0.0035293, -0.0034212, -0.0000641, 0.0000489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001542, upper bound: 0.0001500
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001542, upper bound: 0.0001501
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0083340, -0.0072799, -0.0083279, -0.0072084, -0.0005944, 0.0005007
1: -0.0052883, -0.0049911, -0.0052866, -0.0049710, -0.0001676, 0.0001412
2: -0.0004585, 0.0017342, -0.0004459, 0.0018830, -0.0012364, 0.0010417
3: 0.0015666, 0.0018568, 0.0015683, 0.0018765, -0.0001636, 0.0001378
4: 0.0047958, 0.0064345, 0.0046847, 0.0064251, -0.0007785, 0.0009240
5: 0.9968386, 0.9972939, 0.9968079, 0.9972913, -0.0002163, 0.0002567
6: 0.0050141, 0.0054274, 0.0049861, 0.0054250, -0.0001963, 0.0002330
7: -0.0046698, -0.0031275, -0.0047743, -0.0031364, -0.0007326, 0.0008696
8: -0.0067587, -0.0055584, -0.0067518, -0.0054770, -0.0006768, 0.0005702
9: -0.0035302, -0.0034266, -0.0035372, -0.0034272, -0.0000492, 0.0000584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001517
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001517
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0083383, -0.0072680, -0.0083279, -0.0072084, -0.0006134, 0.0005253
1: -0.0052895, -0.0049878, -0.0052866, -0.0049710, -0.0001729, 0.0001481
2: -0.0004676, 0.0017589, -0.0004459, 0.0018830, -0.0012759, 0.0010927
3: 0.0015654, 0.0018601, 0.0015683, 0.0018765, -0.0001689, 0.0001446
4: 0.0047774, 0.0064413, 0.0046847, 0.0064251, -0.0008166, 0.0009536
5: 0.9968335, 0.9972957, 0.9968079, 0.9972913, -0.0002269, 0.0002649
6: 0.0050095, 0.0054291, 0.0049861, 0.0054250, -0.0002059, 0.0002405
7: -0.0046871, -0.0031211, -0.0047743, -0.0031364, -0.0007685, 0.0008974
8: -0.0067637, -0.0055449, -0.0067518, -0.0054770, -0.0006985, 0.0005981
9: -0.0035313, -0.0034262, -0.0035372, -0.0034272, -0.0000516, 0.0000603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001531
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001531
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083629, -0.0072859, -0.0083010, -0.0072007, -0.0006476, 0.0004930
1: -0.0052965, -0.0049928, -0.0052790, -0.0049688, -0.0001826, 0.0001390
2: -0.0005186, 0.0017216, -0.0003899, 0.0018989, -0.0013472, 0.0010256
3: 0.0015587, 0.0018551, 0.0015757, 0.0018786, -0.0001783, 0.0001357
4: 0.0048052, 0.0064795, 0.0046728, 0.0063832, -0.0007665, 0.0010068
5: 0.9968413, 0.9973063, 0.9968045, 0.9972796, -0.0002129, 0.0002797
6: 0.0050165, 0.0054387, 0.0049831, 0.0054144, -0.0001933, 0.0002539
7: -0.0046609, -0.0030852, -0.0047855, -0.0031758, -0.0007213, 0.0009475
8: -0.0067916, -0.0055653, -0.0067211, -0.0054683, -0.0007374, 0.0005614
9: -0.0035296, -0.0034238, -0.0035380, -0.0034299, -0.0000484, 0.0000636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001280, upper bound: 0.0001453
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001280, upper bound: 0.0001453
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083719, -0.0072884, -0.0083053, -0.0072005, -0.0006514, 0.0004948
1: -0.0052990, -0.0049935, -0.0052802, -0.0049687, -0.0001836, 0.0001395
2: -0.0005374, 0.0017165, -0.0003989, 0.0018994, -0.0013550, 0.0010293
3: 0.0015562, 0.0018545, 0.0015745, 0.0018787, -0.0001793, 0.0001362
4: 0.0048090, 0.0064935, 0.0046724, 0.0063900, -0.0007692, 0.0010126
5: 0.9968423, 0.9973103, 0.9968044, 0.9972816, -0.0002137, 0.0002813
6: 0.0050174, 0.0054422, 0.0049830, 0.0054161, -0.0001940, 0.0002554
7: -0.0046573, -0.0030721, -0.0047859, -0.0031695, -0.0007239, 0.0009530
8: -0.0068019, -0.0055681, -0.0067261, -0.0054680, -0.0007417, 0.0005634
9: -0.0035293, -0.0034229, -0.0035380, -0.0034294, -0.0000486, 0.0000640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001417, upper bound: 0.0001539
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001417, upper bound: 0.0001555
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083629, -0.0072859, -0.0083225, -0.0072087, -0.0006231, 0.0004968
1: -0.0052965, -0.0049928, -0.0052851, -0.0049711, -0.0001757, 0.0001401
2: -0.0005186, 0.0017216, -0.0004347, 0.0018823, -0.0012961, 0.0010334
3: 0.0015587, 0.0018551, 0.0015698, 0.0018764, -0.0001715, 0.0001367
4: 0.0048052, 0.0064795, 0.0046852, 0.0064167, -0.0007723, 0.0009686
5: 0.9968413, 0.9973063, 0.9968079, 0.9972889, -0.0002146, 0.0002691
6: 0.0050165, 0.0054387, 0.0049862, 0.0054229, -0.0001948, 0.0002443
7: -0.0046609, -0.0030852, -0.0047739, -0.0031443, -0.0007268, 0.0009116
8: -0.0067916, -0.0055653, -0.0067457, -0.0054774, -0.0007095, 0.0005657
9: -0.0035296, -0.0034238, -0.0035372, -0.0034278, -0.0000488, 0.0000612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001429, upper bound: 0.0001503
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001429, upper bound: 0.0001506
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083719, -0.0072884, -0.0083268, -0.0072085, -0.0006273, 0.0005002
1: -0.0052990, -0.0049935, -0.0052863, -0.0049710, -0.0001769, 0.0001410
2: -0.0005374, 0.0017165, -0.0004436, 0.0018828, -0.0013050, 0.0010405
3: 0.0015562, 0.0018545, 0.0015686, 0.0018765, -0.0001727, 0.0001377
4: 0.0048090, 0.0064935, 0.0046848, 0.0064234, -0.0007776, 0.0009752
5: 0.9968423, 0.9973103, 0.9968078, 0.9972908, -0.0002160, 0.0002710
6: 0.0050174, 0.0054422, 0.0049861, 0.0054246, -0.0001961, 0.0002459
7: -0.0046573, -0.0030721, -0.0047742, -0.0031380, -0.0007318, 0.0009178
8: -0.0068019, -0.0055681, -0.0067505, -0.0054771, -0.0007143, 0.0005696
9: -0.0035293, -0.0034229, -0.0035372, -0.0034273, -0.0000491, 0.0000616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001460, upper bound: 0.0001546
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001460, upper bound: 0.0001562
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0082702, -0.0071973, -0.0083236, -0.0072086, -0.0004742, 0.0004989
1: -0.0052703, -0.0049678, -0.0052854, -0.0049710, -0.0001337, 0.0001407
2: -0.0003258, 0.0019060, -0.0004370, 0.0018825, -0.0009865, 0.0010379
3: 0.0015842, 0.0018795, 0.0015695, 0.0018764, -0.0001305, 0.0001373
4: 0.0046674, 0.0063353, 0.0046850, 0.0064185, -0.0007757, 0.0007372
5: 0.9968030, 0.9972664, 0.9968079, 0.9972895, -0.0002155, 0.0002048
6: 0.0049817, 0.0054023, 0.0049862, 0.0054233, -0.0001956, 0.0001859
7: -0.0047906, -0.0032209, -0.0047740, -0.0031426, -0.0007300, 0.0006938
8: -0.0066860, -0.0054644, -0.0067469, -0.0054773, -0.0005400, 0.0005681
9: -0.0035383, -0.0034329, -0.0035372, -0.0034276, -0.0000490, 0.0000466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001485
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001485
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0082724, -0.0071876, -0.0083236, -0.0072086, -0.0004921, 0.0005217
1: -0.0052710, -0.0049651, -0.0052854, -0.0049710, -0.0001387, 0.0001471
2: -0.0003304, 0.0019261, -0.0004370, 0.0018825, -0.0010236, 0.0010852
3: 0.0015836, 0.0018822, 0.0015695, 0.0018764, -0.0001355, 0.0001436
4: 0.0046524, 0.0063388, 0.0046850, 0.0064185, -0.0008110, 0.0007650
5: 0.9967988, 0.9972673, 0.9968079, 0.9972895, -0.0002253, 0.0002125
6: 0.0049779, 0.0054032, 0.0049862, 0.0054233, -0.0002045, 0.0001929
7: -0.0048047, -0.0032176, -0.0047740, -0.0031426, -0.0007632, 0.0007199
8: -0.0066886, -0.0054534, -0.0067469, -0.0054773, -0.0005603, 0.0005940
9: -0.0035392, -0.0034327, -0.0035372, -0.0034276, -0.0000512, 0.0000483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001493
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001494
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0082816, -0.0071972, -0.0083279, -0.0072084, -0.0004797, 0.0005047
1: -0.0052735, -0.0049678, -0.0052866, -0.0049710, -0.0001353, 0.0001423
2: -0.0003495, 0.0019061, -0.0004459, 0.0018830, -0.0009979, 0.0010498
3: 0.0015810, 0.0018795, 0.0015683, 0.0018765, -0.0001321, 0.0001389
4: 0.0046673, 0.0063530, 0.0046847, 0.0064251, -0.0007845, 0.0007458
5: 0.9968030, 0.9972713, 0.9968079, 0.9972913, -0.0002180, 0.0002072
6: 0.0049817, 0.0054068, 0.0049861, 0.0054250, -0.0001978, 0.0001881
7: -0.0047906, -0.0032042, -0.0047743, -0.0031364, -0.0007383, 0.0007019
8: -0.0066990, -0.0054643, -0.0067518, -0.0054770, -0.0005463, 0.0005746
9: -0.0035383, -0.0034318, -0.0035372, -0.0034272, -0.0000496, 0.0000471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001501
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001501
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0082835, -0.0071878, -0.0083279, -0.0072084, -0.0004986, 0.0005300
1: -0.0052741, -0.0049652, -0.0052866, -0.0049710, -0.0001406, 0.0001494
2: -0.0003535, 0.0019258, -0.0004459, 0.0018830, -0.0010372, 0.0011025
3: 0.0015805, 0.0018821, 0.0015683, 0.0018765, -0.0001373, 0.0001459
4: 0.0046527, 0.0063560, 0.0046847, 0.0064251, -0.0008240, 0.0007752
5: 0.9967989, 0.9972721, 0.9968079, 0.9972913, -0.0002289, 0.0002154
6: 0.0049780, 0.0054076, 0.0049861, 0.0054250, -0.0002078, 0.0001955
7: -0.0048045, -0.0032014, -0.0047743, -0.0031364, -0.0007754, 0.0007295
8: -0.0067012, -0.0054535, -0.0067518, -0.0054770, -0.0005678, 0.0006035
9: -0.0035392, -0.0034316, -0.0035372, -0.0034272, -0.0000521, 0.0000490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001522
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001522
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0083214, -0.0072057, -0.0082880, -0.0071990, -0.0005274, 0.0004805
1: -0.0052848, -0.0049702, -0.0052753, -0.0049683, -0.0001487, 0.0001355
2: -0.0004324, 0.0018886, -0.0003628, 0.0019025, -0.0010972, 0.0009996
3: 0.0015701, 0.0018772, 0.0015793, 0.0018791, -0.0001452, 0.0001323
4: 0.0046805, 0.0064150, 0.0046701, 0.0063630, -0.0007470, 0.0008200
5: 0.9968067, 0.9972885, 0.9968037, 0.9972740, -0.0002075, 0.0002278
6: 0.0049850, 0.0054224, 0.0049824, 0.0054093, -0.0001884, 0.0002068
7: -0.0047783, -0.0031459, -0.0047881, -0.0031948, -0.0007030, 0.0007717
8: -0.0067444, -0.0054739, -0.0067063, -0.0054663, -0.0006006, 0.0005472
9: -0.0035375, -0.0034279, -0.0035381, -0.0034311, -0.0000472, 0.0000518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001524, upper bound: 0.0001518
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001524, upper bound: 0.0001536
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083261, -0.0072054, -0.0082983, -0.0072008, -0.0005304, 0.0004858
1: -0.0052861, -0.0049701, -0.0052783, -0.0049688, -0.0001495, 0.0001370
2: -0.0004421, 0.0018891, -0.0003843, 0.0018988, -0.0011033, 0.0010105
3: 0.0015688, 0.0018773, 0.0015764, 0.0018786, -0.0001460, 0.0001337
4: 0.0046801, 0.0064223, 0.0046728, 0.0063791, -0.0007552, 0.0008245
5: 0.9968066, 0.9972905, 0.9968045, 0.9972785, -0.0002098, 0.0002291
6: 0.0049849, 0.0054243, 0.0049831, 0.0054134, -0.0001905, 0.0002079
7: -0.0047787, -0.0031391, -0.0047855, -0.0031797, -0.0007107, 0.0007760
8: -0.0067497, -0.0054736, -0.0067181, -0.0054683, -0.0006039, 0.0005532
9: -0.0035375, -0.0034274, -0.0035380, -0.0034301, -0.0000477, 0.0000521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001541, upper bound: 0.0001519
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001541, upper bound: 0.0001537
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083078, -0.0072051, -0.0083225, -0.0072087, -0.0005035, 0.0005036
1: -0.0052809, -0.0049701, -0.0052851, -0.0049711, -0.0001419, 0.0001420
2: -0.0004041, 0.0018897, -0.0004347, 0.0018823, -0.0010473, 0.0010477
3: 0.0015738, 0.0018774, 0.0015698, 0.0018764, -0.0001386, 0.0001386
4: 0.0046796, 0.0063939, 0.0046852, 0.0064167, -0.0007830, 0.0007827
5: 0.9968064, 0.9972826, 0.9968079, 0.9972889, -0.0002175, 0.0002174
6: 0.0049848, 0.0054171, 0.0049862, 0.0054229, -0.0001975, 0.0001974
7: -0.0047791, -0.0031658, -0.0047739, -0.0031443, -0.0007369, 0.0007366
8: -0.0067289, -0.0054733, -0.0067457, -0.0054774, -0.0005733, 0.0005735
9: -0.0035375, -0.0034292, -0.0035372, -0.0034278, -0.0000495, 0.0000495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001504
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001512
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083192, -0.0072059, -0.0083268, -0.0072085, -0.0005085, 0.0005072
1: -0.0052842, -0.0049703, -0.0052863, -0.0049710, -0.0001434, 0.0001430
2: -0.0004278, 0.0018881, -0.0004436, 0.0018828, -0.0010577, 0.0010550
3: 0.0015707, 0.0018772, 0.0015686, 0.0018765, -0.0001400, 0.0001396
4: 0.0046808, 0.0064116, 0.0046848, 0.0064234, -0.0007884, 0.0007905
5: 0.9968067, 0.9972875, 0.9968078, 0.9972908, -0.0002191, 0.0002196
6: 0.0049851, 0.0054216, 0.0049861, 0.0054246, -0.0001988, 0.0001993
7: -0.0047780, -0.0031491, -0.0047742, -0.0031380, -0.0007420, 0.0007439
8: -0.0067419, -0.0054741, -0.0067505, -0.0054771, -0.0005790, 0.0005775
9: -0.0035375, -0.0034281, -0.0035372, -0.0034273, -0.0000498, 0.0000500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001541, upper bound: 0.0001519
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001541, upper bound: 0.0001537
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083524, -0.0072794, -0.0083122, -0.0071964, -0.0006178, 0.0004830
1: -0.0052935, -0.0049910, -0.0052822, -0.0049676, -0.0001742, 0.0001362
2: -0.0004969, 0.0017352, -0.0004133, 0.0019079, -0.0012851, 0.0010047
3: 0.0015615, 0.0018569, 0.0015726, 0.0018798, -0.0001701, 0.0001330
4: 0.0047950, 0.0064632, 0.0046660, 0.0064007, -0.0007508, 0.0009604
5: 0.9968385, 0.9973019, 0.9968026, 0.9972845, -0.0002086, 0.0002668
6: 0.0050139, 0.0054346, 0.0049814, 0.0054188, -0.0001893, 0.0002422
7: -0.0046705, -0.0031005, -0.0047919, -0.0031594, -0.0007066, 0.0009039
8: -0.0067797, -0.0055578, -0.0067339, -0.0054633, -0.0007035, 0.0005500
9: -0.0035302, -0.0034248, -0.0035384, -0.0034288, -0.0000474, 0.0000607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B2_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001507
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001531
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083524, -0.0072794, -0.0083174, -0.0071999, -0.0006289, 0.0005036
1: -0.0052935, -0.0049910, -0.0052837, -0.0049686, -0.0001773, 0.0001420
2: -0.0004969, 0.0017352, -0.0004241, 0.0019007, -0.0013082, 0.0010476
3: 0.0015615, 0.0018569, 0.0015712, 0.0018788, -0.0001731, 0.0001386
4: 0.0047950, 0.0064632, 0.0046714, 0.0064088, -0.0007829, 0.0009777
5: 0.9968385, 0.9973019, 0.9968041, 0.9972868, -0.0002175, 0.0002716
6: 0.0050139, 0.0054346, 0.0049827, 0.0054209, -0.0001974, 0.0002466
7: -0.0046705, -0.0031005, -0.0047868, -0.0031517, -0.0007368, 0.0009201
8: -0.0067797, -0.0055578, -0.0067399, -0.0054673, -0.0007161, 0.0005735
9: -0.0035302, -0.0034248, -0.0035380, -0.0034283, -0.0000495, 0.0000618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B2_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001508
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001531
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0083621, -0.0072861, -0.0083027, -0.0071912, -0.0006625, 0.0004951
1: -0.0052963, -0.0049929, -0.0052795, -0.0049661, -0.0001868, 0.0001396
2: -0.0005171, 0.0017213, -0.0003934, 0.0019186, -0.0013781, 0.0010298
3: 0.0015589, 0.0018551, 0.0015752, 0.0018812, -0.0001824, 0.0001363
4: 0.0048055, 0.0064783, 0.0046580, 0.0063859, -0.0007696, 0.0010299
5: 0.9968414, 0.9973062, 0.9968004, 0.9972804, -0.0002138, 0.0002861
6: 0.0050165, 0.0054384, 0.0049794, 0.0054151, -0.0001941, 0.0002597
7: -0.0046606, -0.0030863, -0.0047994, -0.0031733, -0.0007243, 0.0009693
8: -0.0067908, -0.0055655, -0.0067231, -0.0054575, -0.0007544, 0.0005637
9: -0.0035296, -0.0034239, -0.0035389, -0.0034297, -0.0000486, 0.0000651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001292, upper bound: 0.0001451
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001292, upper bound: 0.0001451
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083714, -0.0072886, -0.0083073, -0.0071910, -0.0006643, 0.0004967
1: -0.0052989, -0.0049936, -0.0052808, -0.0049661, -0.0001873, 0.0001400
2: -0.0005364, 0.0017162, -0.0004030, 0.0019191, -0.0013818, 0.0010332
3: 0.0015563, 0.0018544, 0.0015740, 0.0018813, -0.0001829, 0.0001367
4: 0.0048093, 0.0064927, 0.0046576, 0.0063931, -0.0007722, 0.0010327
5: 0.9968424, 0.9973100, 0.9968003, 0.9972824, -0.0002145, 0.0002869
6: 0.0050175, 0.0054420, 0.0049793, 0.0054169, -0.0001947, 0.0002604
7: -0.0046570, -0.0030727, -0.0047998, -0.0031666, -0.0007267, 0.0009719
8: -0.0068013, -0.0055683, -0.0067283, -0.0054572, -0.0007564, 0.0005656
9: -0.0035293, -0.0034229, -0.0035389, -0.0034292, -0.0000488, 0.0000653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001555
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001555
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083621, -0.0072861, -0.0083235, -0.0071999, -0.0006384, 0.0004985
1: -0.0052963, -0.0049929, -0.0052854, -0.0049686, -0.0001800, 0.0001406
2: -0.0005171, 0.0017213, -0.0004367, 0.0019005, -0.0013280, 0.0010370
3: 0.0015589, 0.0018551, 0.0015695, 0.0018788, -0.0001757, 0.0001372
4: 0.0048055, 0.0064783, 0.0046715, 0.0064182, -0.0007750, 0.0009925
5: 0.9968414, 0.9973062, 0.9968042, 0.9972894, -0.0002153, 0.0002757
6: 0.0050165, 0.0054384, 0.0049828, 0.0054232, -0.0001954, 0.0002503
7: -0.0046606, -0.0030863, -0.0047867, -0.0031429, -0.0007294, 0.0009340
8: -0.0067908, -0.0055655, -0.0067467, -0.0054674, -0.0007270, 0.0005677
9: -0.0035296, -0.0034239, -0.0035380, -0.0034277, -0.0000490, 0.0000627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001451, upper bound: 0.0001506
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001451, upper bound: 0.0001506
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083714, -0.0072886, -0.0083283, -0.0071997, -0.0006409, 0.0005018
1: -0.0052989, -0.0049936, -0.0052867, -0.0049685, -0.0001807, 0.0001415
2: -0.0005364, 0.0017162, -0.0004466, 0.0019010, -0.0013333, 0.0010438
3: 0.0015563, 0.0018544, 0.0015682, 0.0018789, -0.0001764, 0.0001381
4: 0.0048093, 0.0064927, 0.0046712, 0.0064256, -0.0007801, 0.0009964
5: 0.9968424, 0.9973100, 0.9968041, 0.9972915, -0.0002167, 0.0002768
6: 0.0050175, 0.0054420, 0.0049827, 0.0054251, -0.0001967, 0.0002513
7: -0.0046570, -0.0030727, -0.0047871, -0.0031359, -0.0007342, 0.0009377
8: -0.0068013, -0.0055683, -0.0067522, -0.0054671, -0.0007299, 0.0005714
9: -0.0035293, -0.0034229, -0.0035381, -0.0034272, -0.0000493, 0.0000630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001561
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001561
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0082872, -0.0071968, -0.0083075, -0.0071966, -0.0005030, 0.0004781
1: -0.0052751, -0.0049677, -0.0052809, -0.0049676, -0.0001418, 0.0001348
2: -0.0003613, 0.0019071, -0.0004035, 0.0019075, -0.0010463, 0.0009946
3: 0.0015795, 0.0018797, 0.0015739, 0.0018797, -0.0001385, 0.0001316
4: 0.0046667, 0.0063619, 0.0046664, 0.0063934, -0.0007433, 0.0007820
5: 0.9968027, 0.9972737, 0.9968027, 0.9972825, -0.0002065, 0.0002173
6: 0.0049815, 0.0054090, 0.0049815, 0.0054170, -0.0001874, 0.0001972
7: -0.0047913, -0.0031959, -0.0047916, -0.0031663, -0.0006995, 0.0007359
8: -0.0067055, -0.0054638, -0.0067286, -0.0054636, -0.0005728, 0.0005444
9: -0.0035383, -0.0034312, -0.0035384, -0.0034292, -0.0000470, 0.0000494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B2_A2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001482
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001493
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0082872, -0.0071968, -0.0083130, -0.0072001, -0.0005157, 0.0005088
1: -0.0052751, -0.0049677, -0.0052824, -0.0049686, -0.0001454, 0.0001435
2: -0.0003613, 0.0019071, -0.0004150, 0.0019002, -0.0010728, 0.0010585
3: 0.0015795, 0.0018797, 0.0015724, 0.0018788, -0.0001420, 0.0001401
4: 0.0046667, 0.0063619, 0.0046718, 0.0064020, -0.0007911, 0.0008018
5: 0.9968027, 0.9972737, 0.9968042, 0.9972849, -0.0002198, 0.0002228
6: 0.0049815, 0.0054090, 0.0049828, 0.0054192, -0.0001995, 0.0002022
7: -0.0047913, -0.0031959, -0.0047865, -0.0031582, -0.0007445, 0.0007546
8: -0.0067055, -0.0054638, -0.0067349, -0.0054676, -0.0005873, 0.0005794
9: -0.0035383, -0.0034312, -0.0035380, -0.0034287, -0.0000500, 0.0000507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B2_A2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001482
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001493
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0082980, -0.0071967, -0.0083122, -0.0071964, -0.0005050, 0.0004838
1: -0.0052782, -0.0049677, -0.0052822, -0.0049676, -0.0001424, 0.0001364
2: -0.0003836, 0.0019072, -0.0004133, 0.0019079, -0.0010504, 0.0010064
3: 0.0015765, 0.0018797, 0.0015726, 0.0018798, -0.0001390, 0.0001332
4: 0.0046666, 0.0063786, 0.0046660, 0.0064007, -0.0007521, 0.0007850
5: 0.9968028, 0.9972785, 0.9968026, 0.9972845, -0.0002090, 0.0002181
6: 0.0049815, 0.0054133, 0.0049814, 0.0054188, -0.0001897, 0.0001980
7: -0.0047914, -0.0031802, -0.0047919, -0.0031594, -0.0007078, 0.0007388
8: -0.0067177, -0.0054637, -0.0067339, -0.0054633, -0.0005750, 0.0005509
9: -0.0035384, -0.0034302, -0.0035384, -0.0034288, -0.0000475, 0.0000496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B2_A2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001490
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001522
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0082980, -0.0071967, -0.0083174, -0.0071999, -0.0005206, 0.0005138
1: -0.0052782, -0.0049677, -0.0052837, -0.0049686, -0.0001468, 0.0001449
2: -0.0003836, 0.0019072, -0.0004241, 0.0019007, -0.0010829, 0.0010688
3: 0.0015765, 0.0018797, 0.0015712, 0.0018788, -0.0001433, 0.0001414
4: 0.0046666, 0.0063786, 0.0046714, 0.0064088, -0.0007988, 0.0008093
5: 0.9968028, 0.9972785, 0.9968041, 0.9972868, -0.0002219, 0.0002249
6: 0.0049815, 0.0054133, 0.0049827, 0.0054209, -0.0002014, 0.0002041
7: -0.0047914, -0.0031802, -0.0047868, -0.0031517, -0.0007517, 0.0007617
8: -0.0067177, -0.0054637, -0.0067399, -0.0054673, -0.0005928, 0.0005851
9: -0.0035384, -0.0034302, -0.0035380, -0.0034283, -0.0000505, 0.0000511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_B2_A2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001490
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001522
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0083209, -0.0072059, -0.0082893, -0.0071908, -0.0005476, 0.0004811
1: -0.0052846, -0.0049703, -0.0052757, -0.0049660, -0.0001544, 0.0001356
2: -0.0004312, 0.0018882, -0.0003656, 0.0019195, -0.0011390, 0.0010007
3: 0.0015702, 0.0018772, 0.0015789, 0.0018813, -0.0001507, 0.0001324
4: 0.0046808, 0.0064142, 0.0046573, 0.0063651, -0.0007479, 0.0008512
5: 0.9968067, 0.9972883, 0.9968002, 0.9972747, -0.0002078, 0.0002365
6: 0.0049851, 0.0054222, 0.0049792, 0.0054099, -0.0001886, 0.0002147
7: -0.0047780, -0.0031467, -0.0048001, -0.0031929, -0.0007038, 0.0008011
8: -0.0067438, -0.0054741, -0.0067079, -0.0054570, -0.0006235, 0.0005478
9: -0.0035375, -0.0034279, -0.0035389, -0.0034310, -0.0000473, 0.0000538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001535
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001535
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083257, -0.0072056, -0.0083005, -0.0071913, -0.0005523, 0.0004876
1: -0.0052860, -0.0049702, -0.0052789, -0.0049662, -0.0001557, 0.0001375
2: -0.0004413, 0.0018887, -0.0003888, 0.0019185, -0.0011490, 0.0010142
3: 0.0015689, 0.0018772, 0.0015758, 0.0018812, -0.0001521, 0.0001342
4: 0.0046804, 0.0064216, 0.0046581, 0.0063825, -0.0007580, 0.0008587
5: 0.9968066, 0.9972903, 0.9968004, 0.9972795, -0.0002106, 0.0002386
6: 0.0049850, 0.0054241, 0.0049794, 0.0054142, -0.0001911, 0.0002165
7: -0.0047784, -0.0031397, -0.0047993, -0.0031765, -0.0007133, 0.0008081
8: -0.0067493, -0.0054738, -0.0067206, -0.0054575, -0.0006290, 0.0005552
9: -0.0035375, -0.0034274, -0.0035389, -0.0034299, -0.0000479, 0.0000543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001537
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001537
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0083072, -0.0072053, -0.0083235, -0.0071999, -0.0005255, 0.0005041
1: -0.0052808, -0.0049701, -0.0052854, -0.0049686, -0.0001482, 0.0001421
2: -0.0004028, 0.0018893, -0.0004367, 0.0019005, -0.0010931, 0.0010487
3: 0.0015740, 0.0018773, 0.0015695, 0.0018788, -0.0001447, 0.0001388
4: 0.0046799, 0.0063929, 0.0046715, 0.0064182, -0.0007838, 0.0008169
5: 0.9968065, 0.9972823, 0.9968042, 0.9972894, -0.0002177, 0.0002270
6: 0.0049849, 0.0054169, 0.0049828, 0.0054232, -0.0001977, 0.0002060
7: -0.0047788, -0.0031667, -0.0047867, -0.0031429, -0.0007376, 0.0007688
8: -0.0067282, -0.0054735, -0.0067467, -0.0054674, -0.0005984, 0.0005741
9: -0.0035375, -0.0034293, -0.0035380, -0.0034277, -0.0000495, 0.0000516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001513
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001512
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083189, -0.0072061, -0.0083283, -0.0071997, -0.0005301, 0.0005076
1: -0.0052841, -0.0049703, -0.0052867, -0.0049685, -0.0001495, 0.0001431
2: -0.0004272, 0.0018877, -0.0004466, 0.0019010, -0.0011027, 0.0010559
3: 0.0015708, 0.0018771, 0.0015682, 0.0018789, -0.0001459, 0.0001397
4: 0.0046811, 0.0064112, 0.0046712, 0.0064256, -0.0007891, 0.0008241
5: 0.9968068, 0.9972875, 0.9968041, 0.9972915, -0.0002192, 0.0002290
6: 0.0049852, 0.0054215, 0.0049827, 0.0054251, -0.0001990, 0.0002078
7: -0.0047777, -0.0031495, -0.0047871, -0.0031359, -0.0007427, 0.0007756
8: -0.0067416, -0.0054744, -0.0067522, -0.0054671, -0.0006036, 0.0005780
9: -0.0035374, -0.0034281, -0.0035381, -0.0034272, -0.0000499, 0.0000521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001537
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001537
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083685, -0.0072896, -0.0083523, -0.0072792, -0.0005085, 0.0004964
1: -0.0052981, -0.0049939, -0.0052935, -0.0049909, -0.0001434, 0.0001399
2: -0.0005304, 0.0017139, -0.0004966, 0.0017356, -0.0010577, 0.0010325
3: 0.0015571, 0.0018541, 0.0015616, 0.0018570, -0.0001400, 0.0001366
4: 0.0048110, 0.0064883, 0.0047948, 0.0064630, -0.0007717, 0.0007905
5: 0.9968429, 0.9973089, 0.9968384, 0.9973019, -0.0002144, 0.0002196
6: 0.0050179, 0.0054409, 0.0050138, 0.0054345, -0.0001946, 0.0001994
7: -0.0046555, -0.0030770, -0.0046707, -0.0031007, -0.0007262, 0.0007439
8: -0.0067981, -0.0055695, -0.0067795, -0.0055576, -0.0005790, 0.0005652
9: -0.0035292, -0.0034232, -0.0035302, -0.0034248, -0.0000488, 0.0000500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001462
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001462
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083685, -0.0072896, -0.0083575, -0.0072815, -0.0004962, 0.0004821
1: -0.0052981, -0.0049939, -0.0052950, -0.0049916, -0.0001399, 0.0001359
2: -0.0005304, 0.0017139, -0.0005075, 0.0017308, -0.0010322, 0.0010028
3: 0.0015571, 0.0018541, 0.0015601, 0.0018563, -0.0001366, 0.0001327
4: 0.0048110, 0.0064883, 0.0047984, 0.0064712, -0.0007495, 0.0007714
5: 0.9968429, 0.9973089, 0.9968394, 0.9973041, -0.0002082, 0.0002143
6: 0.0050179, 0.0054409, 0.0050147, 0.0054366, -0.0001890, 0.0001945
7: -0.0046555, -0.0030770, -0.0046673, -0.0030931, -0.0007053, 0.0007260
8: -0.0067981, -0.0055695, -0.0067855, -0.0055603, -0.0005650, 0.0005490
9: -0.0035292, -0.0034232, -0.0035300, -0.0034243, -0.0000474, 0.0000487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001427, upper bound: 0.0001465
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001427, upper bound: 0.0001465
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083164, -0.0072086, -0.0083523, -0.0072792, -0.0005013, 0.0006156
1: -0.0052834, -0.0049710, -0.0052935, -0.0049909, -0.0001413, 0.0001735
2: -0.0004220, 0.0018824, -0.0004966, 0.0017356, -0.0010429, 0.0012805
3: 0.0015715, 0.0018764, 0.0015616, 0.0018570, -0.0001380, 0.0001695
4: 0.0046851, 0.0064072, 0.0047948, 0.0064630, -0.0009570, 0.0007794
5: 0.9968079, 0.9972863, 0.9968384, 0.9973019, -0.0002659, 0.0002165
6: 0.0049862, 0.0054205, 0.0050138, 0.0054345, -0.0002413, 0.0001965
7: -0.0047740, -0.0031532, -0.0046707, -0.0031007, -0.0009006, 0.0007335
8: -0.0067387, -0.0054773, -0.0067795, -0.0055576, -0.0005709, 0.0007009
9: -0.0035372, -0.0034284, -0.0035302, -0.0034248, -0.0000605, 0.0000493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001462
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001462
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083164, -0.0072086, -0.0083575, -0.0072815, -0.0004942, 0.0006076
1: -0.0052834, -0.0049710, -0.0052950, -0.0049916, -0.0001393, 0.0001713
2: -0.0004220, 0.0018824, -0.0005075, 0.0017308, -0.0010280, 0.0012639
3: 0.0015715, 0.0018764, 0.0015601, 0.0018563, -0.0001360, 0.0001673
4: 0.0046851, 0.0064072, 0.0047984, 0.0064712, -0.0009446, 0.0007683
5: 0.9968079, 0.9972863, 0.9968394, 0.9973041, -0.0002624, 0.0002134
6: 0.0049862, 0.0054205, 0.0050147, 0.0054366, -0.0002382, 0.0001937
7: -0.0047740, -0.0031532, -0.0046673, -0.0030931, -0.0008889, 0.0007230
8: -0.0067387, -0.0054773, -0.0067855, -0.0055603, -0.0005627, 0.0006919
9: -0.0035372, -0.0034284, -0.0035300, -0.0034243, -0.0000597, 0.0000485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001465
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001465
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083480, -0.0072816, -0.0083810, -0.0072878, -0.0004998, 0.0005393
1: -0.0052923, -0.0049916, -0.0053016, -0.0049934, -0.0001409, 0.0001521
2: -0.0004877, 0.0017307, -0.0005564, 0.0017177, -0.0010397, 0.0011219
3: 0.0015628, 0.0018563, 0.0015537, 0.0018546, -0.0001376, 0.0001485
4: 0.0047985, 0.0064563, 0.0048081, 0.0065077, -0.0008385, 0.0007770
5: 0.9968394, 0.9973000, 0.9968421, 0.9973143, -0.0002329, 0.0002159
6: 0.0050148, 0.0054329, 0.0050172, 0.0054458, -0.0002114, 0.0001960
7: -0.0046672, -0.0031070, -0.0046582, -0.0030587, -0.0007891, 0.0007313
8: -0.0067747, -0.0055603, -0.0068123, -0.0055674, -0.0005692, 0.0006141
9: -0.0035300, -0.0034253, -0.0035294, -0.0034220, -0.0000530, 0.0000491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001375, upper bound: 0.0001158
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001418
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083480, -0.0072816, -0.0083875, -0.0072888, -0.0004773, 0.0005269
1: -0.0052923, -0.0049916, -0.0053034, -0.0049936, -0.0001346, 0.0001486
2: -0.0004877, 0.0017307, -0.0005698, 0.0017157, -0.0009928, 0.0010961
3: 0.0015628, 0.0018563, 0.0015519, 0.0018543, -0.0001314, 0.0001451
4: 0.0047985, 0.0064563, 0.0048096, 0.0065177, -0.0008192, 0.0007420
5: 0.9968394, 0.9973000, 0.9968426, 0.9973170, -0.0002276, 0.0002061
6: 0.0050148, 0.0054329, 0.0050176, 0.0054483, -0.0002066, 0.0001871
7: -0.0046672, -0.0031070, -0.0046568, -0.0030493, -0.0007710, 0.0006983
8: -0.0067747, -0.0055603, -0.0068196, -0.0055685, -0.0005435, 0.0006000
9: -0.0035300, -0.0034253, -0.0035293, -0.0034214, -0.0000518, 0.0000469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001375, upper bound: 0.0001158
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001420
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083810, -0.0072878, -0.0005080, 0.0005185
1: -0.0052982, -0.0049939, -0.0053016, -0.0049934, -0.0001432, 0.0001462
2: -0.0005318, 0.0017140, -0.0005564, 0.0017177, -0.0010567, 0.0010786
3: 0.0015569, 0.0018541, 0.0015537, 0.0018546, -0.0001398, 0.0001427
4: 0.0048109, 0.0064893, 0.0048081, 0.0065077, -0.0008061, 0.0007897
5: 0.9968429, 0.9973091, 0.9968421, 0.9973143, -0.0002240, 0.0002194
6: 0.0050179, 0.0054412, 0.0050172, 0.0054458, -0.0002033, 0.0001992
7: -0.0046555, -0.0030760, -0.0046582, -0.0030587, -0.0007586, 0.0007432
8: -0.0067988, -0.0055695, -0.0068123, -0.0055674, -0.0005784, 0.0005904
9: -0.0035292, -0.0034232, -0.0035294, -0.0034220, -0.0000509, 0.0000499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001401, upper bound: 0.0001448
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001484
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083875, -0.0072888, -0.0004854, 0.0005046
1: -0.0052982, -0.0049939, -0.0053034, -0.0049936, -0.0001369, 0.0001423
2: -0.0005318, 0.0017140, -0.0005698, 0.0017157, -0.0010097, 0.0010496
3: 0.0015569, 0.0018541, 0.0015519, 0.0018543, -0.0001336, 0.0001389
4: 0.0048109, 0.0064893, 0.0048096, 0.0065177, -0.0007844, 0.0007546
5: 0.9968429, 0.9973091, 0.9968426, 0.9973170, -0.0002179, 0.0002097
6: 0.0050179, 0.0054412, 0.0050176, 0.0054483, -0.0001978, 0.0001903
7: -0.0046555, -0.0030760, -0.0046568, -0.0030493, -0.0007382, 0.0007102
8: -0.0067988, -0.0055695, -0.0068196, -0.0055685, -0.0005527, 0.0005746
9: -0.0035292, -0.0034232, -0.0035293, -0.0034214, -0.0000496, 0.0000477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001401, upper bound: 0.0001449
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001486
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0082955, -0.0072006, -0.0083810, -0.0072878, -0.0004938, 0.0006667
1: -0.0052775, -0.0049688, -0.0053016, -0.0049934, -0.0001392, 0.0001880
2: -0.0003786, 0.0018991, -0.0005564, 0.0017177, -0.0010272, 0.0013869
3: 0.0015772, 0.0018786, 0.0015537, 0.0018546, -0.0001359, 0.0001835
4: 0.0046726, 0.0063748, 0.0048081, 0.0065077, -0.0010365, 0.0007677
5: 0.9968045, 0.9972773, 0.9968421, 0.9973143, -0.0002880, 0.0002133
6: 0.0049830, 0.0054123, 0.0050172, 0.0054458, -0.0002614, 0.0001936
7: -0.0047857, -0.0031838, -0.0046582, -0.0030587, -0.0009755, 0.0007224
8: -0.0067149, -0.0054682, -0.0068123, -0.0055674, -0.0005623, 0.0007592
9: -0.0035380, -0.0034304, -0.0035294, -0.0034220, -0.0000655, 0.0000485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001520, upper bound: 0.0001223
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001417
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0082955, -0.0072006, -0.0083875, -0.0072888, -0.0004788, 0.0006593
1: -0.0052775, -0.0049688, -0.0053034, -0.0049936, -0.0001350, 0.0001859
2: -0.0003786, 0.0018991, -0.0005698, 0.0017157, -0.0009959, 0.0013714
3: 0.0015772, 0.0018786, 0.0015519, 0.0018543, -0.0001318, 0.0001815
4: 0.0046726, 0.0063748, 0.0048096, 0.0065177, -0.0010249, 0.0007443
5: 0.9968045, 0.9972773, 0.9968426, 0.9973170, -0.0002847, 0.0002068
6: 0.0049830, 0.0054123, 0.0050176, 0.0054483, -0.0002585, 0.0001877
7: -0.0047857, -0.0031838, -0.0046568, -0.0030493, -0.0009645, 0.0007004
8: -0.0067149, -0.0054682, -0.0068196, -0.0055685, -0.0005452, 0.0007507
9: -0.0035380, -0.0034304, -0.0035293, -0.0034214, -0.0000648, 0.0000470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001520, upper bound: 0.0001223
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001420
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083166, -0.0072086, -0.0083810, -0.0072878, -0.0004990, 0.0006425
1: -0.0052834, -0.0049710, -0.0053016, -0.0049934, -0.0001407, 0.0001812
2: -0.0004223, 0.0018824, -0.0005564, 0.0017177, -0.0010381, 0.0013366
3: 0.0015714, 0.0018764, 0.0015537, 0.0018546, -0.0001374, 0.0001769
4: 0.0046850, 0.0064075, 0.0048081, 0.0065077, -0.0009989, 0.0007758
5: 0.9968079, 0.9972864, 0.9968421, 0.9973143, -0.0002775, 0.0002155
6: 0.0049862, 0.0054205, 0.0050172, 0.0054458, -0.0002519, 0.0001957
7: -0.0047740, -0.0031530, -0.0046582, -0.0030587, -0.0009400, 0.0007301
8: -0.0067389, -0.0054773, -0.0068123, -0.0055674, -0.0005683, 0.0007316
9: -0.0035372, -0.0034283, -0.0035294, -0.0034220, -0.0000631, 0.0000490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001448
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001543, upper bound: 0.0001481
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083166, -0.0072086, -0.0083875, -0.0072888, -0.0004842, 0.0006346
1: -0.0052834, -0.0049710, -0.0053034, -0.0049936, -0.0001365, 0.0001789
2: -0.0004223, 0.0018824, -0.0005698, 0.0017157, -0.0010072, 0.0013201
3: 0.0015714, 0.0018764, 0.0015519, 0.0018543, -0.0001333, 0.0001747
4: 0.0046850, 0.0064075, 0.0048096, 0.0065177, -0.0009866, 0.0007527
5: 0.9968079, 0.9972864, 0.9968426, 0.9973170, -0.0002741, 0.0002091
6: 0.0049862, 0.0054205, 0.0050176, 0.0054483, -0.0002488, 0.0001898
7: -0.0047740, -0.0031530, -0.0046568, -0.0030493, -0.0009285, 0.0007084
8: -0.0067389, -0.0054773, -0.0068196, -0.0055685, -0.0005513, 0.0007226
9: -0.0035372, -0.0034283, -0.0035293, -0.0034214, -0.0000623, 0.0000476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001449
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001543, upper bound: 0.0001482
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083711, -0.0072797, -0.0083524, -0.0072794, -0.0005107, 0.0005157
1: -0.0052988, -0.0049911, -0.0052935, -0.0049910, -0.0001440, 0.0001454
2: -0.0005357, 0.0017347, -0.0004969, 0.0017352, -0.0010625, 0.0010728
3: 0.0015564, 0.0018569, 0.0015615, 0.0018569, -0.0001406, 0.0001420
4: 0.0047955, 0.0064922, 0.0047950, 0.0064632, -0.0008018, 0.0007940
5: 0.9968385, 0.9973099, 0.9968385, 0.9973019, -0.0002228, 0.0002206
6: 0.0050140, 0.0054419, 0.0050139, 0.0054346, -0.0002022, 0.0002002
7: -0.0046701, -0.0030732, -0.0046705, -0.0031005, -0.0007546, 0.0007473
8: -0.0068010, -0.0055582, -0.0067797, -0.0055578, -0.0005816, 0.0005873
9: -0.0035302, -0.0034230, -0.0035302, -0.0034248, -0.0000507, 0.0000502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001484
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001484
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083711, -0.0072797, -0.0083571, -0.0072817, -0.0004981, 0.0005033
1: -0.0052988, -0.0049911, -0.0052948, -0.0049916, -0.0001404, 0.0001419
2: -0.0005357, 0.0017347, -0.0005065, 0.0017304, -0.0010362, 0.0010470
3: 0.0015564, 0.0018569, 0.0015603, 0.0018563, -0.0001371, 0.0001386
4: 0.0047955, 0.0064922, 0.0047987, 0.0064704, -0.0007825, 0.0007744
5: 0.9968385, 0.9973099, 0.9968395, 0.9973039, -0.0002174, 0.0002152
6: 0.0050140, 0.0054419, 0.0050148, 0.0054364, -0.0001973, 0.0001953
7: -0.0046701, -0.0030732, -0.0046670, -0.0030938, -0.0007364, 0.0007288
8: -0.0068010, -0.0055582, -0.0067850, -0.0055605, -0.0005672, 0.0005731
9: -0.0035302, -0.0034230, -0.0035300, -0.0034244, -0.0000494, 0.0000489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001411, upper bound: 0.0001485
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001411, upper bound: 0.0001485
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083174, -0.0071999, -0.0083524, -0.0072794, -0.0005036, 0.0006289
1: -0.0052837, -0.0049686, -0.0052935, -0.0049910, -0.0001420, 0.0001773
2: -0.0004241, 0.0019007, -0.0004969, 0.0017352, -0.0010476, 0.0013082
3: 0.0015712, 0.0018788, 0.0015615, 0.0018569, -0.0001386, 0.0001731
4: 0.0046714, 0.0064088, 0.0047950, 0.0064632, -0.0009777, 0.0007829
5: 0.9968041, 0.9972868, 0.9968385, 0.9973019, -0.0002716, 0.0002175
6: 0.0049827, 0.0054209, 0.0050139, 0.0054346, -0.0002466, 0.0001974
7: -0.0047868, -0.0031517, -0.0046705, -0.0031005, -0.0009201, 0.0007368
8: -0.0067399, -0.0054673, -0.0067797, -0.0055578, -0.0005735, 0.0007161
9: -0.0035380, -0.0034283, -0.0035302, -0.0034248, -0.0000618, 0.0000495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001507, upper bound: 0.0001484
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001507, upper bound: 0.0001484
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083174, -0.0071999, -0.0083571, -0.0072817, -0.0004962, 0.0006207
1: -0.0052837, -0.0049686, -0.0052948, -0.0049916, -0.0001399, 0.0001750
2: -0.0004241, 0.0019007, -0.0005065, 0.0017304, -0.0010322, 0.0012911
3: 0.0015712, 0.0018788, 0.0015603, 0.0018563, -0.0001366, 0.0001709
4: 0.0046714, 0.0064088, 0.0047987, 0.0064704, -0.0009649, 0.0007714
5: 0.9968041, 0.9972868, 0.9968395, 0.9973039, -0.0002681, 0.0002143
6: 0.0049827, 0.0054209, 0.0050148, 0.0054364, -0.0002433, 0.0001945
7: -0.0047868, -0.0031517, -0.0046670, -0.0030938, -0.0009081, 0.0007259
8: -0.0067399, -0.0054673, -0.0067850, -0.0055605, -0.0005650, 0.0007067
9: -0.0035380, -0.0034283, -0.0035300, -0.0034244, -0.0000610, 0.0000487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001485
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001485
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0083500, -0.0072718, -0.0083805, -0.0072880, -0.0005017, 0.0005621
1: -0.0052928, -0.0049889, -0.0053014, -0.0049934, -0.0001415, 0.0001585
2: -0.0004919, 0.0017510, -0.0005552, 0.0017174, -0.0010437, 0.0011693
3: 0.0015622, 0.0018590, 0.0015538, 0.0018546, -0.0001381, 0.0001547
4: 0.0047833, 0.0064595, 0.0048084, 0.0065068, -0.0008739, 0.0007800
5: 0.9968352, 0.9973009, 0.9968421, 0.9973140, -0.0002428, 0.0002167
6: 0.0050109, 0.0054337, 0.0050173, 0.0054456, -0.0002204, 0.0001967
7: -0.0046815, -0.0031041, -0.0046579, -0.0030595, -0.0008224, 0.0007341
8: -0.0067770, -0.0055492, -0.0068116, -0.0055676, -0.0005713, 0.0006401
9: -0.0035310, -0.0034251, -0.0035294, -0.0034221, -0.0000552, 0.0000493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001360, upper bound: 0.0001145
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001436
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0083500, -0.0072718, -0.0083869, -0.0072890, -0.0004791, 0.0005544
1: -0.0052928, -0.0049889, -0.0053033, -0.0049937, -0.0001351, 0.0001563
2: -0.0004919, 0.0017510, -0.0005687, 0.0017153, -0.0009967, 0.0011533
3: 0.0015622, 0.0018590, 0.0015520, 0.0018543, -0.0001319, 0.0001526
4: 0.0047833, 0.0064595, 0.0048100, 0.0065169, -0.0008619, 0.0007448
5: 0.9968352, 0.9973009, 0.9968426, 0.9973167, -0.0002395, 0.0002069
6: 0.0050109, 0.0054337, 0.0050177, 0.0054481, -0.0002174, 0.0001878
7: -0.0046815, -0.0031041, -0.0046564, -0.0030500, -0.0008111, 0.0007010
8: -0.0067770, -0.0055492, -0.0068190, -0.0055688, -0.0005456, 0.0006313
9: -0.0035310, -0.0034251, -0.0035293, -0.0034214, -0.0000545, 0.0000471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0001360, upper bound: 0.0001145
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001436
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083717, -0.0072796, -0.0083805, -0.0072880, -0.0005096, 0.0005411
1: -0.0052989, -0.0049911, -0.0053014, -0.0049934, -0.0001437, 0.0001526
2: -0.0005369, 0.0017347, -0.0005552, 0.0017174, -0.0010601, 0.0011256
3: 0.0015562, 0.0018569, 0.0015538, 0.0018546, -0.0001403, 0.0001490
4: 0.0047954, 0.0064931, 0.0048084, 0.0065068, -0.0008412, 0.0007923
5: 0.9968385, 0.9973102, 0.9968421, 0.9973140, -0.0002337, 0.0002201
6: 0.0050140, 0.0054421, 0.0050173, 0.0054456, -0.0002121, 0.0001998
7: -0.0046701, -0.0030724, -0.0046579, -0.0030595, -0.0007917, 0.0007456
8: -0.0068016, -0.0055581, -0.0068116, -0.0055676, -0.0005803, 0.0006162
9: -0.0035302, -0.0034229, -0.0035294, -0.0034221, -0.0000532, 0.0000501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001391, upper bound: 0.0001471
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001511
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083717, -0.0072796, -0.0083869, -0.0072890, -0.0004875, 0.0005320
1: -0.0052989, -0.0049911, -0.0053033, -0.0049937, -0.0001375, 0.0001500
2: -0.0005369, 0.0017347, -0.0005687, 0.0017153, -0.0010142, 0.0011066
3: 0.0015562, 0.0018569, 0.0015520, 0.0018543, -0.0001342, 0.0001464
4: 0.0047954, 0.0064931, 0.0048100, 0.0065169, -0.0008270, 0.0007579
5: 0.9968385, 0.9973102, 0.9968426, 0.9973167, -0.0002298, 0.0002106
6: 0.0050140, 0.0054421, 0.0050177, 0.0054481, -0.0002086, 0.0001911
7: -0.0046701, -0.0030724, -0.0046564, -0.0030500, -0.0007783, 0.0007133
8: -0.0068016, -0.0055581, -0.0068190, -0.0055688, -0.0005552, 0.0006058
9: -0.0035302, -0.0034229, -0.0035293, -0.0034214, -0.0000523, 0.0000479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001391, upper bound: 0.0001472
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001513
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0082969, -0.0071912, -0.0083805, -0.0072880, -0.0004957, 0.0006825
1: -0.0052779, -0.0049661, -0.0053014, -0.0049934, -0.0001398, 0.0001924
2: -0.0003814, 0.0019188, -0.0005552, 0.0017174, -0.0010311, 0.0014197
3: 0.0015768, 0.0018812, 0.0015538, 0.0018546, -0.0001365, 0.0001879
4: 0.0046579, 0.0063769, 0.0048084, 0.0065068, -0.0010610, 0.0007706
5: 0.9968003, 0.9972779, 0.9968421, 0.9973140, -0.0002948, 0.0002141
6: 0.0049793, 0.0054128, 0.0050173, 0.0054456, -0.0002676, 0.0001943
7: -0.0047995, -0.0031818, -0.0046579, -0.0030595, -0.0009985, 0.0007252
8: -0.0067165, -0.0054574, -0.0068116, -0.0055676, -0.0005644, 0.0007772
9: -0.0035389, -0.0034303, -0.0035294, -0.0034221, -0.0000670, 0.0000487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001451, upper bound: 0.0001292
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001554, upper bound: 0.0001436
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0082969, -0.0071912, -0.0083869, -0.0072890, -0.0004808, 0.0006763
1: -0.0052779, -0.0049661, -0.0053033, -0.0049937, -0.0001355, 0.0001907
2: -0.0003814, 0.0019188, -0.0005687, 0.0017153, -0.0010001, 0.0014068
3: 0.0015768, 0.0018812, 0.0015520, 0.0018543, -0.0001323, 0.0001862
4: 0.0046579, 0.0063769, 0.0048100, 0.0065169, -0.0010514, 0.0007474
5: 0.9968003, 0.9972779, 0.9968426, 0.9973167, -0.0002921, 0.0002077
6: 0.0049793, 0.0054128, 0.0050177, 0.0054481, -0.0002651, 0.0001885
7: -0.0047995, -0.0031818, -0.0046564, -0.0030500, -0.0009895, 0.0007034
8: -0.0067165, -0.0054574, -0.0068190, -0.0055688, -0.0005475, 0.0007701
9: -0.0035389, -0.0034303, -0.0035293, -0.0034214, -0.0000664, 0.0000472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001451, upper bound: 0.0001292
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001554, upper bound: 0.0001436
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083177, -0.0071999, -0.0083805, -0.0072880, -0.0005007, 0.0006586
1: -0.0052837, -0.0049686, -0.0053014, -0.0049934, -0.0001412, 0.0001857
2: -0.0004247, 0.0019007, -0.0005552, 0.0017174, -0.0010416, 0.0013700
3: 0.0015711, 0.0018788, 0.0015538, 0.0018546, -0.0001378, 0.0001813
4: 0.0046714, 0.0064092, 0.0048084, 0.0065068, -0.0010238, 0.0007784
5: 0.9968041, 0.9972869, 0.9968421, 0.9973140, -0.0002844, 0.0002163
6: 0.0049827, 0.0054210, 0.0050173, 0.0054456, -0.0002582, 0.0001963
7: -0.0047868, -0.0031513, -0.0046579, -0.0030595, -0.0009635, 0.0007326
8: -0.0067402, -0.0054673, -0.0068116, -0.0055676, -0.0005702, 0.0007499
9: -0.0035380, -0.0034282, -0.0035294, -0.0034221, -0.0000647, 0.0000492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001471
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001542, upper bound: 0.0001510
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083177, -0.0071999, -0.0083869, -0.0072890, -0.0004864, 0.0006505
1: -0.0052837, -0.0049686, -0.0053033, -0.0049937, -0.0001371, 0.0001834
2: -0.0004247, 0.0019007, -0.0005687, 0.0017153, -0.0010117, 0.0013532
3: 0.0015711, 0.0018788, 0.0015520, 0.0018543, -0.0001339, 0.0001791
4: 0.0046714, 0.0064092, 0.0048100, 0.0065169, -0.0010113, 0.0007561
5: 0.9968041, 0.9972869, 0.9968426, 0.9973167, -0.0002810, 0.0002101
6: 0.0049827, 0.0054210, 0.0050177, 0.0054481, -0.0002550, 0.0001907
7: -0.0047868, -0.0031513, -0.0046564, -0.0030500, -0.0009517, 0.0007116
8: -0.0067402, -0.0054673, -0.0068190, -0.0055688, -0.0005538, 0.0007407
9: -0.0035380, -0.0034282, -0.0035293, -0.0034214, -0.0000639, 0.0000478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001472
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001542, upper bound: 0.0001511
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083395, -0.0072822, -0.0083279, -0.0072053, -0.0006113, 0.0005150
1: -0.0052899, -0.0049918, -0.0052866, -0.0049701, -0.0001724, 0.0001452
2: -0.0004699, 0.0017294, -0.0004459, 0.0018893, -0.0012717, 0.0010713
3: 0.0015651, 0.0018561, 0.0015683, 0.0018773, -0.0001683, 0.0001418
4: 0.0047994, 0.0064430, 0.0046799, 0.0064251, -0.0008006, 0.0009504
5: 0.9968396, 0.9972963, 0.9968065, 0.9972913, -0.0002224, 0.0002641
6: 0.0050150, 0.0054295, 0.0049849, 0.0054250, -0.0002019, 0.0002397
7: -0.0046663, -0.0031195, -0.0047788, -0.0031364, -0.0007535, 0.0008944
8: -0.0067649, -0.0055611, -0.0067518, -0.0054735, -0.0006961, 0.0005864
9: -0.0035300, -0.0034261, -0.0035375, -0.0034272, -0.0000506, 0.0000601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001456, upper bound: 0.0001508
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001456, upper bound: 0.0001508
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083395, -0.0072822, -0.0083334, -0.0072078, -0.0005877, 0.0004993
1: -0.0052899, -0.0049918, -0.0052882, -0.0049708, -0.0001657, 0.0001408
2: -0.0004699, 0.0017294, -0.0004573, 0.0018841, -0.0012226, 0.0010386
3: 0.0015651, 0.0018561, 0.0015668, 0.0018766, -0.0001618, 0.0001374
4: 0.0047994, 0.0064430, 0.0046838, 0.0064336, -0.0007762, 0.0009137
5: 0.9968396, 0.9972963, 0.9968075, 0.9972938, -0.0002157, 0.0002538
6: 0.0050150, 0.0054295, 0.0049859, 0.0054271, -0.0001958, 0.0002304
7: -0.0046663, -0.0031195, -0.0047752, -0.0031284, -0.0007305, 0.0008599
8: -0.0067649, -0.0055611, -0.0067581, -0.0054764, -0.0006692, 0.0005686
9: -0.0035300, -0.0034261, -0.0035373, -0.0034267, -0.0000491, 0.0000577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001508
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001508
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083074, -0.0071962, -0.0006627, 0.0005087
1: -0.0052982, -0.0049939, -0.0052808, -0.0049675, -0.0001868, 0.0001434
2: -0.0005318, 0.0017140, -0.0004033, 0.0019083, -0.0013785, 0.0010582
3: 0.0015569, 0.0018541, 0.0015739, 0.0018798, -0.0001824, 0.0001400
4: 0.0048109, 0.0064893, 0.0046658, 0.0063933, -0.0007908, 0.0010302
5: 0.9968429, 0.9973091, 0.9968026, 0.9972826, -0.0002197, 0.0002862
6: 0.0050179, 0.0054412, 0.0049813, 0.0054170, -0.0001994, 0.0002598
7: -0.0046555, -0.0030760, -0.0047921, -0.0031663, -0.0007443, 0.0009696
8: -0.0067988, -0.0055695, -0.0067285, -0.0054631, -0.0007546, 0.0005793
9: -0.0035292, -0.0034232, -0.0035384, -0.0034292, -0.0000500, 0.0000651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001210, upper bound: 0.0001512
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001545
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083126, -0.0071999, -0.0006412, 0.0004953
1: -0.0052982, -0.0049939, -0.0052823, -0.0049686, -0.0001808, 0.0001396
2: -0.0005318, 0.0017140, -0.0004141, 0.0019006, -0.0013338, 0.0010302
3: 0.0015569, 0.0018541, 0.0015725, 0.0018788, -0.0001765, 0.0001363
4: 0.0048109, 0.0064893, 0.0046715, 0.0064013, -0.0007699, 0.0009968
5: 0.9968429, 0.9973091, 0.9968042, 0.9972847, -0.0002139, 0.0002769
6: 0.0050179, 0.0054412, 0.0049827, 0.0054190, -0.0001942, 0.0002514
7: -0.0046555, -0.0030760, -0.0047868, -0.0031588, -0.0007246, 0.0009381
8: -0.0067988, -0.0055695, -0.0067344, -0.0054673, -0.0007301, 0.0005639
9: -0.0035292, -0.0034232, -0.0035380, -0.0034287, -0.0000487, 0.0000630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001210, upper bound: 0.0001512
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001545
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083276, -0.0072053, -0.0006384, 0.0005144
1: -0.0052982, -0.0049939, -0.0052865, -0.0049701, -0.0001800, 0.0001450
2: -0.0005318, 0.0017140, -0.0004453, 0.0018893, -0.0013280, 0.0010701
3: 0.0015569, 0.0018541, 0.0015684, 0.0018773, -0.0001757, 0.0001416
4: 0.0048109, 0.0064893, 0.0046799, 0.0064247, -0.0007997, 0.0009925
5: 0.9968429, 0.9973091, 0.9968066, 0.9972911, -0.0002222, 0.0002757
6: 0.0050179, 0.0054412, 0.0049849, 0.0054249, -0.0002017, 0.0002503
7: -0.0046555, -0.0030760, -0.0047788, -0.0031368, -0.0007526, 0.0009340
8: -0.0067988, -0.0055695, -0.0067515, -0.0054735, -0.0007269, 0.0005858
9: -0.0035292, -0.0034232, -0.0035375, -0.0034273, -0.0000505, 0.0000627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001456, upper bound: 0.0001516
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001551
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083692, -0.0072896, -0.0083338, -0.0072078, -0.0006154, 0.0005003
1: -0.0052982, -0.0049939, -0.0052883, -0.0049708, -0.0001735, 0.0001411
2: -0.0005318, 0.0017140, -0.0004581, 0.0018842, -0.0012801, 0.0010408
3: 0.0015569, 0.0018541, 0.0015667, 0.0018766, -0.0001694, 0.0001377
4: 0.0048109, 0.0064893, 0.0046838, 0.0064342, -0.0007778, 0.0009567
5: 0.9968429, 0.9973091, 0.9968076, 0.9972938, -0.0002161, 0.0002658
6: 0.0050179, 0.0054412, 0.0049858, 0.0054273, -0.0001962, 0.0002413
7: -0.0046555, -0.0030760, -0.0047752, -0.0031278, -0.0007320, 0.0009003
8: -0.0067988, -0.0055695, -0.0067585, -0.0054763, -0.0007007, 0.0005697
9: -0.0035292, -0.0034232, -0.0035373, -0.0034266, -0.0000492, 0.0000605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001456, upper bound: 0.0001516
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001551
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0083122, -0.0072089, -0.0082879, -0.0071966, -0.0005070, 0.0004950
1: -0.0052822, -0.0049711, -0.0052753, -0.0049677, -0.0001429, 0.0001396
2: -0.0004132, 0.0018819, -0.0003627, 0.0019074, -0.0010546, 0.0010298
3: 0.0015726, 0.0018763, 0.0015793, 0.0018797, -0.0001396, 0.0001363
4: 0.0046855, 0.0064007, 0.0046664, 0.0063630, -0.0007696, 0.0007881
5: 0.9968081, 0.9972845, 0.9968027, 0.9972741, -0.0002138, 0.0002190
6: 0.0049863, 0.0054188, 0.0049815, 0.0054093, -0.0001941, 0.0001988
7: -0.0047736, -0.0031594, -0.0047916, -0.0031949, -0.0007243, 0.0007417
8: -0.0067339, -0.0054776, -0.0067063, -0.0054636, -0.0005773, 0.0005637
9: -0.0035372, -0.0034288, -0.0035384, -0.0034312, -0.0000486, 0.0000498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001514
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001514
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083164, -0.0072086, -0.0082989, -0.0071965, -0.0005120, 0.0005009
1: -0.0052834, -0.0049710, -0.0052784, -0.0049676, -0.0001444, 0.0001412
2: -0.0004220, 0.0018824, -0.0003855, 0.0019076, -0.0010651, 0.0010420
3: 0.0015715, 0.0018764, 0.0015763, 0.0018797, -0.0001410, 0.0001379
4: 0.0046851, 0.0064072, 0.0046663, 0.0063800, -0.0007788, 0.0007960
5: 0.9968079, 0.9972863, 0.9968027, 0.9972787, -0.0002164, 0.0002212
6: 0.0049862, 0.0054205, 0.0049814, 0.0054136, -0.0001964, 0.0002007
7: -0.0047740, -0.0031532, -0.0047917, -0.0031789, -0.0007329, 0.0007491
8: -0.0067387, -0.0054773, -0.0067187, -0.0054635, -0.0005831, 0.0005704
9: -0.0035372, -0.0034284, -0.0035384, -0.0034301, -0.0000492, 0.0000503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001551, upper bound: 0.0001516
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001551, upper bound: 0.0001516
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0083122, -0.0072089, -0.0082938, -0.0071985, -0.0004801, 0.0004816
1: -0.0052822, -0.0049711, -0.0052770, -0.0049682, -0.0001354, 0.0001358
2: -0.0004132, 0.0018819, -0.0003750, 0.0019036, -0.0009987, 0.0010019
3: 0.0015726, 0.0018763, 0.0015777, 0.0018792, -0.0001322, 0.0001326
4: 0.0046855, 0.0064007, 0.0046693, 0.0063721, -0.0007487, 0.0007464
5: 0.9968081, 0.9972845, 0.9968035, 0.9972765, -0.0002080, 0.0002074
6: 0.0049863, 0.0054188, 0.0049822, 0.0054116, -0.0001888, 0.0001882
7: -0.0047736, -0.0031594, -0.0047888, -0.0031863, -0.0007046, 0.0007024
8: -0.0067339, -0.0054776, -0.0067130, -0.0054657, -0.0005467, 0.0005484
9: -0.0035372, -0.0034288, -0.0035382, -0.0034306, -0.0000473, 0.0000472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001515
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001514
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0083164, -0.0072086, -0.0083034, -0.0072003, -0.0004858, 0.0004866
1: -0.0052834, -0.0049710, -0.0052797, -0.0049687, -0.0001370, 0.0001372
2: -0.0004220, 0.0018824, -0.0003949, 0.0018998, -0.0010105, 0.0010122
3: 0.0015715, 0.0018764, 0.0015750, 0.0018787, -0.0001337, 0.0001340
4: 0.0046851, 0.0064072, 0.0046720, 0.0063870, -0.0007565, 0.0007552
5: 0.9968079, 0.9972863, 0.9968043, 0.9972807, -0.0002102, 0.0002098
6: 0.0049862, 0.0054205, 0.0049829, 0.0054154, -0.0001908, 0.0001904
7: -0.0047740, -0.0031532, -0.0047862, -0.0031723, -0.0007119, 0.0007107
8: -0.0067387, -0.0054773, -0.0067239, -0.0054677, -0.0005531, 0.0005541
9: -0.0035372, -0.0034284, -0.0035380, -0.0034296, -0.0000478, 0.0000477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001549, upper bound: 0.0001516
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001549, upper bound: 0.0001516
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0082955, -0.0072006, -0.0083276, -0.0072053, -0.0005038, 0.0005441
1: -0.0052775, -0.0049688, -0.0052865, -0.0049701, -0.0001420, 0.0001534
2: -0.0003786, 0.0018991, -0.0004453, 0.0018893, -0.0010479, 0.0011318
3: 0.0015772, 0.0018786, 0.0015684, 0.0018773, -0.0001387, 0.0001498
4: 0.0046726, 0.0063748, 0.0046799, 0.0064247, -0.0008458, 0.0007831
5: 0.9968045, 0.9972773, 0.9968066, 0.9972911, -0.0002350, 0.0002176
6: 0.0049830, 0.0054123, 0.0049849, 0.0054249, -0.0002133, 0.0001975
7: -0.0047857, -0.0031838, -0.0047788, -0.0031368, -0.0007960, 0.0007370
8: -0.0067149, -0.0054682, -0.0067515, -0.0054735, -0.0005736, 0.0006195
9: -0.0035380, -0.0034304, -0.0035375, -0.0034273, -0.0000535, 0.0000495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001584, upper bound: 0.0001487
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001585, upper bound: 0.0001494
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0082955, -0.0072006, -0.0083338, -0.0072078, -0.0004817, 0.0005315
1: -0.0052775, -0.0049688, -0.0052883, -0.0049708, -0.0001358, 0.0001498
2: -0.0003786, 0.0018991, -0.0004581, 0.0018842, -0.0010020, 0.0011056
3: 0.0015772, 0.0018786, 0.0015667, 0.0018766, -0.0001326, 0.0001463
4: 0.0046726, 0.0063748, 0.0046838, 0.0064342, -0.0008262, 0.0007488
5: 0.9968045, 0.9972773, 0.9968076, 0.9972938, -0.0002296, 0.0002080
6: 0.0049830, 0.0054123, 0.0049858, 0.0054273, -0.0002084, 0.0001888
7: -0.0047857, -0.0031838, -0.0047752, -0.0031278, -0.0007776, 0.0007047
8: -0.0067149, -0.0054682, -0.0067585, -0.0054763, -0.0005485, 0.0006052
9: -0.0035380, -0.0034304, -0.0035373, -0.0034266, -0.0000522, 0.0000473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001584, upper bound: 0.0001487
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001585, upper bound: 0.0001495
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0083166, -0.0072086, -0.0083276, -0.0072053, -0.0005119, 0.0005229
1: -0.0052834, -0.0049710, -0.0052865, -0.0049701, -0.0001443, 0.0001474
2: -0.0004223, 0.0018824, -0.0004453, 0.0018893, -0.0010648, 0.0010878
3: 0.0015714, 0.0018764, 0.0015684, 0.0018773, -0.0001409, 0.0001439
4: 0.0046850, 0.0064075, 0.0046799, 0.0064247, -0.0008129, 0.0007958
5: 0.9968079, 0.9972864, 0.9968066, 0.9972911, -0.0002259, 0.0002211
6: 0.0049862, 0.0054205, 0.0049849, 0.0054249, -0.0002050, 0.0002007
7: -0.0047740, -0.0031530, -0.0047788, -0.0031368, -0.0007650, 0.0007489
8: -0.0067389, -0.0054773, -0.0067515, -0.0054735, -0.0005829, 0.0005954
9: -0.0035372, -0.0034283, -0.0035375, -0.0034273, -0.0000514, 0.0000503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001514
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001516
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0083166, -0.0072086, -0.0083338, -0.0072078, -0.0004905, 0.0005098
1: -0.0052834, -0.0049710, -0.0052883, -0.0049708, -0.0001383, 0.0001437
2: -0.0004223, 0.0018824, -0.0004581, 0.0018842, -0.0010203, 0.0010604
3: 0.0015714, 0.0018764, 0.0015667, 0.0018766, -0.0001350, 0.0001403
4: 0.0046850, 0.0064075, 0.0046838, 0.0064342, -0.0007925, 0.0007625
5: 0.9968079, 0.9972864, 0.9968076, 0.9972938, -0.0002202, 0.0002119
6: 0.0049862, 0.0054205, 0.0049858, 0.0054273, -0.0001999, 0.0001923
7: -0.0047740, -0.0031530, -0.0047752, -0.0031278, -0.0007458, 0.0007176
8: -0.0067389, -0.0054773, -0.0067585, -0.0054763, -0.0005585, 0.0005805
9: -0.0035372, -0.0034283, -0.0035373, -0.0034266, -0.0000501, 0.0000482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001514
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001516
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0083418, -0.0072724, -0.0083270, -0.0072055, -0.0006142, 0.0005338
1: -0.0052905, -0.0049890, -0.0052863, -0.0049702, -0.0001732, 0.0001505
2: -0.0004748, 0.0017497, -0.0004440, 0.0018889, -0.0012776, 0.0011105
3: 0.0015645, 0.0018588, 0.0015685, 0.0018773, -0.0001691, 0.0001470
4: 0.0047843, 0.0064467, 0.0046803, 0.0064237, -0.0008299, 0.0009548
5: 0.9968355, 0.9972973, 0.9968066, 0.9972909, -0.0002306, 0.0002653
6: 0.0050112, 0.0054304, 0.0049850, 0.0054246, -0.0002093, 0.0002408
7: -0.0046806, -0.0031161, -0.0047785, -0.0031378, -0.0007810, 0.0008986
8: -0.0067676, -0.0055499, -0.0067507, -0.0054738, -0.0006994, 0.0006079
9: -0.0035309, -0.0034259, -0.0035375, -0.0034273, -0.0000524, 0.0000603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001534
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001534
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0083418, -0.0072724, -0.0083322, -0.0072080, -0.0005900, 0.0005224
1: -0.0052905, -0.0049890, -0.0052878, -0.0049709, -0.0001663, 0.0001473
2: -0.0004748, 0.0017497, -0.0004549, 0.0018837, -0.0012273, 0.0010867
3: 0.0015645, 0.0018588, 0.0015671, 0.0018766, -0.0001624, 0.0001438
4: 0.0047843, 0.0064467, 0.0046841, 0.0064318, -0.0008121, 0.0009172
5: 0.9968355, 0.9972973, 0.9968076, 0.9972932, -0.0002256, 0.0002548
6: 0.0050112, 0.0054304, 0.0049859, 0.0054267, -0.0002048, 0.0002313
7: -0.0046806, -0.0031161, -0.0047748, -0.0031301, -0.0007643, 0.0008632
8: -0.0067676, -0.0055499, -0.0067567, -0.0054766, -0.0006718, 0.0005949
9: -0.0035309, -0.0034259, -0.0035372, -0.0034268, -0.0000513, 0.0000580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001454, upper bound: 0.0001534
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001454, upper bound: 0.0001534
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0083717, -0.0072796, -0.0083067, -0.0071964, -0.0006647, 0.0005260
1: -0.0052989, -0.0049911, -0.0052806, -0.0049676, -0.0001874, 0.0001483
2: -0.0005369, 0.0017347, -0.0004018, 0.0019078, -0.0013828, 0.0010942
3: 0.0015562, 0.0018569, 0.0015741, 0.0018798, -0.0001830, 0.0001448
4: 0.0047954, 0.0064931, 0.0046661, 0.0063921, -0.0008178, 0.0010334
5: 0.9968385, 0.9973102, 0.9968026, 0.9972821, -0.0002272, 0.0002871
6: 0.0050140, 0.0054421, 0.0049814, 0.0054167, -0.0002062, 0.0002606
7: -0.0046701, -0.0030724, -0.0047919, -0.0031675, -0.0007696, 0.0009725
8: -0.0068016, -0.0055581, -0.0067276, -0.0054634, -0.0007569, 0.0005990
9: -0.0035302, -0.0034229, -0.0035384, -0.0034293, -0.0000517, 0.0000653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001208, upper bound: 0.0001531
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001561
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0083717, -0.0072796, -0.0083117, -0.0072001, -0.0006432, 0.0005162
1: -0.0052989, -0.0049911, -0.0052820, -0.0049686, -0.0001814, 0.0001455
2: -0.0005369, 0.0017347, -0.0004122, 0.0019002, -0.0013381, 0.0010739
3: 0.0015562, 0.0018569, 0.0015727, 0.0018788, -0.0001771, 0.0001421
4: 0.0047954, 0.0064931, 0.0046718, 0.0064000, -0.0008025, 0.0010000
5: 0.9968385, 0.9973102, 0.9968042, 0.9972843, -0.0002230, 0.0002778
6: 0.0050140, 0.0054421, 0.0049828, 0.0054186, -0.0002024, 0.0002522
7: -0.0046701, -0.0030724, -0.0047864, -0.0031601, -0.0007553, 0.0009411
8: -0.0068016, -0.0055581, -0.0067334, -0.0054676, -0.0007325, 0.0005878
9: -0.0035302, -0.0034229, -0.0035380, -0.0034288, -0.0000507, 0.0000632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001208, upper bound: 0.0001531
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001561
time: 0.74 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.51 seconds
IS_A1_B1_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001449
IS_A1_B1_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001449
IS_A1_B1_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001449
IS_A1_B1_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001449
IS_A1_B1_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001449
IS_A1_B1_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001449
IS_A1_B1_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001449
IS_A1_B1_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001449
IS_A1_B1_A1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001405
IS_A1_B1_A1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001405
IS_A1_B1_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001411, upper bound: 0.0001439
IS_A1_B1_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001411, upper bound: 0.0001439
IS_A1_B1_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001472, upper bound: 0.0001477
IS_A1_B1_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001472, upper bound: 0.0001477
IS_A1_B1_A1_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001526, upper bound: 0.0001215
IS_A1_B1_A1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001526, upper bound: 0.0001215
IS_A1_B1_A1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001547, upper bound: 0.0001405
IS_A1_B1_A1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001547, upper bound: 0.0001405
IS_A1_B1_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001439
IS_A1_B1_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001439
IS_A1_B1_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001529, upper bound: 0.0001473
IS_A1_B1_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001529, upper bound: 0.0001472
IS_A1_B1_A2_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001471
IS_A1_B1_A2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001471
IS_A1_B1_A2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001471
IS_A1_B1_A2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001471
IS_A1_B1_A2_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001471
IS_A1_B1_A2_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001471
IS_A1_B1_A2_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001471
IS_A1_B1_A2_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001471
IS_A1_B1_A2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001434
IS_A1_B1_A2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001434
IS_A1_B1_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001391, upper bound: 0.0001463
IS_A1_B1_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001391, upper bound: 0.0001463
IS_A1_B1_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001502
IS_A1_B1_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001503
IS_A1_B1_A2_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001521, upper bound: 0.0001208
IS_A1_B1_A2_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001521, upper bound: 0.0001208
IS_A1_B1_A2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001434
IS_A1_B1_A2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001434
IS_A1_B1_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001463
IS_A1_B1_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001463
IS_A1_B1_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001542, upper bound: 0.0001500
IS_A1_B1_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001542, upper bound: 0.0001501
IS_A1_B2_B1_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001517
IS_A1_B2_B1_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001517
IS_A1_B2_B1_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001531
IS_A1_B2_B1_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001531
IS_A1_B2_B1_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001280, upper bound: 0.0001453
IS_A1_B2_B1_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001280, upper bound: 0.0001453
IS_A1_B2_B1_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001417, upper bound: 0.0001539
IS_A1_B2_B1_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001417, upper bound: 0.0001555
IS_A1_B2_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001429, upper bound: 0.0001503
IS_A1_B2_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001429, upper bound: 0.0001506
IS_A1_B2_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001460, upper bound: 0.0001546
IS_A1_B2_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001460, upper bound: 0.0001562
IS_A1_B2_B1_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001485
IS_A1_B2_B1_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001485
IS_A1_B2_B1_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001493
IS_A1_B2_B1_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001494
IS_A1_B2_B1_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001501
IS_A1_B2_B1_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001501
IS_A1_B2_B1_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001522
IS_A1_B2_B1_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001522
IS_A1_B2_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001524, upper bound: 0.0001518
IS_A1_B2_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001524, upper bound: 0.0001536
IS_A1_B2_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001541, upper bound: 0.0001519
IS_A1_B2_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001541, upper bound: 0.0001537
IS_A1_B2_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001504
IS_A1_B2_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001539, upper bound: 0.0001512
IS_A1_B2_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001541, upper bound: 0.0001519
IS_A1_B2_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001541, upper bound: 0.0001537
IS_A1_B2_B2_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001507
IS_A1_B2_B2_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001531
IS_A1_B2_B2_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001508
IS_A1_B2_B2_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001531
IS_A1_B2_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001292, upper bound: 0.0001451
IS_A1_B2_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001292, upper bound: 0.0001451
IS_A1_B2_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001555
IS_A1_B2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001555
IS_A1_B2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001451, upper bound: 0.0001506
IS_A1_B2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001451, upper bound: 0.0001506
IS_A1_B2_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001561
IS_A1_B2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001561
IS_A1_B2_B2_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001482
IS_A1_B2_B2_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001493
IS_A1_B2_B2_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001482
IS_A1_B2_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001493
IS_A1_B2_B2_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001490
IS_A1_B2_B2_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001522
IS_A1_B2_B2_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001490
IS_A1_B2_B2_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001566, upper bound: 0.0001522
IS_A1_B2_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001535
IS_A1_B2_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001535
IS_A1_B2_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001537
IS_A1_B2_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001537
IS_A1_B2_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001513
IS_A1_B2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001565, upper bound: 0.0001512
IS_A1_B2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001537
IS_A1_B2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001537
IS_A2_B1_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001462
IS_A2_B1_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001420, upper bound: 0.0001462
IS_A2_B1_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001427, upper bound: 0.0001465
IS_A2_B1_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001427, upper bound: 0.0001465
IS_A2_B1_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001462
IS_A2_B1_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001462
IS_A2_B1_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001465
IS_A2_B1_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001465
IS_A2_B1_A1_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001375, upper bound: 0.0001158
IS_A2_B1_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001418
IS_A2_B1_A1_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001375, upper bound: 0.0001158
IS_A2_B1_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001420
IS_A2_B1_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001401, upper bound: 0.0001448
IS_A2_B1_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001484
IS_A2_B1_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001401, upper bound: 0.0001449
IS_A2_B1_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001486
IS_A2_B1_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001520, upper bound: 0.0001223
IS_A2_B1_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001417
IS_A2_B1_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001520, upper bound: 0.0001223
IS_A2_B1_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001555, upper bound: 0.0001420
IS_A2_B1_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001448
IS_A2_B1_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001543, upper bound: 0.0001481
IS_A2_B1_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001449
IS_A2_B1_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001543, upper bound: 0.0001482
IS_A2_B1_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001484
IS_A2_B1_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001405, upper bound: 0.0001484
IS_A2_B1_A2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001411, upper bound: 0.0001485
IS_A2_B1_A2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001411, upper bound: 0.0001485
IS_A2_B1_A2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001507, upper bound: 0.0001484
IS_A2_B1_A2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001507, upper bound: 0.0001484
IS_A2_B1_A2_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001485
IS_A2_B1_A2_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001506, upper bound: 0.0001485
IS_A2_B1_A2_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001360, upper bound: 0.0001145
IS_A2_B1_A2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001436
IS_A2_B1_A2_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001360, upper bound: 0.0001145
IS_A2_B1_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001471, upper bound: 0.0001436
IS_A2_B1_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001391, upper bound: 0.0001471
IS_A2_B1_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001511
IS_A2_B1_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001391, upper bound: 0.0001472
IS_A2_B1_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001485, upper bound: 0.0001513
IS_A2_B1_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001451, upper bound: 0.0001292
IS_A2_B1_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001554, upper bound: 0.0001436
IS_A2_B1_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001451, upper bound: 0.0001292
IS_A2_B1_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001554, upper bound: 0.0001436
IS_A2_B1_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001471
IS_A2_B1_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001542, upper bound: 0.0001510
IS_A2_B1_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001480, upper bound: 0.0001472
IS_A2_B1_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001542, upper bound: 0.0001511
IS_A2_B2_A1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001456, upper bound: 0.0001508
IS_A2_B2_A1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001456, upper bound: 0.0001508
IS_A2_B2_A1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001508
IS_A2_B2_A1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001462, upper bound: 0.0001508
IS_A2_B2_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001210, upper bound: 0.0001512
IS_A2_B2_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001545
IS_A2_B2_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001210, upper bound: 0.0001512
IS_A2_B2_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001545
IS_A2_B2_A1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001456, upper bound: 0.0001516
IS_A2_B2_A1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001551
IS_A2_B2_A1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001456, upper bound: 0.0001516
IS_A2_B2_A1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001484, upper bound: 0.0001551
IS_A2_B2_A1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001514
IS_A2_B2_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001517, upper bound: 0.0001514
IS_A2_B2_A1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001551, upper bound: 0.0001516
IS_A2_B2_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001551, upper bound: 0.0001516
IS_A2_B2_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001515
IS_A2_B2_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001516, upper bound: 0.0001514
IS_A2_B2_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001549, upper bound: 0.0001516
IS_A2_B2_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001549, upper bound: 0.0001516
IS_A2_B2_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001584, upper bound: 0.0001487
IS_A2_B2_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001585, upper bound: 0.0001494
IS_A2_B2_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001584, upper bound: 0.0001487
IS_A2_B2_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001585, upper bound: 0.0001495
IS_A2_B2_A1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001514
IS_A2_B2_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001516
IS_A2_B2_A1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001514
IS_A2_B2_A1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001516
IS_A2_B2_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001534
IS_A2_B2_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001449, upper bound: 0.0001534
IS_A2_B2_A2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001454, upper bound: 0.0001534
IS_A2_B2_A2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001454, upper bound: 0.0001534
IS_A2_B2_A2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001208, upper bound: 0.0001531
IS_A2_B2_A2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001561
IS_A2_B2_A2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001208, upper bound: 0.0001531
IS_A2_B2_A2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.51
Output dim: 5, lower bound: -0.0001434, upper bound: 0.0001561
IS_A2_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001593
IS_A2_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001461, upper bound: 0.0001593
IS_A2_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001544
IS_A2_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001545
IS_A2_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001525, upper bound: 0.0001545
IS_A2_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001567, upper bound: 0.0001546
IS_A2_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001547
IS_A2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001546
IS_A2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001572
IS_A2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 5, lower bound: -0.0001592, upper bound: 0.0001572

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.03 + 597.32 = 600.35 seconds
