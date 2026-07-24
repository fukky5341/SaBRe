## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00399952


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0152226, 0.0179737, 0.0152226, 0.0179737, -0.0023687, 0.0023687)
1: (-0.0018159, 0.0001410, -0.0018159, 0.0001410, -0.0017294, 0.0017294)
2: (0.0036573, 0.0045475, 0.0036573, 0.0045475, -0.0007586, 0.0007586)
3: (0.0013559, 0.0027948, 0.0013559, 0.0027948, -0.0011189, 0.0011189)
4: (-0.0045978, -0.0025822, -0.0045978, -0.0025822, -0.0014630, 0.0014630)
5: (-0.0002342, 0.0009479, -0.0002342, 0.0009479, -0.0010524, 0.0010524)
6: (-0.0050397, -0.0015669, -0.0050397, -0.0015669, -0.0024895, 0.0024895)
7: (-0.0225769, -0.0109873, -0.0225769, -0.0109873, -0.0084705, 0.0084705)
8: (0.9749432, 0.9855437, 0.9749432, 0.9855437, -0.0080937, 0.0080937)
9: (-0.0005370, 0.0070756, -0.0005370, 0.0070756, -0.0056105, 0.0056105)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.68 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0055877, upper bound: 0.0055877

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053411, upper bound: 0.0054208
time: 0.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054208, upper bound: 0.0054208
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 8, lower bound: -0.0053411, upper bound: 0.0054208
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 8, lower bound: -0.0054208, upper bound: 0.0054208

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0152517, 0.0179436, 0.0152308, 0.0179650, -0.0023312, 0.0023139
1: -0.0017912, 0.0001279, -0.0018091, 0.0001373, -0.0017020, 0.0016947
2: 0.0036682, 0.0045389, 0.0036604, 0.0045450, -0.0007405, 0.0007467
3: 0.0013593, 0.0027549, 0.0013569, 0.0027831, -0.0011021, 0.0010765
4: -0.0045331, -0.0025932, -0.0045796, -0.0025853, -0.0013908, 0.0014331
5: -0.0002289, 0.0009323, -0.0002327, 0.0009436, -0.0010323, 0.0010358
6: -0.0050301, -0.0017035, -0.0050369, -0.0016062, -0.0024346, 0.0023403
7: -0.0222142, -0.0110525, -0.0224744, -0.0110055, -0.0080636, 0.0083001
8: 0.9752246, 0.9854726, 0.9750236, 0.9855240, -0.0077618, 0.0079452
9: -0.0004928, 0.0068462, -0.0005247, 0.0070104, -0.0054998, 0.0053491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053411, upper bound: 0.0053411
time: 0.74 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053411, upper bound: 0.0054208
time: 0.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0150493, 0.0179474, 0.0152286, 0.0179649, -0.0025528, 0.0023224
1: -0.0019266, 0.0001307, -0.0018109, 0.0001372, -0.0018505, 0.0017035
2: 0.0036667, 0.0046039, 0.0036606, 0.0045456, -0.0007436, 0.0008193
3: 0.0013325, 0.0027822, 0.0013569, 0.0027853, -0.0011485, 0.0011033
4: -0.0045503, -0.0024892, -0.0045822, -0.0025843, -0.0014065, 0.0015814
5: -0.0002305, 0.0010123, -0.0002326, 0.0009448, -0.0010399, 0.0011228
6: -0.0051277, -0.0016603, -0.0050371, -0.0015993, -0.0025968, 0.0023814
7: -0.0223090, -0.0104364, -0.0224890, -0.0109998, -0.0081520, 0.0091687
8: 0.9751463, 0.9861301, 0.9750102, 0.9855298, -0.0078265, 0.0088013
9: -0.0009132, 0.0069041, -0.0005286, 0.0070196, -0.0060829, 0.0054057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052108, upper bound: 0.0051897
time: 0.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052826, upper bound: 0.0052826
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.84 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 8, lower bound: -0.0053411, upper bound: 0.0053411
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 8, lower bound: -0.0053411, upper bound: 0.0054208
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 8, lower bound: -0.0052108, upper bound: 0.0051897
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 8, lower bound: -0.0052826, upper bound: 0.0052826

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0152517, 0.0179436, 0.0152517, 0.0179436, -0.0022971, 0.0022971
1: -0.0017912, 0.0001279, -0.0017912, 0.0001279, -0.0016814, 0.0016814
2: 0.0036682, 0.0045389, 0.0036682, 0.0045389, -0.0007353, 0.0007353
3: 0.0013593, 0.0027549, 0.0013593, 0.0027549, -0.0010734, 0.0010734
4: -0.0045331, -0.0025932, -0.0045331, -0.0025932, -0.0013843, 0.0013843
5: -0.0002289, 0.0009323, -0.0002289, 0.0009323, -0.0010239, 0.0010239
6: -0.0050301, -0.0017035, -0.0050301, -0.0017035, -0.0023319, 0.0023319
7: -0.0222142, -0.0110525, -0.0222142, -0.0110525, -0.0080257, 0.0080256
8: 0.9752246, 0.9854726, 0.9752246, 0.9854726, -0.0077221, 0.0077221
9: -0.0004928, 0.0068462, -0.0004928, 0.0068462, -0.0053237, 0.0053237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051054, upper bound: 0.0051537
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051753, upper bound: 0.0052088
time: 0.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0152517, 0.0179436, 0.0150493, 0.0179474, -0.0023080, 0.0025149
1: -0.0017912, 0.0001279, -0.0019266, 0.0001307, -0.0016837, 0.0018282
2: 0.0036682, 0.0045389, 0.0036667, 0.0046039, -0.0008066, 0.0007397
3: 0.0013593, 0.0027549, 0.0013325, 0.0027822, -0.0011014, 0.0011167
4: -0.0045331, -0.0025932, -0.0045503, -0.0024892, -0.0015234, 0.0014215
5: -0.0002289, 0.0009323, -0.0002305, 0.0010123, -0.0011106, 0.0010252
6: -0.0050301, -0.0017035, -0.0051277, -0.0016603, -0.0024059, 0.0024774
7: -0.0222142, -0.0110525, -0.0223090, -0.0104364, -0.0088414, 0.0082351
8: 0.9752246, 0.9854726, 0.9751463, 0.9861301, -0.0085350, 0.0078886
9: -0.0004928, 0.0068462, -0.0009132, 0.0069041, -0.0054584, 0.0058719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051054, upper bound: 0.0052108
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051753, upper bound: 0.0052826
time: 0.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150631, 0.0179434, 0.0151090, 0.0179529, -0.0025178, 0.0024161
1: -0.0019150, 0.0001302, -0.0018800, 0.0001420, -0.0018420, 0.0017679
2: 0.0036687, 0.0045996, 0.0036667, 0.0045859, -0.0007740, 0.0008058
3: 0.0013330, 0.0027515, 0.0013165, 0.0027099, -0.0010645, 0.0010842
4: -0.0045207, -0.0024920, -0.0044946, -0.0024748, -0.0014469, 0.0014706
5: -0.0002301, 0.0010047, -0.0002386, 0.0009812, -0.0010777, 0.0011215
6: -0.0051263, -0.0017377, -0.0051932, -0.0018232, -0.0023193, 0.0023490
7: -0.0221444, -0.0104542, -0.0220041, -0.0103677, -0.0083998, 0.0085517
8: 0.9752634, 0.9861060, 0.9753661, 0.9861105, -0.0081062, 0.0083321
9: -0.0009005, 0.0068015, -0.0009464, 0.0067133, -0.0056920, 0.0055792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051897
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051897
time: 0.85 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0150493, 0.0179474, 0.0152400, 0.0179600, -0.0025247, 0.0023126
1: -0.0019266, 0.0001307, -0.0018012, 0.0001368, -0.0018425, 0.0016958
2: 0.0036667, 0.0046039, 0.0036630, 0.0045422, -0.0007405, 0.0008087
3: 0.0013325, 0.0027822, 0.0013573, 0.0027589, -0.0010708, 0.0011027
4: -0.0045503, -0.0024892, -0.0045518, -0.0025866, -0.0014039, 0.0015007
5: -0.0002305, 0.0010123, -0.0002323, 0.0009386, -0.0010350, 0.0011205
6: -0.0051277, -0.0016603, -0.0050355, -0.0016777, -0.0023926, 0.0023795
7: -0.0223090, -0.0104364, -0.0223199, -0.0110145, -0.0081359, 0.0087174
8: 0.9751463, 0.9861301, 0.9751369, 0.9855096, -0.0078058, 0.0084551
9: -0.0009132, 0.0069041, -0.0005178, 0.0069128, -0.0057967, 0.0053942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0052108
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0052826
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0051054, upper bound: 0.0051537
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0051753, upper bound: 0.0052088
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0051054, upper bound: 0.0052108
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0051753, upper bound: 0.0052826
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051897
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051897
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0052108
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0052826

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0151338, 0.0179316, 0.0152655, 0.0179396, -0.0023887, 0.0022615
1: -0.0018588, 0.0001328, -0.0017792, 0.0001274, -0.0017438, 0.0016719
2: 0.0036743, 0.0045780, 0.0036702, 0.0045346, -0.0007216, 0.0007650
3: 0.0013190, 0.0026814, 0.0013598, 0.0027245, -0.0010547, 0.0009933
4: -0.0044474, -0.0024844, -0.0045048, -0.0025960, -0.0012734, 0.0014246
5: -0.0002349, 0.0009679, -0.0002286, 0.0009247, -0.0010222, 0.0010607
6: -0.0051863, -0.0019256, -0.0050286, -0.0017794, -0.0022986, 0.0020570
7: -0.0217401, -0.0104245, -0.0220574, -0.0110707, -0.0074095, 0.0082710
8: 0.9755719, 0.9860469, 0.9753393, 0.9854469, -0.0072610, 0.0079987
9: -0.0009080, 0.0065455, -0.0004794, 0.0067461, -0.0054955, 0.0049362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051548, upper bound: 0.0051548
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051548, upper bound: 0.0051706
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0152633, 0.0179387, 0.0152517, 0.0179436, -0.0022868, 0.0022725
1: -0.0017811, 0.0001274, -0.0017912, 0.0001279, -0.0016733, 0.0016735
2: 0.0036707, 0.0045353, 0.0036682, 0.0045389, -0.0007260, 0.0007321
3: 0.0013598, 0.0027284, 0.0013593, 0.0027549, -0.0010728, 0.0009982
4: -0.0045019, -0.0025958, -0.0045331, -0.0025932, -0.0013075, 0.0013815
5: -0.0002286, 0.0009259, -0.0002289, 0.0009323, -0.0010215, 0.0010189
6: -0.0050285, -0.0017825, -0.0050301, -0.0017035, -0.0023298, 0.0021338
7: -0.0220424, -0.0110689, -0.0222142, -0.0110525, -0.0076001, 0.0080084
8: 0.9753515, 0.9854505, 0.9752246, 0.9854726, -0.0074037, 0.0077003
9: -0.0004809, 0.0067372, -0.0004928, 0.0068462, -0.0053115, 0.0050570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051706, upper bound: 0.0051555
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051706, upper bound: 0.0052261
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0151338, 0.0179316, 0.0150631, 0.0179434, -0.0023994, 0.0024798
1: -0.0018588, 0.0001328, -0.0019150, 0.0001302, -0.0017459, 0.0018190
2: 0.0036743, 0.0045780, 0.0036687, 0.0045996, -0.0007930, 0.0007693
3: 0.0013190, 0.0026814, 0.0013330, 0.0027515, -0.0010817, 0.0010367
4: -0.0044474, -0.0024844, -0.0045207, -0.0024920, -0.0014126, 0.0014608
5: -0.0002349, 0.0009679, -0.0002301, 0.0010047, -0.0011089, 0.0010619
6: -0.0051863, -0.0019256, -0.0051263, -0.0017377, -0.0023720, 0.0022026
7: -0.0217401, -0.0104245, -0.0221444, -0.0104542, -0.0082261, 0.0084752
8: 0.9755719, 0.9860469, 0.9752634, 0.9861060, -0.0080745, 0.0081591
9: -0.0009080, 0.0065455, -0.0009005, 0.0068015, -0.0056258, 0.0054849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051031, upper bound: 0.0051897
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051031, upper bound: 0.0052108
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0152633, 0.0179387, 0.0150493, 0.0179474, -0.0022977, 0.0024878
1: -0.0017811, 0.0001274, -0.0019266, 0.0001307, -0.0016756, 0.0018205
2: 0.0036707, 0.0045353, 0.0036667, 0.0046039, -0.0007962, 0.0007365
3: 0.0013598, 0.0027284, 0.0013325, 0.0027822, -0.0011008, 0.0010411
4: -0.0045019, -0.0025958, -0.0045503, -0.0024892, -0.0014428, 0.0014187
5: -0.0002286, 0.0009259, -0.0002305, 0.0010123, -0.0011084, 0.0010201
6: -0.0050285, -0.0017825, -0.0051277, -0.0016603, -0.0024038, 0.0022755
7: -0.0220424, -0.0110689, -0.0223090, -0.0104364, -0.0083923, 0.0082178
8: 0.9753515, 0.9854505, 0.9751463, 0.9861301, -0.0081977, 0.0078667
9: -0.0004809, 0.0067372, -0.0009132, 0.0069041, -0.0054462, 0.0055896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051107, upper bound: 0.0051897
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051107, upper bound: 0.0052826
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0149418, 0.0179354, 0.0151090, 0.0179529, -0.0026338, 0.0024072
1: -0.0019901, 0.0001354, -0.0018800, 0.0001420, -0.0019210, 0.0017731
2: 0.0036728, 0.0046402, 0.0036667, 0.0045859, -0.0007694, 0.0008433
3: 0.0012892, 0.0027073, 0.0013165, 0.0027099, -0.0010792, 0.0010392
4: -0.0044619, -0.0023872, -0.0044946, -0.0024748, -0.0013750, 0.0015447
5: -0.0002364, 0.0010451, -0.0002386, 0.0009812, -0.0010832, 0.0011672
6: -0.0052794, -0.0018840, -0.0051932, -0.0018232, -0.0023839, 0.0021697
7: -0.0218183, -0.0098512, -0.0220041, -0.0103677, -0.0079996, 0.0089847
8: 0.9755049, 0.9866629, 0.9753661, 0.9861105, -0.0078173, 0.0087560
9: -0.0012963, 0.0065963, -0.0009464, 0.0067133, -0.0059818, 0.0053274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051053
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051053
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0150606, 0.0179425, 0.0151090, 0.0179529, -0.0025196, 0.0024170
1: -0.0019171, 0.0001303, -0.0018800, 0.0001420, -0.0018435, 0.0017670
2: 0.0036691, 0.0046004, 0.0036667, 0.0045859, -0.0007746, 0.0008064
3: 0.0013330, 0.0027561, 0.0013165, 0.0027099, -0.0010645, 0.0010990
4: -0.0045209, -0.0024917, -0.0044946, -0.0024748, -0.0014614, 0.0014707
5: -0.0002302, 0.0010060, -0.0002386, 0.0009812, -0.0010771, 0.0011225
6: -0.0051262, -0.0017378, -0.0051932, -0.0018232, -0.0023193, 0.0023920
7: -0.0221432, -0.0104516, -0.0220041, -0.0103677, -0.0084776, 0.0085524
8: 0.9752718, 0.9861102, 0.9753661, 0.9861105, -0.0081622, 0.0083345
9: -0.0009023, 0.0067982, -0.0009464, 0.0067133, -0.0056928, 0.0056267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051054
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051054
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0149418, 0.0179354, 0.0152400, 0.0179600, -0.0026438, 0.0022890
1: -0.0019901, 0.0001354, -0.0018012, 0.0001368, -0.0019143, 0.0016957
2: 0.0036728, 0.0046402, 0.0036630, 0.0045422, -0.0007305, 0.0008485
3: 0.0012892, 0.0027073, 0.0013573, 0.0027589, -0.0011428, 0.0010223
4: -0.0044619, -0.0023872, -0.0045518, -0.0025866, -0.0012959, 0.0016309
5: -0.0002364, 0.0010451, -0.0002323, 0.0009386, -0.0010385, 0.0011601
6: -0.0052794, -0.0018840, -0.0050355, -0.0016777, -0.0026068, 0.0021041
7: -0.0218183, -0.0098512, -0.0223199, -0.0110145, -0.0075370, 0.0094619
8: 0.9755049, 0.9866629, 0.9751369, 0.9855096, -0.0073652, 0.0091068
9: -0.0012963, 0.0065963, -0.0005178, 0.0069128, -0.0062824, 0.0050182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051107
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051107
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0150606, 0.0179425, 0.0152400, 0.0179600, -0.0025148, 0.0022888
1: -0.0019171, 0.0001303, -0.0018012, 0.0001368, -0.0018346, 0.0016889
2: 0.0036691, 0.0046004, 0.0036630, 0.0045422, -0.0007311, 0.0008056
3: 0.0013330, 0.0027561, 0.0013573, 0.0027589, -0.0010702, 0.0010270
4: -0.0045209, -0.0024917, -0.0045518, -0.0025866, -0.0013271, 0.0014982
5: -0.0002302, 0.0010060, -0.0002323, 0.0009386, -0.0010329, 0.0011155
6: -0.0051262, -0.0017378, -0.0050355, -0.0016777, -0.0023908, 0.0021798
7: -0.0221432, -0.0104516, -0.0223199, -0.0110145, -0.0077113, 0.0087020
8: 0.9752718, 0.9861102, 0.9751369, 0.9855096, -0.0074910, 0.0084349
9: -0.0009023, 0.0067982, -0.0005178, 0.0069128, -0.0057856, 0.0051279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051751
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051751
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.03 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051548, upper bound: 0.0051548
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051548, upper bound: 0.0051706
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051706, upper bound: 0.0051555
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051706, upper bound: 0.0052261
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051031, upper bound: 0.0051897
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051031, upper bound: 0.0052108
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051107, upper bound: 0.0051897
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051107, upper bound: 0.0052826
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051053
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051053
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051054
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051054
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051107
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051107
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051751
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 8, lower bound: -0.0051897, upper bound: 0.0051751

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0151338, 0.0179316, 0.0151338, 0.0179316, -0.0023798, 0.0023798
1: -0.0018588, 0.0001328, -0.0018588, 0.0001328, -0.0017494, 0.0017494
2: 0.0036743, 0.0045780, 0.0036743, 0.0045780, -0.0007604, 0.0007604
3: 0.0013190, 0.0026814, 0.0013190, 0.0026814, -0.0010101, 0.0010101
4: -0.0044474, -0.0024844, -0.0044474, -0.0024844, -0.0013521, 0.0013521
5: -0.0002349, 0.0009679, -0.0002349, 0.0009679, -0.0010670, 0.0010670
6: -0.0051863, -0.0019256, -0.0051863, -0.0019256, -0.0021227, 0.0021227
7: -0.0217401, -0.0104245, -0.0217401, -0.0104245, -0.0078701, 0.0078701
8: 0.9755719, 0.9860469, 0.9755719, 0.9860469, -0.0077110, 0.0077110
9: -0.0009080, 0.0065455, -0.0009080, 0.0065455, -0.0052441, 0.0052441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0037189, upper bound: 0.0044986
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050768, upper bound: 0.0050760
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0151338, 0.0179316, 0.0152633, 0.0179387, -0.0023897, 0.0022633
1: -0.0018588, 0.0001328, -0.0017811, 0.0001274, -0.0017432, 0.0016734
2: 0.0036743, 0.0045780, 0.0036707, 0.0045353, -0.0007221, 0.0007657
3: 0.0013190, 0.0026814, 0.0013598, 0.0027284, -0.0010694, 0.0009933
4: -0.0044474, -0.0024844, -0.0045019, -0.0025958, -0.0012734, 0.0014383
5: -0.0002349, 0.0009679, -0.0002286, 0.0009259, -0.0010232, 0.0010601
6: -0.0051863, -0.0019256, -0.0050285, -0.0017825, -0.0023418, 0.0020569
7: -0.0217401, -0.0104245, -0.0220424, -0.0110689, -0.0074098, 0.0083459
8: 0.9755719, 0.9860469, 0.9753515, 0.9854505, -0.0072629, 0.0080499
9: -0.0009080, 0.0065455, -0.0004809, 0.0067372, -0.0055417, 0.0049366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0037189, upper bound: 0.0045996
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050768, upper bound: 0.0050926
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0152633, 0.0179387, 0.0151338, 0.0179316, -0.0022633, 0.0023897
1: -0.0017811, 0.0001274, -0.0018588, 0.0001328, -0.0016734, 0.0017432
2: 0.0036707, 0.0045353, 0.0036743, 0.0045780, -0.0007657, 0.0007221
3: 0.0013598, 0.0027284, 0.0013190, 0.0026814, -0.0009933, 0.0010694
4: -0.0045019, -0.0025958, -0.0044474, -0.0024844, -0.0014383, 0.0012734
5: -0.0002286, 0.0009259, -0.0002349, 0.0009679, -0.0010601, 0.0010232
6: -0.0050285, -0.0017825, -0.0051863, -0.0019256, -0.0020569, 0.0023418
7: -0.0220424, -0.0110689, -0.0217401, -0.0104245, -0.0083459, 0.0074097
8: 0.9753515, 0.9854505, 0.9755719, 0.9860469, -0.0080500, 0.0072629
9: -0.0004809, 0.0067372, -0.0009080, 0.0065455, -0.0049366, 0.0055417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038568, upper bound: 0.0045128
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050926, upper bound: 0.0050765
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0152633, 0.0179387, 0.0152633, 0.0179387, -0.0022621, 0.0022621
1: -0.0017811, 0.0001274, -0.0017811, 0.0001274, -0.0016654, 0.0016654
2: 0.0036707, 0.0045353, 0.0036707, 0.0045353, -0.0007228, 0.0007228
3: 0.0013598, 0.0027284, 0.0013598, 0.0027284, -0.0009976, 0.0009976
4: -0.0045019, -0.0025958, -0.0045019, -0.0025958, -0.0013049, 0.0013049
5: -0.0002286, 0.0009259, -0.0002286, 0.0009259, -0.0010164, 0.0010164
6: -0.0050285, -0.0017825, -0.0050285, -0.0017825, -0.0021319, 0.0021319
7: -0.0220424, -0.0110689, -0.0220424, -0.0110689, -0.0075838, 0.0075838
8: 0.9753515, 0.9854505, 0.9753515, 0.9854505, -0.0073821, 0.0073821
9: -0.0004809, 0.0067372, -0.0004809, 0.0067372, -0.0050452, 0.0050452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038568, upper bound: 0.0048191
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050926, upper bound: 0.0051495
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0151338, 0.0179316, 0.0149418, 0.0179354, -0.0023906, 0.0025958
1: -0.0018588, 0.0001328, -0.0019901, 0.0001354, -0.0017522, 0.0018981
2: 0.0036743, 0.0045780, 0.0036728, 0.0046402, -0.0008305, 0.0007647
3: 0.0013190, 0.0026814, 0.0012892, 0.0027073, -0.0010357, 0.0010514
4: -0.0044474, -0.0024844, -0.0044619, -0.0023872, -0.0014868, 0.0013882
5: -0.0002349, 0.0009679, -0.0002364, 0.0010451, -0.0011547, 0.0010681
6: -0.0051863, -0.0019256, -0.0052794, -0.0018840, -0.0021922, 0.0022672
7: -0.0217401, -0.0104245, -0.0218183, -0.0098512, -0.0086591, 0.0080714
8: 0.9755719, 0.9860469, 0.9755049, 0.9866629, -0.0084984, 0.0078648
9: -0.0009080, 0.0065455, -0.0012963, 0.0065963, -0.0053713, 0.0057746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0032414, upper bound: 0.0042974
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050288, upper bound: 0.0051177
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0151338, 0.0179316, 0.0150606, 0.0179425, -0.0024008, 0.0024817
1: -0.0018588, 0.0001328, -0.0019171, 0.0001303, -0.0017456, 0.0018206
2: 0.0036743, 0.0045780, 0.0036691, 0.0046004, -0.0007935, 0.0007700
3: 0.0013190, 0.0026814, 0.0013330, 0.0027561, -0.0010974, 0.0010367
4: -0.0044474, -0.0024844, -0.0045209, -0.0024917, -0.0014127, 0.0014756
5: -0.0002349, 0.0009679, -0.0002302, 0.0010060, -0.0011100, 0.0010613
6: -0.0051863, -0.0019256, -0.0051262, -0.0017378, -0.0024164, 0.0022026
7: -0.0217401, -0.0104245, -0.0221432, -0.0104516, -0.0082268, 0.0085553
8: 0.9755719, 0.9860469, 0.9752718, 0.9861102, -0.0080769, 0.0082211
9: -0.0009080, 0.0065455, -0.0009023, 0.0067982, -0.0056753, 0.0054856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0032414, upper bound: 0.0044608
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050288, upper bound: 0.0051385
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0152633, 0.0179387, 0.0149418, 0.0179354, -0.0022740, 0.0026056
1: -0.0017811, 0.0001274, -0.0019901, 0.0001354, -0.0016762, 0.0018918
2: 0.0036707, 0.0045353, 0.0036728, 0.0046402, -0.0008358, 0.0007265
3: 0.0013598, 0.0027284, 0.0012892, 0.0027073, -0.0010189, 0.0011106
4: -0.0045019, -0.0025958, -0.0044619, -0.0023872, -0.0015729, 0.0013095
5: -0.0002286, 0.0009259, -0.0002364, 0.0010451, -0.0011478, 0.0010243
6: -0.0050285, -0.0017825, -0.0052794, -0.0018840, -0.0021264, 0.0024864
7: -0.0220424, -0.0110689, -0.0218183, -0.0098512, -0.0091349, 0.0076110
8: 0.9753515, 0.9854505, 0.9755049, 0.9866629, -0.0088374, 0.0074167
9: -0.0004809, 0.0067372, -0.0012963, 0.0065963, -0.0050639, 0.0060722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0034856, upper bound: 0.0043565
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050363, upper bound: 0.0051177
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0152633, 0.0179387, 0.0150606, 0.0179425, -0.0022719, 0.0024779
1: -0.0017811, 0.0001274, -0.0019171, 0.0001303, -0.0016674, 0.0018126
2: 0.0036707, 0.0045353, 0.0036691, 0.0046004, -0.0007931, 0.0007268
3: 0.0013598, 0.0027284, 0.0013330, 0.0027561, -0.0010243, 0.0010406
4: -0.0045019, -0.0025958, -0.0045209, -0.0024917, -0.0014404, 0.0013407
5: -0.0002286, 0.0009259, -0.0002302, 0.0010060, -0.0011034, 0.0010176
6: -0.0050285, -0.0017825, -0.0051262, -0.0017378, -0.0022014, 0.0022737
7: -0.0220424, -0.0110689, -0.0221432, -0.0104516, -0.0083769, 0.0077839
8: 0.9753515, 0.9854505, 0.9752718, 0.9861102, -0.0081775, 0.0075352
9: -0.0004809, 0.0067372, -0.0009023, 0.0067982, -0.0051722, 0.0055785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0034856, upper bound: 0.0047276
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050363, upper bound: 0.0052065
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0149418, 0.0179354, 0.0151338, 0.0179316, -0.0025958, 0.0023906
1: -0.0019901, 0.0001354, -0.0018588, 0.0001328, -0.0018981, 0.0017522
2: 0.0036728, 0.0046402, 0.0036743, 0.0045780, -0.0007647, 0.0008305
3: 0.0012892, 0.0027073, 0.0013190, 0.0026814, -0.0010514, 0.0010357
4: -0.0044619, -0.0023872, -0.0044474, -0.0024844, -0.0013882, 0.0014868
5: -0.0002364, 0.0010451, -0.0002349, 0.0009679, -0.0010681, 0.0011547
6: -0.0052794, -0.0018840, -0.0051863, -0.0019256, -0.0022672, 0.0021922
7: -0.0218183, -0.0098512, -0.0217401, -0.0104245, -0.0080714, 0.0086591
8: 0.9755049, 0.9866629, 0.9755719, 0.9860469, -0.0078648, 0.0084984
9: -0.0012963, 0.0065963, -0.0009080, 0.0065455, -0.0057746, 0.0053713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043981, upper bound: 0.0040876
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038535, upper bound: 0.0038535
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0149418, 0.0179354, 0.0149418, 0.0179354, -0.0025543, 0.0025543
1: -0.0019901, 0.0001354, -0.0019901, 0.0001354, -0.0018752, 0.0018752
2: 0.0036728, 0.0046402, 0.0036728, 0.0046402, -0.0008171, 0.0008171
3: 0.0012892, 0.0027073, 0.0012892, 0.0027073, -0.0010681, 0.0010681
4: -0.0044619, -0.0023872, -0.0044619, -0.0023872, -0.0014569, 0.0014569
5: -0.0002364, 0.0010451, -0.0002364, 0.0010451, -0.0011439, 0.0011439
6: -0.0052794, -0.0018840, -0.0052794, -0.0018840, -0.0022414, 0.0022414
7: -0.0218183, -0.0098512, -0.0218183, -0.0098512, -0.0084848, 0.0084848
8: 0.9755049, 0.9866629, 0.9755049, 0.9866629, -0.0083258, 0.0083258
9: -0.0012963, 0.0065963, -0.0012963, 0.0065963, -0.0056579, 0.0056579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043981, upper bound: 0.0040876
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038535, upper bound: 0.0038535
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150606, 0.0179425, 0.0151338, 0.0179316, -0.0024817, 0.0024008
1: -0.0019171, 0.0001303, -0.0018588, 0.0001328, -0.0018206, 0.0017456
2: 0.0036691, 0.0046004, 0.0036743, 0.0045780, -0.0007700, 0.0007935
3: 0.0013330, 0.0027561, 0.0013190, 0.0026814, -0.0010367, 0.0010974
4: -0.0045209, -0.0024917, -0.0044474, -0.0024844, -0.0014756, 0.0014127
5: -0.0002302, 0.0010060, -0.0002349, 0.0009679, -0.0010613, 0.0011100
6: -0.0051262, -0.0017378, -0.0051863, -0.0019256, -0.0022026, 0.0024164
7: -0.0221432, -0.0104516, -0.0217401, -0.0104245, -0.0085553, 0.0082268
8: 0.9752718, 0.9861102, 0.9755719, 0.9860469, -0.0082211, 0.0080769
9: -0.0009023, 0.0067982, -0.0009080, 0.0065455, -0.0054856, 0.0056753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033366, upper bound: 0.0039975
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051385, upper bound: 0.0050293
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0150606, 0.0179425, 0.0149418, 0.0179354, -0.0024396, 0.0025641
1: -0.0019171, 0.0001303, -0.0019901, 0.0001354, -0.0017992, 0.0018690
2: 0.0036691, 0.0046004, 0.0036728, 0.0046402, -0.0008223, 0.0007796
3: 0.0013330, 0.0027561, 0.0012892, 0.0027073, -0.0010533, 0.0011279
4: -0.0045209, -0.0024917, -0.0044619, -0.0023872, -0.0015433, 0.0013821
5: -0.0002302, 0.0010060, -0.0002364, 0.0010451, -0.0011379, 0.0011001
6: -0.0051262, -0.0017378, -0.0052794, -0.0018840, -0.0021808, 0.0024637
7: -0.0221432, -0.0104516, -0.0218183, -0.0098512, -0.0089629, 0.0080481
8: 0.9752718, 0.9861102, 0.9755049, 0.9866629, -0.0086706, 0.0078976
9: -0.0009023, 0.0067982, -0.0012963, 0.0065963, -0.0053660, 0.0059572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033366, upper bound: 0.0039975
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051385, upper bound: 0.0050293
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0149418, 0.0179354, 0.0152633, 0.0179387, -0.0026056, 0.0022740
1: -0.0019901, 0.0001354, -0.0017811, 0.0001274, -0.0018918, 0.0016762
2: 0.0036728, 0.0046402, 0.0036707, 0.0045353, -0.0007265, 0.0008358
3: 0.0012892, 0.0027073, 0.0013598, 0.0027284, -0.0011106, 0.0010189
4: -0.0044619, -0.0023872, -0.0045019, -0.0025958, -0.0013095, 0.0015729
5: -0.0002364, 0.0010451, -0.0002286, 0.0009259, -0.0010243, 0.0011478
6: -0.0052794, -0.0018840, -0.0050285, -0.0017825, -0.0024864, 0.0021265
7: -0.0218183, -0.0098512, -0.0220424, -0.0110689, -0.0076110, 0.0091349
8: 0.9755049, 0.9866629, 0.9753515, 0.9854505, -0.0074167, 0.0088374
9: -0.0012963, 0.0065963, -0.0004809, 0.0067372, -0.0060722, 0.0050639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046170, upper bound: 0.0043993
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0149418, 0.0179354, 0.0150606, 0.0179425, -0.0025641, 0.0024396
1: -0.0019901, 0.0001354, -0.0019171, 0.0001303, -0.0018690, 0.0017992
2: 0.0036728, 0.0046402, 0.0036691, 0.0046004, -0.0007796, 0.0008223
3: 0.0012892, 0.0027073, 0.0013330, 0.0027561, -0.0011279, 0.0010533
4: -0.0044619, -0.0023872, -0.0045209, -0.0024917, -0.0013821, 0.0015433
5: -0.0002364, 0.0010451, -0.0002302, 0.0010060, -0.0011001, 0.0011379
6: -0.0052794, -0.0018840, -0.0051262, -0.0017378, -0.0024637, 0.0021808
7: -0.0218183, -0.0098512, -0.0221432, -0.0104516, -0.0080481, 0.0089629
8: 0.9755049, 0.9866629, 0.9752718, 0.9861102, -0.0078976, 0.0086706
9: -0.0012963, 0.0065963, -0.0009023, 0.0067982, -0.0059572, 0.0053660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046170, upper bound: 0.0043993
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150606, 0.0179425, 0.0152633, 0.0179387, -0.0024780, 0.0022719
1: -0.0019171, 0.0001303, -0.0017811, 0.0001274, -0.0018126, 0.0016674
2: 0.0036691, 0.0046004, 0.0036707, 0.0045353, -0.0007268, 0.0007931
3: 0.0013330, 0.0027561, 0.0013598, 0.0027284, -0.0010406, 0.0010243
4: -0.0045209, -0.0024917, -0.0045019, -0.0025958, -0.0013407, 0.0014404
5: -0.0002302, 0.0010060, -0.0002286, 0.0009259, -0.0010176, 0.0011034
6: -0.0051262, -0.0017378, -0.0050285, -0.0017825, -0.0022737, 0.0022014
7: -0.0221432, -0.0104516, -0.0220424, -0.0110689, -0.0077839, 0.0083769
8: 0.9752718, 0.9861102, 0.9753515, 0.9854505, -0.0075352, 0.0081775
9: -0.0009023, 0.0067982, -0.0004809, 0.0067372, -0.0055785, 0.0051722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038477, upper bound: 0.0045426
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052105, upper bound: 0.0051043
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0150606, 0.0179425, 0.0150606, 0.0179425, -0.0024383, 0.0024383
1: -0.0019171, 0.0001303, -0.0019171, 0.0001303, -0.0017921, 0.0017921
2: 0.0036691, 0.0046004, 0.0036691, 0.0046004, -0.0007796, 0.0007796
3: 0.0013330, 0.0027561, 0.0013330, 0.0027561, -0.0010576, 0.0010576
4: -0.0045209, -0.0024917, -0.0045209, -0.0024917, -0.0014103, 0.0014103
5: -0.0002302, 0.0010060, -0.0002302, 0.0010060, -0.0010944, 0.0010944
6: -0.0051262, -0.0017378, -0.0051262, -0.0017378, -0.0022523, 0.0022523
7: -0.0221432, -0.0104516, -0.0221432, -0.0104516, -0.0082022, 0.0082022
8: 0.9752718, 0.9861102, 0.9752718, 0.9861102, -0.0080062, 0.0080062
9: -0.0009023, 0.0067982, -0.0009023, 0.0067982, -0.0054620, 0.0054620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038477, upper bound: 0.0045426
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052105, upper bound: 0.0051043
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.91 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0037189, upper bound: 0.0044986
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0050768, upper bound: 0.0050760
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0037189, upper bound: 0.0045996
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0050768, upper bound: 0.0050926
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0038568, upper bound: 0.0045128
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0050926, upper bound: 0.0050765
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0038568, upper bound: 0.0048191
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0050926, upper bound: 0.0051495
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0032414, upper bound: 0.0042974
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0050288, upper bound: 0.0051177
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0032414, upper bound: 0.0044608
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0050288, upper bound: 0.0051385
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0034856, upper bound: 0.0043565
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0050363, upper bound: 0.0051177
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0034856, upper bound: 0.0047276
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0050363, upper bound: 0.0052065
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0043981, upper bound: 0.0040876
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0038535, upper bound: 0.0038535
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0043981, upper bound: 0.0040876
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0038535, upper bound: 0.0038535
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0033366, upper bound: 0.0039975
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0051385, upper bound: 0.0050293
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0033366, upper bound: 0.0039975
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0051385, upper bound: 0.0050293
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0046170, upper bound: 0.0043993
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0046170, upper bound: 0.0043993
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0038477, upper bound: 0.0045426
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0052105, upper bound: 0.0051043
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0038477, upper bound: 0.0045426
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.91
Output dim: 8, lower bound: -0.0052105, upper bound: 0.0051043

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0148882, 0.0177005, 0.0151436, 0.0178959, -0.0024367, 0.0021039
1: -0.0020286, -0.0000027, -0.0018502, 0.0001104, -0.0018027, 0.0015811
2: 0.0037515, 0.0046576, 0.0036860, 0.0045750, -0.0006678, 0.0007775
3: 0.0013065, 0.0026347, 0.0013196, 0.0026708, -0.0010055, 0.0009958
4: -0.0042388, -0.0023313, -0.0044162, -0.0024874, -0.0010819, 0.0013213
5: -0.0001619, 0.0010681, -0.0002233, 0.0009624, -0.0009730, 0.0011027
6: -0.0053232, -0.0022621, -0.0051848, -0.0019764, -0.0020565, 0.0017284
7: -0.0205366, -0.0095255, -0.0215583, -0.0104419, -0.0063080, 0.0077107
8: 0.9766482, 0.9869439, 0.9757373, 0.9860267, -0.0062953, 0.0076293
9: -0.0015097, 0.0057559, -0.0008961, 0.0064256, -0.0051524, 0.0042159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036208, upper bound: 0.0036208
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036208, upper bound: 0.0036208
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0151338, 0.0179316, -0.0023785, 0.0022483
1: -0.0018571, 0.0001243, -0.0018588, 0.0001328, -0.0017484, 0.0016733
2: 0.0036782, 0.0045774, 0.0036743, 0.0045780, -0.0007161, 0.0007600
3: 0.0013192, 0.0026791, 0.0013190, 0.0026814, -0.0010094, 0.0009918
4: -0.0044398, -0.0024850, -0.0044474, -0.0024844, -0.0011968, 0.0013516
5: -0.0002291, 0.0009668, -0.0002349, 0.0009679, -0.0010266, 0.0010663
6: -0.0051858, -0.0019378, -0.0051863, -0.0019256, -0.0021221, 0.0019107
7: -0.0216961, -0.0104282, -0.0217401, -0.0104245, -0.0069826, 0.0078670
8: 0.9756122, 0.9860428, 0.9755719, 0.9860469, -0.0069418, 0.0077076
9: -0.0009055, 0.0065166, -0.0009080, 0.0065455, -0.0052419, 0.0046660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044986, upper bound: 0.0037189
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044986, upper bound: 0.0050768
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0148882, 0.0177005, 0.0152717, 0.0179028, -0.0024465, 0.0019878
1: -0.0020286, -0.0000027, -0.0017728, 0.0001053, -0.0017959, 0.0015053
2: 0.0037515, 0.0046576, 0.0036827, 0.0045327, -0.0006297, 0.0007827
3: 0.0013065, 0.0026347, 0.0013604, 0.0027172, -0.0010637, 0.0009765
4: -0.0042388, -0.0023313, -0.0044705, -0.0025984, -0.0010033, 0.0014239
5: -0.0001619, 0.0010681, -0.0002168, 0.0009206, -0.0009293, 0.0010962
6: -0.0053232, -0.0022621, -0.0050270, -0.0018336, -0.0022880, 0.0016625
7: -0.0205366, -0.0095255, -0.0218589, -0.0110849, -0.0058482, 0.0082786
8: 0.9766482, 0.9869439, 0.9755203, 0.9854317, -0.0058483, 0.0080423
9: -0.0015097, 0.0057559, -0.0004698, 0.0066159, -0.0055099, 0.0039088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037423, upper bound: 0.0037834
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0037423, upper bound: 0.0045996
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0152633, 0.0179387, -0.0023884, 0.0021325
1: -0.0018571, 0.0001243, -0.0017811, 0.0001274, -0.0017421, 0.0015977
2: 0.0036782, 0.0045774, 0.0036707, 0.0045353, -0.0006778, 0.0007653
3: 0.0013192, 0.0026791, 0.0013598, 0.0027284, -0.0010686, 0.0009761
4: -0.0044398, -0.0024850, -0.0045019, -0.0025958, -0.0011152, 0.0014377
5: -0.0002291, 0.0009668, -0.0002286, 0.0009259, -0.0009833, 0.0010595
6: -0.0051858, -0.0019378, -0.0050285, -0.0017825, -0.0023413, 0.0018552
7: -0.0216961, -0.0104282, -0.0220424, -0.0110689, -0.0065038, 0.0083428
8: 0.9756122, 0.9860428, 0.9753515, 0.9854505, -0.0064675, 0.0080466
9: -0.0009055, 0.0065166, -0.0004809, 0.0067372, -0.0055396, 0.0043452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045128, upper bound: 0.0038568
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045128, upper bound: 0.0050926
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0150200, 0.0177067, 0.0151436, 0.0178959, -0.0023344, 0.0021124
1: -0.0019467, -0.0000077, -0.0018502, 0.0001104, -0.0017340, 0.0015762
2: 0.0037482, 0.0046140, 0.0036860, 0.0045750, -0.0006724, 0.0007441
3: 0.0013484, 0.0026774, 0.0013196, 0.0026708, -0.0009897, 0.0010429
4: -0.0042883, -0.0024499, -0.0044162, -0.0024874, -0.0011757, 0.0012510
5: -0.0001555, 0.0010234, -0.0002233, 0.0009624, -0.0009681, 0.0010631
6: -0.0051619, -0.0021186, -0.0051848, -0.0019764, -0.0020146, 0.0019594
7: -0.0208082, -0.0102067, -0.0215583, -0.0104419, -0.0068219, 0.0072976
8: 0.9764539, 0.9863257, 0.9757373, 0.9860267, -0.0066491, 0.0072213
9: -0.0010620, 0.0059265, -0.0008961, 0.0064256, -0.0048747, 0.0045347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037834, upper bound: 0.0037423
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037834, upper bound: 0.0037423
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0151338, 0.0179316, -0.0022620, 0.0022582
1: -0.0017794, 0.0001188, -0.0018588, 0.0001328, -0.0016724, 0.0016680
2: 0.0036745, 0.0045348, 0.0036743, 0.0045780, -0.0007215, 0.0007217
3: 0.0013600, 0.0027258, 0.0013190, 0.0026814, -0.0009925, 0.0010486
4: -0.0044946, -0.0025963, -0.0044474, -0.0024844, -0.0012888, 0.0012729
5: -0.0002228, 0.0009248, -0.0002349, 0.0009679, -0.0010214, 0.0010225
6: -0.0050280, -0.0017944, -0.0051863, -0.0019256, -0.0020564, 0.0021393
7: -0.0220005, -0.0110724, -0.0217401, -0.0104245, -0.0074877, 0.0074067
8: 0.9753920, 0.9854466, 0.9755719, 0.9860469, -0.0072935, 0.0072596
9: -0.0004785, 0.0067090, -0.0009080, 0.0065455, -0.0049346, 0.0049802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045996, upper bound: 0.0037947
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045996, upper bound: 0.0050768
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0150200, 0.0177067, 0.0152717, 0.0179028, -0.0023274, 0.0019922
1: -0.0019467, -0.0000077, -0.0017728, 0.0001053, -0.0017273, 0.0015027
2: 0.0037482, 0.0046140, 0.0036827, 0.0045327, -0.0006315, 0.0007421
3: 0.0013484, 0.0026774, 0.0013604, 0.0027172, -0.0009929, 0.0009776
4: -0.0042883, -0.0024499, -0.0044705, -0.0025984, -0.0010340, 0.0012633
5: -0.0001555, 0.0010234, -0.0002168, 0.0009206, -0.0009258, 0.0010583
6: -0.0051619, -0.0021186, -0.0050270, -0.0018336, -0.0020611, 0.0017323
7: -0.0208082, -0.0102067, -0.0218589, -0.0110849, -0.0060189, 0.0073643
8: 0.9764539, 0.9863257, 0.9755203, 0.9854317, -0.0059724, 0.0072631
9: -0.0010620, 0.0059265, -0.0004698, 0.0066159, -0.0049156, 0.0040163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0030576, upper bound: 0.0041887
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029674, upper bound: 0.0037477
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0152633, 0.0179387, -0.0022609, 0.0021336
1: -0.0017794, 0.0001188, -0.0017811, 0.0001274, -0.0016644, 0.0015934
2: 0.0036745, 0.0045348, 0.0036707, 0.0045353, -0.0006789, 0.0007224
3: 0.0013600, 0.0027258, 0.0013598, 0.0027284, -0.0009968, 0.0009778
4: -0.0044946, -0.0025963, -0.0045019, -0.0025958, -0.0011450, 0.0013044
5: -0.0002228, 0.0009248, -0.0002286, 0.0009259, -0.0009790, 0.0010157
6: -0.0050280, -0.0017944, -0.0050285, -0.0017825, -0.0021314, 0.0019174
7: -0.0220005, -0.0110724, -0.0220424, -0.0110689, -0.0066687, 0.0075810
8: 0.9753920, 0.9854466, 0.9753515, 0.9854505, -0.0065874, 0.0073790
9: -0.0004785, 0.0067090, -0.0004809, 0.0067372, -0.0050433, 0.0044479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048303, upper bound: 0.0042085
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048303, upper bound: 0.0051495
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0148882, 0.0177005, 0.0149510, 0.0178990, -0.0024379, 0.0023202
1: -0.0020286, -0.0000027, -0.0019825, 0.0001132, -0.0018025, 0.0017301
2: 0.0037515, 0.0046576, 0.0036849, 0.0046375, -0.0007380, 0.0007785
3: 0.0013065, 0.0026347, 0.0012898, 0.0026967, -0.0010309, 0.0010347
4: -0.0042388, -0.0023313, -0.0044300, -0.0023899, -0.0012168, 0.0013685
5: -0.0001619, 0.0010681, -0.0002249, 0.0010399, -0.0010609, 0.0011032
6: -0.0053232, -0.0022621, -0.0052780, -0.0019347, -0.0021325, 0.0018729
7: -0.0205366, -0.0095255, -0.0216333, -0.0098681, -0.0070981, 0.0079763
8: 0.9766482, 0.9869439, 0.9756713, 0.9866453, -0.0070840, 0.0078256
9: -0.0015097, 0.0057559, -0.0012845, 0.0064747, -0.0053210, 0.0047472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 118

Time for candidate selection: 5.70 seconds

### Candidate
type: B, layer: 3, pos: 240

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0033236
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022144, upper bound: 0.0030147
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0149418, 0.0179354, -0.0023892, 0.0024597
1: -0.0018571, 0.0001243, -0.0019901, 0.0001354, -0.0017512, 0.0018200
2: 0.0036782, 0.0045774, 0.0036728, 0.0046402, -0.0007846, 0.0007643
3: 0.0013192, 0.0026791, 0.0012892, 0.0027073, -0.0010350, 0.0010325
4: -0.0044398, -0.0024850, -0.0044619, -0.0023872, -0.0013322, 0.0013877
5: -0.0002291, 0.0009668, -0.0002364, 0.0010451, -0.0011132, 0.0010675
6: -0.0051858, -0.0019378, -0.0052794, -0.0018840, -0.0021917, 0.0020518
7: -0.0216961, -0.0104282, -0.0218183, -0.0098512, -0.0077734, 0.0080682
8: 0.9756122, 0.9860428, 0.9755049, 0.9866629, -0.0077190, 0.0078615
9: -0.0009055, 0.0065166, -0.0012963, 0.0065963, -0.0053692, 0.0051959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040340, upper bound: 0.0045420
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038083, upper bound: 0.0039873
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0148882, 0.0177005, 0.0150692, 0.0179062, -0.0024482, 0.0022065
1: -0.0020286, -0.0000027, -0.0019094, 0.0001082, -0.0017956, 0.0016527
2: 0.0037515, 0.0046576, 0.0036812, 0.0045977, -0.0007012, 0.0007841
3: 0.0013065, 0.0026347, 0.0013335, 0.0027450, -0.0010912, 0.0010189
4: -0.0042388, -0.0023313, -0.0044879, -0.0024942, -0.0011427, 0.0014654
5: -0.0001619, 0.0010681, -0.0002183, 0.0010007, -0.0010162, 0.0010967
6: -0.0053232, -0.0022621, -0.0051249, -0.0017894, -0.0023684, 0.0018080
7: -0.0205366, -0.0095255, -0.0219534, -0.0104669, -0.0066662, 0.0085119
8: 0.9766482, 0.9869439, 0.9754407, 0.9860938, -0.0066629, 0.0082233
9: -0.0015097, 0.0057559, -0.0008921, 0.0066743, -0.0056573, 0.0044586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029264, upper bound: 0.0031653
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0029264, upper bound: 0.0044608
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0150606, 0.0179425, -0.0023994, 0.0023513
1: -0.0018571, 0.0001243, -0.0019171, 0.0001303, -0.0017446, 0.0017483
2: 0.0036782, 0.0045774, 0.0036691, 0.0046004, -0.0007490, 0.0007696
3: 0.0013192, 0.0026791, 0.0013330, 0.0027561, -0.0010966, 0.0010197
4: -0.0044398, -0.0024850, -0.0045209, -0.0024917, -0.0012594, 0.0014751
5: -0.0002291, 0.0009668, -0.0002302, 0.0010060, -0.0010723, 0.0010607
6: -0.0051858, -0.0019378, -0.0051262, -0.0017378, -0.0024158, 0.0020048
7: -0.0216961, -0.0104282, -0.0221432, -0.0104516, -0.0073486, 0.0085522
8: 0.9756122, 0.9860428, 0.9752718, 0.9861102, -0.0073019, 0.0082177
9: -0.0009055, 0.0065166, -0.0009023, 0.0067982, -0.0056731, 0.0049124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039975, upper bound: 0.0033812
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039975, upper bound: 0.0051385
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0150200, 0.0177067, 0.0149510, 0.0178990, -0.0023355, 0.0023287
1: -0.0019467, -0.0000077, -0.0019825, 0.0001132, -0.0017338, 0.0017252
2: 0.0037482, 0.0046140, 0.0036849, 0.0046375, -0.0007426, 0.0007450
3: 0.0013484, 0.0026774, 0.0012898, 0.0026967, -0.0010151, 0.0010818
4: -0.0042883, -0.0024499, -0.0044300, -0.0023899, -0.0013105, 0.0012983
5: -0.0001555, 0.0010234, -0.0002249, 0.0010399, -0.0010560, 0.0010636
6: -0.0051619, -0.0021186, -0.0052780, -0.0019347, -0.0020905, 0.0021039
7: -0.0208082, -0.0102067, -0.0216333, -0.0098681, -0.0076121, 0.0075633
8: 0.9764539, 0.9863257, 0.9756713, 0.9866453, -0.0074378, 0.0074175
9: -0.0010620, 0.0059265, -0.0012845, 0.0064747, -0.0050434, 0.0050659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026317, upper bound: 0.0036495
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025737, upper bound: 0.0036245
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0149418, 0.0179354, -0.0022727, 0.0024696
1: -0.0017794, 0.0001188, -0.0019901, 0.0001354, -0.0016752, 0.0018147
2: 0.0036745, 0.0045348, 0.0036728, 0.0046402, -0.0007900, 0.0007261
3: 0.0013600, 0.0027258, 0.0012892, 0.0027073, -0.0010181, 0.0010893
4: -0.0044946, -0.0025963, -0.0044619, -0.0023872, -0.0014242, 0.0013090
5: -0.0002228, 0.0009248, -0.0002364, 0.0010451, -0.0011080, 0.0010237
6: -0.0050280, -0.0017944, -0.0052794, -0.0018840, -0.0021259, 0.0022803
7: -0.0220005, -0.0110724, -0.0218183, -0.0098512, -0.0082785, 0.0076079
8: 0.9753920, 0.9854466, 0.9755049, 0.9866629, -0.0080707, 0.0074134
9: -0.0004785, 0.0067090, -0.0012963, 0.0065963, -0.0050618, 0.0055100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043337, upper bound: 0.0046150
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0150200, 0.0177067, 0.0150692, 0.0179062, -0.0023280, 0.0022082
1: -0.0019467, -0.0000077, -0.0019094, 0.0001082, -0.0017268, 0.0016500
2: 0.0037482, 0.0046140, 0.0036812, 0.0045977, -0.0007019, 0.0007424
3: 0.0013484, 0.0026774, 0.0013335, 0.0027450, -0.0010192, 0.0010182
4: -0.0042883, -0.0024499, -0.0044879, -0.0024942, -0.0011696, 0.0013108
5: -0.0001555, 0.0010234, -0.0002183, 0.0010007, -0.0010129, 0.0010586
6: -0.0051619, -0.0021186, -0.0051249, -0.0017894, -0.0021377, 0.0018741
7: -0.0208082, -0.0102067, -0.0219534, -0.0104669, -0.0068127, 0.0076325
8: 0.9764539, 0.9863257, 0.9754407, 0.9860938, -0.0067688, 0.0074625
9: -0.0010620, 0.0059265, -0.0008921, 0.0066743, -0.0050856, 0.0045501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037311, upper bound: 0.0038600
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0037311, upper bound: 0.0047276
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0150606, 0.0179425, -0.0022707, 0.0023503
1: -0.0017794, 0.0001188, -0.0019171, 0.0001303, -0.0016664, 0.0017432
2: 0.0036745, 0.0045348, 0.0036691, 0.0046004, -0.0007493, 0.0007264
3: 0.0013600, 0.0027258, 0.0013330, 0.0027561, -0.0010236, 0.0010209
4: -0.0044946, -0.0025963, -0.0045209, -0.0024917, -0.0012855, 0.0013403
5: -0.0002228, 0.0009248, -0.0002302, 0.0010060, -0.0010676, 0.0010170
6: -0.0050280, -0.0017944, -0.0051262, -0.0017378, -0.0022008, 0.0020595
7: -0.0220005, -0.0110724, -0.0221432, -0.0104516, -0.0074920, 0.0077811
8: 0.9753920, 0.9854466, 0.9752718, 0.9861102, -0.0074038, 0.0075321
9: -0.0004785, 0.0067090, -0.0009023, 0.0067982, -0.0051703, 0.0050013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045542, upper bound: 0.0039127
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045542, upper bound: 0.0052065
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0151338, 0.0179316, -0.0025180, 0.0023886
1: -0.0019338, 0.0001350, -0.0018588, 0.0001328, -0.0018397, 0.0017502
2: 0.0036730, 0.0046160, 0.0036743, 0.0045780, -0.0007642, 0.0008059
3: 0.0012895, 0.0026941, 0.0013190, 0.0026814, -0.0010477, 0.0010200
4: -0.0044618, -0.0024138, -0.0044474, -0.0024844, -0.0013882, 0.0014592
5: -0.0002362, 0.0010116, -0.0002349, 0.0009679, -0.0010666, 0.0011191
6: -0.0052777, -0.0018874, -0.0051863, -0.0019256, -0.0022580, 0.0021853
7: -0.0218182, -0.0100149, -0.0217401, -0.0104245, -0.0080713, 0.0084919
8: 0.9755049, 0.9864652, 0.9755719, 0.9860469, -0.0078648, 0.0083022
9: -0.0011810, 0.0065963, -0.0009080, 0.0065455, -0.0056575, 0.0053713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040664, upper bound: 0.0038853
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040664, upper bound: 0.0038853
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0149418, 0.0179354, -0.0024779, 0.0025522
1: -0.0019338, 0.0001350, -0.0019901, 0.0001354, -0.0018176, 0.0018729
2: 0.0036730, 0.0046160, 0.0036728, 0.0046402, -0.0008165, 0.0007929
3: 0.0012895, 0.0026941, 0.0012892, 0.0027073, -0.0010644, 0.0010519
4: -0.0044618, -0.0024138, -0.0044619, -0.0023872, -0.0014569, 0.0014303
5: -0.0002362, 0.0010116, -0.0002364, 0.0010451, -0.0011424, 0.0011087
6: -0.0052777, -0.0018874, -0.0052794, -0.0018840, -0.0022325, 0.0022343
7: -0.0218182, -0.0100149, -0.0218183, -0.0098512, -0.0084848, 0.0083229
8: 0.9755049, 0.9864652, 0.9755049, 0.9866629, -0.0083258, 0.0081327
9: -0.0011810, 0.0065963, -0.0012963, 0.0065963, -0.0055441, 0.0056578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038535, upper bound: 0.0038535
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038535, upper bound: 0.0038535
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0151338, 0.0179316, -0.0024804, 0.0022639
1: -0.0019155, 0.0001215, -0.0018588, 0.0001328, -0.0018196, 0.0016678
2: 0.0036728, 0.0045999, 0.0036743, 0.0045780, -0.0007241, 0.0007931
3: 0.0013332, 0.0027536, 0.0013190, 0.0026814, -0.0010360, 0.0010762
4: -0.0045141, -0.0024922, -0.0044474, -0.0024844, -0.0013268, 0.0014122
5: -0.0002245, 0.0010050, -0.0002349, 0.0009679, -0.0010215, 0.0011093
6: -0.0051258, -0.0017492, -0.0051863, -0.0019256, -0.0022020, 0.0022146
7: -0.0221041, -0.0104549, -0.0217401, -0.0104245, -0.0076974, 0.0082240
8: 0.9753104, 0.9861066, 0.9755719, 0.9860469, -0.0074442, 0.0080738
9: -0.0009001, 0.0067720, -0.0009080, 0.0065455, -0.0054837, 0.0051114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044608, upper bound: 0.0033639
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044608, upper bound: 0.0050293
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0149418, 0.0179354, -0.0024384, 0.0024355
1: -0.0019155, 0.0001215, -0.0019901, 0.0001354, -0.0017982, 0.0017989
2: 0.0036728, 0.0045999, 0.0036728, 0.0046402, -0.0007778, 0.0007792
3: 0.0013332, 0.0027536, 0.0012892, 0.0027073, -0.0010525, 0.0011067
4: -0.0045141, -0.0024922, -0.0044619, -0.0023872, -0.0013916, 0.0013816
5: -0.0002245, 0.0010050, -0.0002364, 0.0010451, -0.0011007, 0.0010995
6: -0.0051258, -0.0017492, -0.0052794, -0.0018840, -0.0021802, 0.0022525
7: -0.0221041, -0.0104549, -0.0218183, -0.0098512, -0.0080818, 0.0080453
8: 0.9753104, 0.9861066, 0.9755049, 0.9866629, -0.0078816, 0.0078945
9: -0.0009001, 0.0067720, -0.0012963, 0.0065963, -0.0053641, 0.0053766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043334, upper bound: 0.0044865
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0152633, 0.0179387, -0.0025279, 0.0022720
1: -0.0019338, 0.0001350, -0.0017811, 0.0001274, -0.0018335, 0.0016741
2: 0.0036730, 0.0046160, 0.0036707, 0.0045353, -0.0007259, 0.0008113
3: 0.0012895, 0.0026941, 0.0013598, 0.0027284, -0.0011069, 0.0010032
4: -0.0044618, -0.0024138, -0.0045019, -0.0025958, -0.0013095, 0.0015454
5: -0.0002362, 0.0010116, -0.0002286, 0.0009259, -0.0010228, 0.0011123
6: -0.0052777, -0.0018874, -0.0050285, -0.0017825, -0.0024772, 0.0021195
7: -0.0218182, -0.0100149, -0.0220424, -0.0110689, -0.0076110, 0.0089677
8: 0.9755049, 0.9864652, 0.9753515, 0.9854505, -0.0074167, 0.0086412
9: -0.0011810, 0.0065963, -0.0004809, 0.0067372, -0.0059551, 0.0050639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042072, upper bound: 0.0041607
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042072, upper bound: 0.0041607
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0150622, 0.0180513, 0.0153103, 0.0179380, -0.0025204, 0.0023484
1: -0.0019017, 0.0002160, -0.0017449, 0.0001267, -0.0018239, 0.0017252
2: 0.0036350, 0.0046019, 0.0036709, 0.0045203, -0.0007508, 0.0008093
3: 0.0012745, 0.0026965, 0.0013603, 0.0027126, -0.0011123, 0.0010339
4: -0.0045055, -0.0024334, -0.0045019, -0.0026130, -0.0013375, 0.0015394
5: -0.0002841, 0.0009916, -0.0002282, 0.0009037, -0.0010519, 0.0011057
6: -0.0052833, -0.0018351, -0.0050267, -0.0017850, -0.0024708, 0.0022005
7: -0.0220911, -0.0101302, -0.0220423, -0.0111726, -0.0077859, 0.0089361
8: 0.9751749, 0.9863459, 0.9753515, 0.9853328, -0.0076319, 0.0086178
9: -0.0011036, 0.0067883, -0.0004090, 0.0067372, -0.0059355, 0.0051868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042072, upper bound: 0.0041607
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042072, upper bound: 0.0041607
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0150606, 0.0179425, -0.0024878, 0.0024375
1: -0.0019338, 0.0001350, -0.0019171, 0.0001303, -0.0018114, 0.0017969
2: 0.0036730, 0.0046160, 0.0036691, 0.0046004, -0.0007791, 0.0007981
3: 0.0012895, 0.0026941, 0.0013330, 0.0027561, -0.0011242, 0.0010371
4: -0.0044618, -0.0024138, -0.0045209, -0.0024917, -0.0013820, 0.0015168
5: -0.0002362, 0.0010116, -0.0002302, 0.0010060, -0.0010986, 0.0011027
6: -0.0052777, -0.0018874, -0.0051262, -0.0017378, -0.0024548, 0.0021737
7: -0.0218182, -0.0100149, -0.0221432, -0.0104516, -0.0080480, 0.0088010
8: 0.9755049, 0.9864652, 0.9752718, 0.9861102, -0.0078976, 0.0084776
9: -0.0011810, 0.0065963, -0.0009023, 0.0067982, -0.0058434, 0.0053660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0150622, 0.0180513, 0.0151049, 0.0179418, -0.0024818, 0.0025121
1: -0.0019017, 0.0002160, -0.0018834, 0.0001296, -0.0018042, 0.0018434
2: 0.0036350, 0.0046019, 0.0036693, 0.0045863, -0.0008033, 0.0007965
3: 0.0012745, 0.0026965, 0.0013335, 0.0027400, -0.0011301, 0.0010663
4: -0.0045055, -0.0024334, -0.0045209, -0.0025077, -0.0014087, 0.0015104
5: -0.0002841, 0.0009916, -0.0002297, 0.0009849, -0.0011252, 0.0010973
6: -0.0052833, -0.0018351, -0.0051245, -0.0017403, -0.0024486, 0.0022563
7: -0.0220911, -0.0101302, -0.0221431, -0.0105486, -0.0082152, 0.0087667
8: 0.9751749, 0.9863459, 0.9752718, 0.9859993, -0.0081058, 0.0084561
9: -0.0011036, 0.0067883, -0.0008358, 0.0067982, -0.0058223, 0.0054841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0148334, 0.0177084, 0.0152717, 0.0179028, -0.0025526, 0.0019875
1: -0.0020734, -0.0000054, -0.0017728, 0.0001053, -0.0018806, 0.0014992
2: 0.0037482, 0.0046743, 0.0036827, 0.0045327, -0.0006311, 0.0008155
3: 0.0013223, 0.0027023, 0.0013604, 0.0027172, -0.0010352, 0.0010005
4: -0.0043014, -0.0023527, -0.0044705, -0.0025984, -0.0010726, 0.0014161
5: -0.0001566, 0.0010967, -0.0002168, 0.0009206, -0.0009253, 0.0011485
6: -0.0052530, -0.0020762, -0.0050270, -0.0018336, -0.0022112, 0.0018019
7: -0.0208803, -0.0096316, -0.0218589, -0.0110849, -0.0062307, 0.0082536
8: 0.9764035, 0.9869339, 0.9755203, 0.9854317, -0.0061177, 0.0081207
9: -0.0014532, 0.0059710, -0.0004698, 0.0066159, -0.0055100, 0.0041479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038621, upper bound: 0.0037196
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038621, upper bound: 0.0045426
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0152633, 0.0179387, -0.0024767, 0.0021369
1: -0.0019155, 0.0001215, -0.0017811, 0.0001274, -0.0018116, 0.0015927
2: 0.0036728, 0.0045999, 0.0036707, 0.0045353, -0.0006807, 0.0007927
3: 0.0013332, 0.0027536, 0.0013598, 0.0027284, -0.0010398, 0.0010044
4: -0.0045141, -0.0024922, -0.0045019, -0.0025958, -0.0011842, 0.0014399
5: -0.0002245, 0.0010050, -0.0002286, 0.0009259, -0.0009791, 0.0011028
6: -0.0051258, -0.0017492, -0.0050285, -0.0017825, -0.0022731, 0.0019863
7: -0.0221041, -0.0104549, -0.0220424, -0.0110689, -0.0068832, 0.0083742
8: 0.9753104, 0.9861066, 0.9753515, 0.9854505, -0.0067335, 0.0081746
9: -0.0009001, 0.0067720, -0.0004809, 0.0067372, -0.0055767, 0.0045808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047544, upper bound: 0.0039210
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047544, upper bound: 0.0051043
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0148334, 0.0177084, 0.0150692, 0.0179062, -0.0025022, 0.0021768
1: -0.0020734, -0.0000054, -0.0019094, 0.0001082, -0.0018562, 0.0016372
2: 0.0037482, 0.0046743, 0.0036812, 0.0045977, -0.0006907, 0.0007975
3: 0.0013223, 0.0027023, 0.0013335, 0.0027450, -0.0010520, 0.0010353
4: -0.0043014, -0.0023527, -0.0044879, -0.0024942, -0.0011367, 0.0013597
5: -0.0001566, 0.0010967, -0.0002183, 0.0010007, -0.0010079, 0.0011362
6: -0.0052530, -0.0020762, -0.0051249, -0.0017894, -0.0021686, 0.0018512
7: -0.0208803, -0.0096316, -0.0219534, -0.0104669, -0.0066234, 0.0079283
8: 0.9764035, 0.9869339, 0.9754407, 0.9860938, -0.0065843, 0.0078072
9: -0.0014532, 0.0059710, -0.0008921, 0.0066743, -0.0052944, 0.0044249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036824, upper bound: 0.0036759
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0036824, upper bound: 0.0045426
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0150606, 0.0179425, -0.0024370, 0.0023142
1: -0.0019155, 0.0001215, -0.0019171, 0.0001303, -0.0017912, 0.0017271
2: 0.0036728, 0.0045999, 0.0036691, 0.0046004, -0.0007366, 0.0007792
3: 0.0013332, 0.0027536, 0.0013330, 0.0027561, -0.0010568, 0.0010385
4: -0.0045141, -0.0024922, -0.0045209, -0.0024917, -0.0012521, 0.0014098
5: -0.0002245, 0.0010050, -0.0002302, 0.0010060, -0.0010602, 0.0010938
6: -0.0051258, -0.0017492, -0.0051262, -0.0017378, -0.0022517, 0.0020286
7: -0.0221041, -0.0104549, -0.0221432, -0.0104516, -0.0072939, 0.0081995
8: 0.9753104, 0.9861066, 0.9752718, 0.9861102, -0.0072081, 0.0080033
9: -0.0009001, 0.0067720, -0.0009023, 0.0067982, -0.0054601, 0.0048690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046587, upper bound: 0.0038477
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046587, upper bound: 0.0051043
time: 0.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.69 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0036208, upper bound: 0.0036208
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0036208, upper bound: 0.0036208
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0044986, upper bound: 0.0037189
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0044986, upper bound: 0.0050768
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0037423, upper bound: 0.0037834
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0037423, upper bound: 0.0045996
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0045128, upper bound: 0.0038568
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0045128, upper bound: 0.0050926
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0037834, upper bound: 0.0037423
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0037834, upper bound: 0.0037423
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0045996, upper bound: 0.0037947
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0045996, upper bound: 0.0050768
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0030576, upper bound: 0.0041887
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0029674, upper bound: 0.0037477
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0048303, upper bound: 0.0042085
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0048303, upper bound: 0.0051495
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0033236
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0022144, upper bound: 0.0030147
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0040340, upper bound: 0.0045420
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0038083, upper bound: 0.0039873
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0029264, upper bound: 0.0031653
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0029264, upper bound: 0.0044608
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0039975, upper bound: 0.0033812
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0039975, upper bound: 0.0051385
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0026317, upper bound: 0.0036495
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0025737, upper bound: 0.0036245
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0043337, upper bound: 0.0046150
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0037311, upper bound: 0.0038600
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0037311, upper bound: 0.0047276
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0045542, upper bound: 0.0039127
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0045542, upper bound: 0.0052065
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0040664, upper bound: 0.0038853
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0040664, upper bound: 0.0038853
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0038535, upper bound: 0.0038535
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0038535, upper bound: 0.0038535
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0044608, upper bound: 0.0033639
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0044608, upper bound: 0.0050293
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0043334, upper bound: 0.0044865
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0042072, upper bound: 0.0041607
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0042072, upper bound: 0.0041607
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0042072, upper bound: 0.0041607
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0042072, upper bound: 0.0041607
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0041090, upper bound: 0.0041564
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0038621, upper bound: 0.0037196
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0038621, upper bound: 0.0045426
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0047544, upper bound: 0.0039210
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0047544, upper bound: 0.0051043
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0036824, upper bound: 0.0036759
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0036824, upper bound: 0.0045426
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0046587, upper bound: 0.0038477
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 8, lower bound: -0.0046587, upper bound: 0.0051043

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0148882, 0.0177005, -0.0021087, 0.0024748
1: -0.0018571, 0.0001243, -0.0020286, -0.0000027, -0.0015849, 0.0018227
2: 0.0036782, 0.0045774, 0.0037515, 0.0046576, -0.0007906, 0.0006693
3: 0.0013192, 0.0026791, 0.0013065, 0.0026347, -0.0009962, 0.0010144
4: -0.0044398, -0.0024850, -0.0042388, -0.0023313, -0.0013634, 0.0010835
5: -0.0002291, 0.0009668, -0.0001619, 0.0010681, -0.0011134, 0.0009753
6: -0.0051858, -0.0019378, -0.0053232, -0.0022621, -0.0017297, 0.0021135
7: -0.0216961, -0.0104282, -0.0205366, -0.0095255, -0.0079554, 0.0063177
8: 0.9756122, 0.9860428, 0.9766482, 0.9869439, -0.0078560, 0.0063066
9: -0.0009055, 0.0065166, -0.0015097, 0.0057559, -0.0042226, 0.0053149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.87 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036218, upper bound: 0.0019997
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033458, upper bound: 0.0028410
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0151357, 0.0179196, -0.0022469, 0.0022469
1: -0.0018571, 0.0001243, -0.0018571, 0.0001243, -0.0016722, 0.0016722
2: 0.0036782, 0.0045774, 0.0036782, 0.0045774, -0.0007156, 0.0007156
3: 0.0013192, 0.0026791, 0.0013192, 0.0026791, -0.0009911, 0.0009911
4: -0.0044398, -0.0024850, -0.0044398, -0.0024850, -0.0011963, 0.0011963
5: -0.0002291, 0.0009668, -0.0002291, 0.0009668, -0.0010259, 0.0010259
6: -0.0051858, -0.0019378, -0.0051858, -0.0019378, -0.0019102, 0.0019102
7: -0.0216961, -0.0104282, -0.0216961, -0.0104282, -0.0069795, 0.0069795
8: 0.9756122, 0.9860428, 0.9756122, 0.9860428, -0.0069383, 0.0069383
9: -0.0009055, 0.0065166, -0.0009055, 0.0065166, -0.0046639, 0.0046639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.83 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036218, upper bound: 0.0028330
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0033458, upper bound: 0.0045947
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0148882, 0.0177005, 0.0152650, 0.0179267, -0.0024840, 0.0019922
1: -0.0020286, -0.0000027, -0.0017794, 0.0001188, -0.0018157, 0.0015089
2: 0.0037515, 0.0046576, 0.0036745, 0.0045348, -0.0006311, 0.0007956
3: 0.0013065, 0.0026347, 0.0013600, 0.0027258, -0.0010736, 0.0009769
4: -0.0042388, -0.0023313, -0.0044946, -0.0025963, -0.0010048, 0.0014627
5: -0.0001619, 0.0010681, -0.0002228, 0.0009248, -0.0009315, 0.0011063
6: -0.0053232, -0.0022621, -0.0050280, -0.0017944, -0.0023421, 0.0016639
7: -0.0205366, -0.0095255, -0.0220005, -0.0110724, -0.0058574, 0.0085051
8: 0.9766482, 0.9869439, 0.9753920, 0.9854466, -0.0058586, 0.0082480
9: -0.0015097, 0.0057559, -0.0004785, 0.0067090, -0.0056582, 0.0039153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0028793, upper bound: 0.0042790
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 4.52 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029144, upper bound: 0.0020200
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0028360, upper bound: 0.0034509
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0150200, 0.0177067, -0.0021173, 0.0023725
1: -0.0018571, 0.0001243, -0.0019467, -0.0000077, -0.0015800, 0.0017540
2: 0.0036782, 0.0045774, 0.0037482, 0.0046140, -0.0007572, 0.0006739
3: 0.0013192, 0.0026791, 0.0013484, 0.0026774, -0.0010433, 0.0009986
4: -0.0044398, -0.0024850, -0.0042883, -0.0024499, -0.0012931, 0.0011773
5: -0.0002291, 0.0009668, -0.0001555, 0.0010234, -0.0010738, 0.0009705
6: -0.0051858, -0.0019378, -0.0051619, -0.0021186, -0.0019606, 0.0020716
7: -0.0216961, -0.0104282, -0.0208082, -0.0102067, -0.0075424, 0.0068317
8: 0.9756122, 0.9860428, 0.9764539, 0.9863257, -0.0074479, 0.0066604
9: -0.0009055, 0.0065166, -0.0010620, 0.0059265, -0.0045414, 0.0050373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039569, upper bound: 0.0032484
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0039297, upper bound: 0.0031968
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0152650, 0.0179267, -0.0022568, 0.0021310
1: -0.0018571, 0.0001243, -0.0017794, 0.0001188, -0.0016668, 0.0015965
2: 0.0036782, 0.0045774, 0.0036745, 0.0045348, -0.0006773, 0.0007210
3: 0.0013192, 0.0026791, 0.0013600, 0.0027258, -0.0010479, 0.0009754
4: -0.0044398, -0.0024850, -0.0044946, -0.0025963, -0.0011146, 0.0012882
5: -0.0002291, 0.0009668, -0.0002228, 0.0009248, -0.0009825, 0.0010207
6: -0.0051858, -0.0019378, -0.0050280, -0.0017944, -0.0021387, 0.0018546
7: -0.0216961, -0.0104282, -0.0220005, -0.0110724, -0.0065007, 0.0074846
8: 0.9756122, 0.9860428, 0.9753920, 0.9854466, -0.0064641, 0.0072900
9: -0.0009055, 0.0065166, -0.0004785, 0.0067090, -0.0049781, 0.0043431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039569, upper bound: 0.0048743
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039297, upper bound: 0.0048726
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0148882, 0.0177005, -0.0019922, 0.0024840
1: -0.0017794, 0.0001188, -0.0020286, -0.0000027, -0.0015089, 0.0018157
2: 0.0036745, 0.0045348, 0.0037515, 0.0046576, -0.0007956, 0.0006311
3: 0.0013600, 0.0027258, 0.0013065, 0.0026347, -0.0009769, 0.0010736
4: -0.0044946, -0.0025963, -0.0042388, -0.0023313, -0.0014627, 0.0010048
5: -0.0002228, 0.0009248, -0.0001619, 0.0010681, -0.0011063, 0.0009315
6: -0.0050280, -0.0017944, -0.0053232, -0.0022621, -0.0016639, 0.0023421
7: -0.0220005, -0.0110724, -0.0205366, -0.0095255, -0.0085051, 0.0058574
8: 0.9753920, 0.9854466, 0.9766482, 0.9869439, -0.0082480, 0.0058586
9: -0.0004785, 0.0067090, -0.0015097, 0.0057559, -0.0039153, 0.0056582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 3.99 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037628, upper bound: 0.0020451
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034509, upper bound: 0.0029994
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0151357, 0.0179196, -0.0021310, 0.0022568
1: -0.0017794, 0.0001188, -0.0018571, 0.0001243, -0.0015965, 0.0016668
2: 0.0036745, 0.0045348, 0.0036782, 0.0045774, -0.0007210, 0.0006773
3: 0.0013600, 0.0027258, 0.0013192, 0.0026791, -0.0009754, 0.0010479
4: -0.0044946, -0.0025963, -0.0044398, -0.0024850, -0.0012882, 0.0011146
5: -0.0002228, 0.0009248, -0.0002291, 0.0009668, -0.0010207, 0.0009825
6: -0.0050280, -0.0017944, -0.0051858, -0.0019378, -0.0018546, 0.0021387
7: -0.0220005, -0.0110724, -0.0216961, -0.0104282, -0.0074846, 0.0065007
8: 0.9753920, 0.9854466, 0.9756122, 0.9860428, -0.0072900, 0.0064641
9: -0.0004785, 0.0067090, -0.0009055, 0.0065166, -0.0043431, 0.0049781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 3.92 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037628, upper bound: 0.0028330
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0034509, upper bound: 0.0046022
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0150200, 0.0177067, 0.0153438, 0.0179023, -0.0023228, 0.0019184
1: -0.0019467, -0.0000077, -0.0017212, 0.0001049, -0.0017233, 0.0014478
2: 0.0037482, 0.0046140, 0.0036828, 0.0045096, -0.0006081, 0.0007407
3: 0.0013484, 0.0026774, 0.0013608, 0.0027041, -0.0009773, 0.0009736
4: -0.0042883, -0.0024499, -0.0044705, -0.0026235, -0.0010085, 0.0012628
5: -0.0001555, 0.0010234, -0.0002165, 0.0008888, -0.0008927, 0.0010557
6: -0.0051619, -0.0021186, -0.0050251, -0.0018372, -0.0020543, 0.0017237
7: -0.0208082, -0.0102067, -0.0218588, -0.0112375, -0.0058634, 0.0073614
8: 0.9764539, 0.9863257, 0.9755203, 0.9852464, -0.0057890, 0.0072581
9: -0.0010620, 0.0059265, -0.0003626, 0.0066159, -0.0049134, 0.0039074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029674, upper bound: 0.0037477
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029674, upper bound: 0.0037477
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0150200, 0.0177067, -0.0019966, 0.0023640
1: -0.0017794, 0.0001188, -0.0019467, -0.0000077, -0.0015062, 0.0017458
2: 0.0036745, 0.0045348, 0.0037482, 0.0046140, -0.0007545, 0.0006329
3: 0.0013600, 0.0027258, 0.0013484, 0.0026774, -0.0009780, 0.0010021
4: -0.0044946, -0.0025963, -0.0042883, -0.0024499, -0.0013053, 0.0010354
5: -0.0002228, 0.0009248, -0.0001555, 0.0010234, -0.0010676, 0.0009280
6: -0.0050280, -0.0017944, -0.0051619, -0.0021186, -0.0017335, 0.0021188
7: -0.0220005, -0.0110724, -0.0208082, -0.0102067, -0.0076084, 0.0060275
8: 0.9753920, 0.9854466, 0.9764539, 0.9863257, -0.0074890, 0.0059826
9: -0.0004785, 0.0067090, -0.0010620, 0.0059265, -0.0040224, 0.0050778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042061, upper bound: 0.0030572
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038333, upper bound: 0.0029709
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0152650, 0.0179267, -0.0021322, 0.0021322
1: -0.0017794, 0.0001188, -0.0017794, 0.0001188, -0.0015922, 0.0015922
2: 0.0036745, 0.0045348, 0.0036745, 0.0045348, -0.0006785, 0.0006785
3: 0.0013600, 0.0027258, 0.0013600, 0.0027258, -0.0009771, 0.0009771
4: -0.0044946, -0.0025963, -0.0044946, -0.0025963, -0.0011445, 0.0011445
5: -0.0002228, 0.0009248, -0.0002228, 0.0009248, -0.0009783, 0.0009783
6: -0.0050280, -0.0017944, -0.0050280, -0.0017944, -0.0019169, 0.0019169
7: -0.0220005, -0.0110724, -0.0220005, -0.0110724, -0.0066658, 0.0066658
8: 0.9753920, 0.9854466, 0.9753920, 0.9854466, -0.0065841, 0.0065841
9: -0.0004785, 0.0067090, -0.0004785, 0.0067090, -0.0044459, 0.0044459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042061, upper bound: 0.0047273
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0038333, upper bound: 0.0047766
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0150178, 0.0179349, -0.0023872, 0.0023833
1: -0.0018571, 0.0001243, -0.0019338, 0.0001350, -0.0017491, 0.0017626
2: 0.0036782, 0.0045774, 0.0036730, 0.0046160, -0.0007603, 0.0007638
3: 0.0013192, 0.0026791, 0.0012895, 0.0026941, -0.0010192, 0.0010288
4: -0.0044398, -0.0024850, -0.0044618, -0.0024138, -0.0013049, 0.0013876
5: -0.0002291, 0.0009668, -0.0002362, 0.0010116, -0.0010780, 0.0010660
6: -0.0051858, -0.0019378, -0.0052777, -0.0018874, -0.0021847, 0.0020426
7: -0.0216961, -0.0104282, -0.0218182, -0.0100149, -0.0076073, 0.0080682
8: 0.9756122, 0.9860428, 0.9755049, 0.9864652, -0.0075229, 0.0078615
9: -0.0009055, 0.0065166, -0.0011810, 0.0065963, -0.0053691, 0.0050796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038083, upper bound: 0.0039873
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038083, upper bound: 0.0039873
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0148882, 0.0177005, 0.0150623, 0.0179310, -0.0024882, 0.0022106
1: -0.0020286, -0.0000027, -0.0019155, 0.0001215, -0.0018164, 0.0016561
2: 0.0037515, 0.0046576, 0.0036728, 0.0045999, -0.0007025, 0.0007979
3: 0.0013065, 0.0026347, 0.0013332, 0.0027536, -0.0011017, 0.0010193
4: -0.0042388, -0.0023313, -0.0045141, -0.0024922, -0.0011442, 0.0015042
5: -0.0001619, 0.0010681, -0.0002245, 0.0010050, -0.0010184, 0.0011070
6: -0.0053232, -0.0022621, -0.0051258, -0.0017492, -0.0024209, 0.0018095
7: -0.0205366, -0.0095255, -0.0221041, -0.0104549, -0.0066747, 0.0087386
8: 0.9766482, 0.9869439, 0.9753104, 0.9861066, -0.0066728, 0.0084311
9: -0.0015097, 0.0057559, -0.0009001, 0.0067720, -0.0058082, 0.0044644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.89 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018134, upper bound: 0.0014940
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015303, upper bound: 0.0016515
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151357, 0.0179196, 0.0150623, 0.0179310, -0.0022625, 0.0023499
1: -0.0018571, 0.0001243, -0.0019155, 0.0001215, -0.0016666, 0.0017472
2: 0.0036782, 0.0045774, 0.0036728, 0.0045999, -0.0007486, 0.0007237
3: 0.0013192, 0.0026791, 0.0013332, 0.0027536, -0.0010755, 0.0010190
4: -0.0044398, -0.0024850, -0.0045141, -0.0024922, -0.0012589, 0.0013262
5: -0.0002291, 0.0009668, -0.0002245, 0.0010050, -0.0010715, 0.0010208
6: -0.0051858, -0.0019378, -0.0051258, -0.0017492, -0.0022140, 0.0020041
7: -0.0216961, -0.0104282, -0.0221041, -0.0104549, -0.0073459, 0.0076943
8: 0.9756122, 0.9860428, 0.9753104, 0.9861066, -0.0072987, 0.0074407
9: -0.0009055, 0.0065166, -0.0009001, 0.0067720, -0.0051093, 0.0049105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.89 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029600, upper bound: 0.0028647
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027187, upper bound: 0.0045907
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0150178, 0.0179349, -0.0022707, 0.0023932
1: -0.0017794, 0.0001188, -0.0019338, 0.0001350, -0.0016731, 0.0017572
2: 0.0036745, 0.0045348, 0.0036730, 0.0046160, -0.0007657, 0.0007255
3: 0.0013600, 0.0027258, 0.0012895, 0.0026941, -0.0010024, 0.0010856
4: -0.0044946, -0.0025963, -0.0044618, -0.0024138, -0.0013969, 0.0013089
5: -0.0002228, 0.0009248, -0.0002362, 0.0010116, -0.0010729, 0.0010221
6: -0.0050280, -0.0017944, -0.0052777, -0.0018874, -0.0021189, 0.0022711
7: -0.0220005, -0.0110724, -0.0218182, -0.0100149, -0.0081125, 0.0076079
8: 0.9753920, 0.9854466, 0.9755049, 0.9864652, -0.0078746, 0.0074134
9: -0.0004785, 0.0067090, -0.0011810, 0.0065963, -0.0050618, 0.0053938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0153120, 0.0179259, 0.0150622, 0.0180513, -0.0023471, 0.0023897
1: -0.0017432, 0.0001181, -0.0019017, 0.0002160, -0.0017242, 0.0017518
2: 0.0036747, 0.0045198, 0.0036350, 0.0046019, -0.0007648, 0.0007504
3: 0.0013605, 0.0027100, 0.0012745, 0.0026965, -0.0010331, 0.0010923
4: -0.0044946, -0.0026135, -0.0045055, -0.0024334, -0.0013906, 0.0013370
5: -0.0002224, 0.0009026, -0.0002841, 0.0009916, -0.0010688, 0.0010512
6: -0.0050262, -0.0017970, -0.0052833, -0.0018351, -0.0022000, 0.0022655
7: -0.0220004, -0.0111760, -0.0220911, -0.0101302, -0.0080785, 0.0077830
8: 0.9753920, 0.9853289, 0.9751749, 0.9863459, -0.0078519, 0.0076287
9: -0.0004066, 0.0067090, -0.0011036, 0.0067883, -0.0051848, 0.0053735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0150200, 0.0177067, 0.0150623, 0.0179310, -0.0023648, 0.0022124
1: -0.0019467, -0.0000077, -0.0019155, 0.0001215, -0.0017465, 0.0016534
2: 0.0037482, 0.0046140, 0.0036728, 0.0045999, -0.0007032, 0.0007551
3: 0.0013484, 0.0026774, 0.0013332, 0.0027536, -0.0010289, 0.0010186
4: -0.0042883, -0.0024499, -0.0045141, -0.0024922, -0.0011709, 0.0013505
5: -0.0001555, 0.0010234, -0.0002245, 0.0010050, -0.0010151, 0.0010684
6: -0.0051619, -0.0021186, -0.0051258, -0.0017492, -0.0021946, 0.0018753
7: -0.0208082, -0.0102067, -0.0221041, -0.0104549, -0.0068208, 0.0078627
8: 0.9764539, 0.9863257, 0.9753104, 0.9861066, -0.0067782, 0.0076792
9: -0.0010620, 0.0059265, -0.0009001, 0.0067720, -0.0052390, 0.0045558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.92 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0027380, upper bound: 0.0027214
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024832, upper bound: 0.0028058
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0148334, 0.0177084, -0.0019919, 0.0025891
1: -0.0017794, 0.0001188, -0.0020734, -0.0000054, -0.0015027, 0.0018991
2: 0.0036745, 0.0045348, 0.0037482, 0.0046743, -0.0008279, 0.0006324
3: 0.0013600, 0.0027258, 0.0013223, 0.0027023, -0.0010009, 0.0010445
4: -0.0044946, -0.0025963, -0.0043014, -0.0023527, -0.0014581, 0.0010740
5: -0.0002228, 0.0009248, -0.0001566, 0.0010967, -0.0011578, 0.0009275
6: -0.0050280, -0.0017944, -0.0052530, -0.0020762, -0.0018031, 0.0022689
7: -0.0220005, -0.0110724, -0.0208803, -0.0096316, -0.0084977, 0.0062394
8: 0.9753920, 0.9854466, 0.9764035, 0.9869339, -0.0083465, 0.0061279
9: -0.0004785, 0.0067090, -0.0014532, 0.0059710, -0.0041540, 0.0056721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 3.91 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036369, upper bound: 0.0020482
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032230, upper bound: 0.0029973
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0152650, 0.0179267, 0.0150623, 0.0179310, -0.0021354, 0.0023489
1: -0.0017794, 0.0001188, -0.0019155, 0.0001215, -0.0015915, 0.0017420
2: 0.0036745, 0.0045348, 0.0036728, 0.0045999, -0.0007489, 0.0006802
3: 0.0013600, 0.0027258, 0.0013332, 0.0027536, -0.0010037, 0.0010201
4: -0.0044946, -0.0025963, -0.0045141, -0.0024922, -0.0012850, 0.0011837
5: -0.0002228, 0.0009248, -0.0002245, 0.0010050, -0.0010669, 0.0009784
6: -0.0050280, -0.0017944, -0.0051258, -0.0017492, -0.0019858, 0.0020589
7: -0.0220005, -0.0110724, -0.0221041, -0.0104549, -0.0074893, 0.0068803
8: 0.9753920, 0.9854466, 0.9753104, 0.9861066, -0.0074007, 0.0067302
9: -0.0004785, 0.0067090, -0.0009001, 0.0067720, -0.0045788, 0.0049994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99

Time for candidate selection: 4.04 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036369, upper bound: 0.0028802
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0032230, upper bound: 0.0047438
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0152070, 0.0179311, -0.0025159, 0.0023137
1: -0.0019338, 0.0001350, -0.0018055, 0.0001323, -0.0018375, 0.0016945
2: 0.0036730, 0.0046160, 0.0036744, 0.0045548, -0.0007405, 0.0008054
3: 0.0012895, 0.0026941, 0.0013194, 0.0026683, -0.0010318, 0.0010164
4: -0.0044618, -0.0024138, -0.0044474, -0.0025097, -0.0013623, 0.0014592
5: -0.0002362, 0.0010116, -0.0002346, 0.0009356, -0.0010329, 0.0011176
6: -0.0052777, -0.0018874, -0.0051845, -0.0019287, -0.0022508, 0.0021765
7: -0.0218182, -0.0100149, -0.0217401, -0.0105803, -0.0079135, 0.0084919
8: 0.9755049, 0.9864652, 0.9755719, 0.9858596, -0.0076777, 0.0083022
9: -0.0011810, 0.0065963, -0.0007983, 0.0065455, -0.0056575, 0.0052605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042416, upper bound: 0.0036703
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0152657, 0.0180417, -0.0026377, 0.0022510
1: -0.0019338, 0.0001350, -0.0017633, 0.0002134, -0.0019231, 0.0016470
2: 0.0036730, 0.0046160, 0.0036389, 0.0045359, -0.0007206, 0.0008447
3: 0.0012895, 0.0026941, 0.0013011, 0.0026692, -0.0010358, 0.0010395
4: -0.0044618, -0.0024138, -0.0044882, -0.0025335, -0.0013373, 0.0015003
5: -0.0002362, 0.0010116, -0.0002825, 0.0009102, -0.0010032, 0.0011680
6: -0.0052777, -0.0018874, -0.0051905, -0.0018768, -0.0023038, 0.0021828
7: -0.0218182, -0.0100149, -0.0219935, -0.0107215, -0.0077614, 0.0087461
8: 0.9755049, 0.9864652, 0.9752560, 0.9857022, -0.0075051, 0.0086151
9: -0.0011810, 0.0065963, -0.0006990, 0.0067257, -0.0058368, 0.0051541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042416, upper bound: 0.0036703
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0148882, 0.0177005, -0.0022106, 0.0024882
1: -0.0019155, 0.0001215, -0.0020286, -0.0000027, -0.0016561, 0.0018164
2: 0.0036728, 0.0045999, 0.0037515, 0.0046576, -0.0007979, 0.0007025
3: 0.0013332, 0.0027536, 0.0013065, 0.0026347, -0.0010193, 0.0011017
4: -0.0045141, -0.0024922, -0.0042388, -0.0023313, -0.0015042, 0.0011442
5: -0.0002245, 0.0010050, -0.0001619, 0.0010681, -0.0011070, 0.0010184
6: -0.0051258, -0.0017492, -0.0053232, -0.0022621, -0.0018095, 0.0024209
7: -0.0221041, -0.0104549, -0.0205366, -0.0095255, -0.0087386, 0.0066747
8: 0.9753104, 0.9861066, 0.9766482, 0.9869439, -0.0084311, 0.0066728
9: -0.0009001, 0.0067720, -0.0015097, 0.0057559, -0.0044644, 0.0058082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.98 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035383, upper bound: 0.0017627
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031466, upper bound: 0.0023956
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0151357, 0.0179196, -0.0023499, 0.0022624
1: -0.0019155, 0.0001215, -0.0018571, 0.0001243, -0.0017472, 0.0016666
2: 0.0036728, 0.0045999, 0.0036782, 0.0045774, -0.0007237, 0.0007486
3: 0.0013332, 0.0027536, 0.0013192, 0.0026791, -0.0010190, 0.0010755
4: -0.0045141, -0.0024922, -0.0044398, -0.0024850, -0.0013262, 0.0012589
5: -0.0002245, 0.0010050, -0.0002291, 0.0009668, -0.0010208, 0.0010715
6: -0.0051258, -0.0017492, -0.0051858, -0.0019378, -0.0020041, 0.0022140
7: -0.0221041, -0.0104549, -0.0216961, -0.0104282, -0.0076943, 0.0073459
8: 0.9753104, 0.9861066, 0.9756122, 0.9860428, -0.0074407, 0.0072988
9: -0.0009001, 0.0067720, -0.0009055, 0.0065166, -0.0049105, 0.0051093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.91 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035384, upper bound: 0.0028209
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0031466, upper bound: 0.0045374
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0150178, 0.0179349, -0.0024363, 0.0023590
1: -0.0019155, 0.0001215, -0.0019338, 0.0001350, -0.0017959, 0.0017418
2: 0.0036728, 0.0045999, 0.0036730, 0.0046160, -0.0007535, 0.0007787
3: 0.0013332, 0.0027536, 0.0012895, 0.0026941, -0.0010364, 0.0011030
4: -0.0045141, -0.0024922, -0.0044618, -0.0024138, -0.0013644, 0.0013816
5: -0.0002245, 0.0010050, -0.0002362, 0.0010116, -0.0010657, 0.0010979
6: -0.0051258, -0.0017492, -0.0052777, -0.0018874, -0.0021731, 0.0022439
7: -0.0221041, -0.0104549, -0.0218182, -0.0100149, -0.0079161, 0.0080452
8: 0.9753104, 0.9861066, 0.9755049, 0.9864652, -0.0076851, 0.0078945
9: -0.0009001, 0.0067720, -0.0011810, 0.0065963, -0.0053641, 0.0052605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0151066, 0.0179302, 0.0150622, 0.0180513, -0.0025108, 0.0023590
1: -0.0018819, 0.0001207, -0.0019017, 0.0002160, -0.0018424, 0.0017390
2: 0.0036731, 0.0045858, 0.0036350, 0.0046019, -0.0007536, 0.0008029
3: 0.0013337, 0.0027374, 0.0012745, 0.0026965, -0.0010656, 0.0011106
4: -0.0045141, -0.0025082, -0.0045055, -0.0024334, -0.0013580, 0.0014082
5: -0.0002240, 0.0009838, -0.0002841, 0.0009916, -0.0010635, 0.0011246
6: -0.0051240, -0.0017518, -0.0052833, -0.0018351, -0.0022558, 0.0022390
7: -0.0221041, -0.0105519, -0.0220911, -0.0101302, -0.0078834, 0.0082125
8: 0.9753104, 0.9859959, 0.9751749, 0.9863459, -0.0076675, 0.0081027
9: -0.0008336, 0.0067720, -0.0011036, 0.0067883, -0.0054822, 0.0052420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0153349, 0.0179382, -0.0025258, 0.0021990
1: -0.0019338, 0.0001350, -0.0017295, 0.0001270, -0.0018312, 0.0016192
2: 0.0036730, 0.0046160, 0.0036708, 0.0045122, -0.0007027, 0.0008107
3: 0.0012895, 0.0026941, 0.0013602, 0.0027154, -0.0010909, 0.0009996
4: -0.0044618, -0.0024138, -0.0045019, -0.0026208, -0.0012839, 0.0015454
5: -0.0002362, 0.0010116, -0.0002284, 0.0008943, -0.0009896, 0.0011108
6: -0.0052777, -0.0018874, -0.0050266, -0.0017859, -0.0024685, 0.0021112
7: -0.0218182, -0.0100149, -0.0220424, -0.0112216, -0.0074562, 0.0089676
8: 0.9755049, 0.9864652, 0.9753515, 0.9852650, -0.0072325, 0.0086412
9: -0.0011810, 0.0065963, -0.0003737, 0.0067372, -0.0059551, 0.0049550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044306, upper bound: 0.0041627
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0153931, 0.0180490, -0.0026476, 0.0021315
1: -0.0019338, 0.0001350, -0.0016852, 0.0002091, -0.0019164, 0.0015694
2: 0.0036730, 0.0046160, 0.0036351, 0.0044930, -0.0006814, 0.0008498
3: 0.0012895, 0.0026941, 0.0013414, 0.0027160, -0.0010989, 0.0010246
4: -0.0044618, -0.0024138, -0.0045424, -0.0026462, -0.0012576, 0.0015859
5: -0.0002362, 0.0010116, -0.0002769, 0.0008686, -0.0009593, 0.0011614
6: -0.0052777, -0.0018874, -0.0050336, -0.0017352, -0.0025167, 0.0021190
7: -0.0218182, -0.0100149, -0.0222938, -0.0113740, -0.0072974, 0.0092219
8: 0.9755049, 0.9864652, 0.9750398, 0.9851038, -0.0070515, 0.0089621
9: -0.0011810, 0.0065963, -0.0002691, 0.0069129, -0.0061363, 0.0048445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044306, upper bound: 0.0041627
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150622, 0.0180513, 0.0153349, 0.0179382, -0.0024733, 0.0023232
1: -0.0019017, 0.0002160, -0.0017295, 0.0001270, -0.0017935, 0.0017064
2: 0.0036350, 0.0046019, 0.0036708, 0.0045122, -0.0007428, 0.0007941
3: 0.0012745, 0.0026965, 0.0013602, 0.0027154, -0.0011103, 0.0010045
4: -0.0045055, -0.0024334, -0.0045019, -0.0026208, -0.0013291, 0.0015258
5: -0.0002841, 0.0009916, -0.0002284, 0.0008943, -0.0010409, 0.0010881
6: -0.0052833, -0.0018351, -0.0050266, -0.0017859, -0.0024735, 0.0021639
7: -0.0220911, -0.0101302, -0.0220424, -0.0112216, -0.0077346, 0.0088492
8: 0.9751749, 0.9863459, 0.9753515, 0.9852650, -0.0075692, 0.0085038
9: -0.0011036, 0.0067883, -0.0003737, 0.0067372, -0.0058725, 0.0051508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036257, upper bound: 0.0036242
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 118

Time for candidate selection: 4.64 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034314, upper bound: 0.0024851
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034240, upper bound: 0.0029998
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0150622, 0.0180513, 0.0153931, 0.0180490, -0.0025107, 0.0021791
1: -0.0019017, 0.0002160, -0.0016852, 0.0002091, -0.0018152, 0.0016010
2: 0.0036350, 0.0046019, 0.0036351, 0.0044930, -0.0006971, 0.0008065
3: 0.0012745, 0.0026965, 0.0013414, 0.0027160, -0.0011182, 0.0010251
4: -0.0045055, -0.0024334, -0.0045424, -0.0026462, -0.0012815, 0.0015448
5: -0.0002841, 0.0009916, -0.0002769, 0.0008686, -0.0009791, 0.0010999
6: -0.0052833, -0.0018351, -0.0050336, -0.0017352, -0.0025392, 0.0021875
7: -0.0220911, -0.0101302, -0.0222938, -0.0113740, -0.0074424, 0.0089655
8: 0.9751749, 0.9863459, 0.9750398, 0.9851038, -0.0072095, 0.0086317
9: -0.0011036, 0.0067883, -0.0002691, 0.0069129, -0.0059528, 0.0049442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036257, upper bound: 0.0036242
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 118

Time for candidate selection: 4.74 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034314, upper bound: 0.0024851
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034240, upper bound: 0.0029998
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0151357, 0.0179420, -0.0024856, 0.0023620
1: -0.0019338, 0.0001350, -0.0018628, 0.0001298, -0.0018091, 0.0017409
2: 0.0036730, 0.0046160, 0.0036692, 0.0045765, -0.0007551, 0.0007975
3: 0.0012895, 0.0026941, 0.0013333, 0.0027430, -0.0011085, 0.0010334
4: -0.0044618, -0.0024138, -0.0045209, -0.0025180, -0.0013557, 0.0015167
5: -0.0002362, 0.0010116, -0.0002299, 0.0009735, -0.0010643, 0.0011011
6: -0.0052777, -0.0018874, -0.0051244, -0.0017411, -0.0024466, 0.0021651
7: -0.0218182, -0.0100149, -0.0221431, -0.0106132, -0.0078882, 0.0088010
8: 0.9755049, 0.9864652, 0.9752718, 0.9859145, -0.0077064, 0.0084776
9: -0.0011810, 0.0065963, -0.0007881, 0.0067982, -0.0058434, 0.0052538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043285, upper bound: 0.0041480
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0150178, 0.0179349, 0.0151825, 0.0180589, -0.0026053, 0.0023062
1: -0.0019338, 0.0001350, -0.0018284, 0.0002118, -0.0018906, 0.0017001
2: 0.0036730, 0.0046160, 0.0036312, 0.0045615, -0.0007373, 0.0008360
3: 0.0012895, 0.0026941, 0.0013163, 0.0027442, -0.0011174, 0.0010557
4: -0.0044618, -0.0024138, -0.0045628, -0.0025397, -0.0013322, 0.0015598
5: -0.0002362, 0.0010116, -0.0002787, 0.0009521, -0.0010395, 0.0011494
6: -0.0052777, -0.0018874, -0.0051308, -0.0016889, -0.0024963, 0.0021718
7: -0.0218182, -0.0100149, -0.0224076, -0.0107403, -0.0077457, 0.0090698
8: 0.9755049, 0.9864652, 0.9749451, 0.9857804, -0.0075473, 0.0088044
9: -0.0011810, 0.0065963, -0.0007032, 0.0069862, -0.0060336, 0.0051551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043285, upper bound: 0.0041480
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150622, 0.0180513, 0.0151357, 0.0179420, -0.0024302, 0.0024816
1: -0.0019017, 0.0002160, -0.0018628, 0.0001298, -0.0017703, 0.0018218
2: 0.0036350, 0.0046019, 0.0036692, 0.0045765, -0.0007937, 0.0007796
3: 0.0012745, 0.0026965, 0.0013333, 0.0027430, -0.0011285, 0.0010390
4: -0.0045055, -0.0024334, -0.0045209, -0.0025180, -0.0013989, 0.0014949
5: -0.0002841, 0.0009916, -0.0002299, 0.0009735, -0.0011123, 0.0010774
6: -0.0052833, -0.0018351, -0.0051244, -0.0017411, -0.0024522, 0.0022188
7: -0.0220911, -0.0101302, -0.0221431, -0.0106132, -0.0081555, 0.0086693
8: 0.9751749, 0.9863459, 0.9752718, 0.9859145, -0.0080301, 0.0083268
9: -0.0011036, 0.0067883, -0.0007881, 0.0067982, -0.0057520, 0.0054417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035184, upper bound: 0.0036203
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 118

Time for candidate selection: 4.68 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032940, upper bound: 0.0024792
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032841, upper bound: 0.0029969
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0150622, 0.0180513, 0.0151825, 0.0180589, -0.0024726, 0.0023520
1: -0.0019017, 0.0002160, -0.0018284, 0.0002118, -0.0017959, 0.0017289
2: 0.0036350, 0.0046019, 0.0036312, 0.0045615, -0.0007522, 0.0007938
3: 0.0012745, 0.0026965, 0.0013163, 0.0027442, -0.0011367, 0.0010582
4: -0.0045055, -0.0024334, -0.0045628, -0.0025397, -0.0013544, 0.0015163
5: -0.0002841, 0.0009916, -0.0002787, 0.0009521, -0.0010566, 0.0010920
6: -0.0052833, -0.0018351, -0.0051308, -0.0016889, -0.0025205, 0.0022434
7: -0.0220911, -0.0101302, -0.0224076, -0.0107403, -0.0078808, 0.0087982
8: 0.9751749, 0.9863459, 0.9749451, 0.9857804, -0.0076986, 0.0084700
9: -0.0011036, 0.0067883, -0.0007032, 0.0069862, -0.0058408, 0.0052484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035184, upper bound: 0.0036203
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 118

Time for candidate selection: 4.61 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032940, upper bound: 0.0024792
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032841, upper bound: 0.0029969
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0148334, 0.0177084, 0.0152650, 0.0179267, -0.0025891, 0.0019919
1: -0.0020734, -0.0000054, -0.0017794, 0.0001188, -0.0018991, 0.0015027
2: 0.0037482, 0.0046743, 0.0036745, 0.0045348, -0.0006324, 0.0008279
3: 0.0013223, 0.0027023, 0.0013600, 0.0027258, -0.0010445, 0.0010009
4: -0.0043014, -0.0023527, -0.0044946, -0.0025963, -0.0010740, 0.0014581
5: -0.0001566, 0.0010967, -0.0002228, 0.0009248, -0.0009275, 0.0011578
6: -0.0052530, -0.0020762, -0.0050280, -0.0017944, -0.0022689, 0.0018031
7: -0.0208803, -0.0096316, -0.0220005, -0.0110724, -0.0062394, 0.0084977
8: 0.9764035, 0.9869339, 0.9753920, 0.9854466, -0.0061279, 0.0083465
9: -0.0014532, 0.0059710, -0.0004785, 0.0067090, -0.0056721, 0.0041540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 4.09 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029898, upper bound: 0.0026192
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0028038, upper bound: 0.0032019
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0150200, 0.0177067, -0.0022125, 0.0023648
1: -0.0019155, 0.0001215, -0.0019467, -0.0000077, -0.0016534, 0.0017465
2: 0.0036728, 0.0045999, 0.0037482, 0.0046140, -0.0007551, 0.0007032
3: 0.0013332, 0.0027536, 0.0013484, 0.0026774, -0.0010186, 0.0010289
4: -0.0045141, -0.0024922, -0.0042883, -0.0024499, -0.0013505, 0.0011709
5: -0.0002245, 0.0010050, -0.0001555, 0.0010234, -0.0010684, 0.0010151
6: -0.0051258, -0.0017492, -0.0051619, -0.0021186, -0.0018753, 0.0021946
7: -0.0221041, -0.0104549, -0.0208082, -0.0102067, -0.0078627, 0.0068208
8: 0.9753104, 0.9861066, 0.9764539, 0.9863257, -0.0076792, 0.0067782
9: -0.0009001, 0.0067720, -0.0010620, 0.0059265, -0.0045558, 0.0052390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043419, upper bound: 0.0032589
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0152650, 0.0179267, -0.0023489, 0.0021354
1: -0.0019155, 0.0001215, -0.0017794, 0.0001188, -0.0017420, 0.0015915
2: 0.0036728, 0.0045999, 0.0036745, 0.0045348, -0.0006802, 0.0007489
3: 0.0013332, 0.0027536, 0.0013600, 0.0027258, -0.0010201, 0.0010037
4: -0.0045141, -0.0024922, -0.0044946, -0.0025963, -0.0011837, 0.0012850
5: -0.0002245, 0.0010050, -0.0002228, 0.0009248, -0.0009784, 0.0010669
6: -0.0051258, -0.0017492, -0.0050280, -0.0017944, -0.0020589, 0.0019858
7: -0.0221041, -0.0104549, -0.0220005, -0.0110724, -0.0068803, 0.0074893
8: 0.9753104, 0.9861066, 0.9753920, 0.9854466, -0.0067302, 0.0074007
9: -0.0009001, 0.0067720, -0.0004785, 0.0067090, -0.0049994, 0.0045788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043419, upper bound: 0.0049255
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0049254
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0148334, 0.0177084, 0.0150623, 0.0179310, -0.0025338, 0.0021812
1: -0.0020734, -0.0000054, -0.0019155, 0.0001215, -0.0018734, 0.0016407
2: 0.0037482, 0.0046743, 0.0036728, 0.0045999, -0.0006920, 0.0008083
3: 0.0013223, 0.0027023, 0.0013332, 0.0027536, -0.0010615, 0.0010357
4: -0.0043014, -0.0023527, -0.0045141, -0.0024922, -0.0011381, 0.0014027
5: -0.0001566, 0.0010967, -0.0002245, 0.0010050, -0.0010103, 0.0011449
6: -0.0052530, -0.0020762, -0.0051258, -0.0017492, -0.0022262, 0.0018525
7: -0.0208803, -0.0096316, -0.0221041, -0.0104549, -0.0066318, 0.0081780
8: 0.9764035, 0.9869339, 0.9753104, 0.9861066, -0.0065939, 0.0080332
9: -0.0014532, 0.0059710, -0.0009001, 0.0067720, -0.0054585, 0.0044307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.98 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026710, upper bound: 0.0026192
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024056, upper bound: 0.0032019
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0148334, 0.0177084, -0.0021812, 0.0025338
1: -0.0019155, 0.0001215, -0.0020734, -0.0000054, -0.0016407, 0.0018734
2: 0.0036728, 0.0045999, 0.0037482, 0.0046743, -0.0008083, 0.0006920
3: 0.0013332, 0.0027536, 0.0013223, 0.0027023, -0.0010357, 0.0010615
4: -0.0045141, -0.0024922, -0.0043014, -0.0023527, -0.0014027, 0.0011381
5: -0.0002245, 0.0010050, -0.0001566, 0.0010967, -0.0011449, 0.0010103
6: -0.0051258, -0.0017492, -0.0052530, -0.0020762, -0.0018525, 0.0022262
7: -0.0221041, -0.0104549, -0.0208803, -0.0096316, -0.0081780, 0.0066318
8: 0.9753104, 0.9861066, 0.9764035, 0.9869339, -0.0080332, 0.0065939
9: -0.0009001, 0.0067720, -0.0014532, 0.0059710, -0.0044307, 0.0054585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.99 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037168, upper bound: 0.0020077
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032859, upper bound: 0.0029243
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0150623, 0.0179310, 0.0150623, 0.0179310, -0.0023128, 0.0023128
1: -0.0019155, 0.0001215, -0.0019155, 0.0001215, -0.0017260, 0.0017260
2: 0.0036728, 0.0045999, 0.0036728, 0.0045999, -0.0007361, 0.0007361
3: 0.0013332, 0.0027536, 0.0013332, 0.0027536, -0.0010377, 0.0010377
4: -0.0045141, -0.0024922, -0.0045141, -0.0024922, -0.0012517, 0.0012517
5: -0.0002245, 0.0010050, -0.0002245, 0.0010050, -0.0010595, 0.0010595
6: -0.0051258, -0.0017492, -0.0051258, -0.0017492, -0.0020281, 0.0020281
7: -0.0221041, -0.0104549, -0.0221041, -0.0104549, -0.0072913, 0.0072913
8: 0.9753104, 0.9861066, 0.9753104, 0.9861066, -0.0072051, 0.0072051
9: -0.0009001, 0.0067720, -0.0009001, 0.0067720, -0.0048672, 0.0048672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 240
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 115
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 99
type: A, layer: 3, pos: 118

Time for candidate selection: 3.99 seconds

### Candidate
type: A, layer: 3, pos: 240

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037168, upper bound: 0.0020077
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0032859, upper bound: 0.0046611
time: 0.90 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.89 seconds
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0036218, upper bound: 0.0019997
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0033458, upper bound: 0.0028410
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0036218, upper bound: 0.0028330
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0033458, upper bound: 0.0045947
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0029144, upper bound: 0.0020200
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0028360, upper bound: 0.0034509
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0039569, upper bound: 0.0032484
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0039297, upper bound: 0.0031968
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0039569, upper bound: 0.0048743
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0039297, upper bound: 0.0048726
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0037628, upper bound: 0.0020451
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0034509, upper bound: 0.0029994
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0037628, upper bound: 0.0028330
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0034509, upper bound: 0.0046022
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0029674, upper bound: 0.0037477
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0029674, upper bound: 0.0037477
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0042061, upper bound: 0.0030572
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0038333, upper bound: 0.0029709
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0042061, upper bound: 0.0047273
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0038333, upper bound: 0.0047766
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0038083, upper bound: 0.0039873
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0038083, upper bound: 0.0039873
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0018134, upper bound: 0.0014940
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0015303, upper bound: 0.0016515
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0029600, upper bound: 0.0028647
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0027187, upper bound: 0.0045907
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0040974, upper bound: 0.0041310
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0027380, upper bound: 0.0027214
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0024832, upper bound: 0.0028058
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0036369, upper bound: 0.0020482
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0032230, upper bound: 0.0029973
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0036369, upper bound: 0.0028802
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0032230, upper bound: 0.0047438
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0042416, upper bound: 0.0036703
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0042416, upper bound: 0.0036703
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0035383, upper bound: 0.0017627
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0031466, upper bound: 0.0023956
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0035384, upper bound: 0.0028209
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0031466, upper bound: 0.0045374
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0040967, upper bound: 0.0040347
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0044306, upper bound: 0.0041627
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0044306, upper bound: 0.0041627
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0034314, upper bound: 0.0024851
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0034240, upper bound: 0.0029998
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0034314, upper bound: 0.0024851
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0034240, upper bound: 0.0029998
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0043285, upper bound: 0.0041480
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0043285, upper bound: 0.0041480
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0032940, upper bound: 0.0024792
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0032841, upper bound: 0.0029969
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0032940, upper bound: 0.0024792
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0032841, upper bound: 0.0029969
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0029898, upper bound: 0.0026192
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0028038, upper bound: 0.0032019
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0043419, upper bound: 0.0032589
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0043419, upper bound: 0.0049255
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0049254
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0026710, upper bound: 0.0026192
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0024056, upper bound: 0.0032019
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0037168, upper bound: 0.0020077
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0032859, upper bound: 0.0029243
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0037168, upper bound: 0.0020077
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.89
Output dim: 8, lower bound: -0.0032859, upper bound: 0.0046611

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0152036, 0.0189020, 0.0151593, 0.0179196, -0.0020946, 0.0032418
1: -0.0017996, 0.0008256, -0.0018354, 0.0001243, -0.0015503, 0.0023841
2: 0.0033628, 0.0045563, 0.0036782, 0.0045701, -0.0010347, 0.0006677
3: 0.0012775, 0.0025242, 0.0013192, 0.0026313, -0.0010969, 0.0009208
4: -0.0048545, -0.0025105, -0.0044398, -0.0024940, -0.0016095, 0.0011359
5: -0.0006466, 0.0009278, -0.0002291, 0.0009526, -0.0014521, 0.0009485
6: -0.0051786, -0.0018963, -0.0051837, -0.0019378, -0.0019005, 0.0019441
7: -0.0242151, -0.0105758, -0.0216961, -0.0104822, -0.0095007, 0.0066186
8: 0.9727498, 0.9858979, 0.9756122, 0.9859923, -0.0098106, 0.0065513
9: -0.0008068, 0.0082637, -0.0008686, 0.0065166, -0.0044184, 0.0064191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036520, upper bound: 0.0037420
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035608, upper bound: 0.0034019
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0151658, 0.0179188, 0.0152678, 0.0179266, -0.0021663, 0.0020757
1: -0.0018330, 0.0001233, -0.0017771, 0.0001187, -0.0015957, 0.0015493
2: 0.0036784, 0.0045681, 0.0036745, 0.0045339, -0.0006605, 0.0006929
3: 0.0013227, 0.0026614, 0.0013603, 0.0027242, -0.0010097, 0.0009324
4: -0.0044393, -0.0024932, -0.0044946, -0.0025973, -0.0011127, 0.0012712
5: -0.0002283, 0.0009516, -0.0002228, 0.0009233, -0.0009526, 0.0009762
6: -0.0051650, -0.0019526, -0.0050262, -0.0017959, -0.0020812, 0.0018044
7: -0.0216943, -0.0104762, -0.0220003, -0.0110778, -0.0064894, 0.0073739
8: 0.9756122, 0.9859844, 0.9753920, 0.9854401, -0.0064109, 0.0071216
9: -0.0008716, 0.0065158, -0.0004747, 0.0067089, -0.0048950, 0.0043345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040644, upper bound: 0.0043064
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0150789, 0.0179491, 0.0152807, 0.0179262, -0.0024346, 0.0020664
1: -0.0019011, 0.0001455, -0.0017669, 0.0001182, -0.0018122, 0.0015394
2: 0.0036687, 0.0045957, 0.0036746, 0.0045299, -0.0006581, 0.0007756
3: 0.0013232, 0.0026935, 0.0013621, 0.0027184, -0.0009936, 0.0010687
4: -0.0044591, -0.0024642, -0.0044945, -0.0026010, -0.0011260, 0.0013234
5: -0.0002417, 0.0009934, -0.0002224, 0.0009169, -0.0009454, 0.0011126
6: -0.0051488, -0.0019155, -0.0050131, -0.0018015, -0.0020517, 0.0019044
7: -0.0218081, -0.0102968, -0.0219999, -0.0110996, -0.0065555, 0.0077128
8: 0.9754973, 0.9861987, 0.9753920, 0.9854133, -0.0064349, 0.0075880
9: -0.0009964, 0.0065918, -0.0004591, 0.0067088, -0.0051438, 0.0043709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039473, upper bound: 0.0042129
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038717, upper bound: 0.0038995
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0153358, 0.0189675, 0.0151567, 0.0179196, -0.0020508, 0.0033139
1: -0.0017159, 0.0008633, -0.0018377, 0.0001243, -0.0015183, 0.0024256
2: 0.0033408, 0.0045125, 0.0036782, 0.0045709, -0.0010594, 0.0006538
3: 0.0013066, 0.0025598, 0.0013192, 0.0026368, -0.0010803, 0.0009620
4: -0.0049372, -0.0026239, -0.0044398, -0.0024930, -0.0017053, 0.0010991
5: -0.0006670, 0.0008809, -0.0002291, 0.0009542, -0.0014743, 0.0009287
6: -0.0050189, -0.0017508, -0.0051840, -0.0019378, -0.0018489, 0.0021721
7: -0.0246837, -0.0112311, -0.0216961, -0.0104758, -0.0100422, 0.0064110
8: 0.9723629, 0.9852887, 0.9756122, 0.9859977, -0.0102565, 0.0063736
9: -0.0003739, 0.0085651, -0.0008728, 0.0065166, -0.0042838, 0.0067676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039345, upper bound: 0.0040521
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038125, upper bound: 0.0036366
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0153367, 0.0179262, 0.0150200, 0.0177067, -0.0019229, 0.0023595
1: -0.0017278, 0.0001183, -0.0019467, -0.0000077, -0.0014515, 0.0017420
2: 0.0036747, 0.0045117, 0.0037482, 0.0046140, -0.0007531, 0.0006095
3: 0.0013604, 0.0027128, 0.0013484, 0.0026774, -0.0009740, 0.0009865
4: -0.0044946, -0.0026214, -0.0042883, -0.0024499, -0.0013048, 0.0010099
5: -0.0002226, 0.0008932, -0.0001555, 0.0010234, -0.0010652, 0.0008951
6: -0.0050262, -0.0017978, -0.0051619, -0.0021186, -0.0017249, 0.0021122
7: -0.0220004, -0.0112250, -0.0208082, -0.0102067, -0.0076058, 0.0058721
8: 0.9753920, 0.9852611, 0.9764539, 0.9863257, -0.0074837, 0.0057992
9: -0.0003713, 0.0067090, -0.0010620, 0.0059265, -0.0039134, 0.0050757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038333, upper bound: 0.0029709
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038333, upper bound: 0.0029709
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0153367, 0.0179262, 0.0152650, 0.0179267, -0.0020601, 0.0021300
1: -0.0017278, 0.0001183, -0.0017794, 0.0001188, -0.0015399, 0.0015898
2: 0.0036747, 0.0045117, 0.0036745, 0.0045348, -0.0006779, 0.0006556
3: 0.0013604, 0.0027128, 0.0013600, 0.0027258, -0.0009736, 0.0009611
4: -0.0044946, -0.0026214, -0.0044946, -0.0025963, -0.0011445, 0.0011190
5: -0.0002226, 0.0008932, -0.0002228, 0.0009248, -0.0009767, 0.0009466
6: -0.0050262, -0.0017978, -0.0050280, -0.0017944, -0.0019085, 0.0019092
7: -0.0220004, -0.0112250, -0.0220005, -0.0110724, -0.0066657, 0.0065105
8: 0.9753920, 0.9852611, 0.9753920, 0.9854466, -0.0065841, 0.0064002
9: -0.0003713, 0.0067090, -0.0004785, 0.0067090, -0.0043372, 0.0044459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048671, upper bound: 0.0047273
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048671, upper bound: 0.0047273
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0153948, 0.0180371, 0.0153120, 0.0179259, -0.0020555, 0.0021998
1: -0.0016837, 0.0002002, -0.0017432, 0.0001181, -0.0015369, 0.0016362
2: 0.0036390, 0.0044925, 0.0036747, 0.0045198, -0.0007005, 0.0006538
3: 0.0013416, 0.0027132, 0.0013605, 0.0027100, -0.0009844, 0.0009893
4: -0.0045349, -0.0026468, -0.0044946, -0.0026135, -0.0011680, 0.0011096
5: -0.0002714, 0.0008675, -0.0002224, 0.0009026, -0.0010037, 0.0009441
6: -0.0050330, -0.0017468, -0.0050262, -0.0017970, -0.0019055, 0.0019859
7: -0.0222496, -0.0113772, -0.0220004, -0.0111760, -0.0068149, 0.0064582
8: 0.9750794, 0.9851004, 0.9753920, 0.9853289, -0.0067882, 0.0063562
9: -0.0002669, 0.0068834, -0.0004066, 0.0067090, -0.0043040, 0.0045537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048671, upper bound: 0.0047766
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048671, upper bound: 0.0047766
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0152036, 0.0189020, 0.0150792, 0.0179310, -0.0022410, 0.0033426
1: -0.0017996, 0.0008256, -0.0019011, 0.0001215, -0.0016270, 0.0024590
2: 0.0033628, 0.0045563, 0.0036728, 0.0045946, -0.0010670, 0.0007186
3: 0.0012775, 0.0025242, 0.0013332, 0.0027238, -0.0011892, 0.0009342
4: -0.0048545, -0.0025105, -0.0045141, -0.0024972, -0.0016706, 0.0013400
5: -0.0006466, 0.0009278, -0.0002245, 0.0009945, -0.0014981, 0.0009868
6: -0.0051786, -0.0018963, -0.0051244, -0.0017492, -0.0022125, 0.0020391
7: -0.0242151, -0.0105758, -0.0221041, -0.0104852, -0.0098570, 0.0077779
8: 0.9727498, 0.9858979, 0.9753104, 0.9860689, -0.0101595, 0.0075367
9: -0.0008068, 0.0082637, -0.0008786, 0.0067720, -0.0051684, 0.0066585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033623, upper bound: 0.0035713
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032412, upper bound: 0.0032144
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0153367, 0.0179262, 0.0150178, 0.0179349, -0.0021977, 0.0023910
1: -0.0017278, 0.0001183, -0.0019338, 0.0001350, -0.0016182, 0.0017549
2: 0.0036747, 0.0045117, 0.0036730, 0.0046160, -0.0007651, 0.0007023
3: 0.0013604, 0.0027128, 0.0012895, 0.0026941, -0.0009989, 0.0010694
4: -0.0044946, -0.0026214, -0.0044618, -0.0024138, -0.0013968, 0.0012834
5: -0.0002226, 0.0008932, -0.0002362, 0.0010116, -0.0010713, 0.0009889
6: -0.0050262, -0.0017978, -0.0052777, -0.0018874, -0.0021106, 0.0022612
7: -0.0220004, -0.0112250, -0.0218182, -0.0100149, -0.0081124, 0.0074531
8: 0.9753920, 0.9852611, 0.9755049, 0.9864652, -0.0078746, 0.0072293
9: -0.0003713, 0.0067090, -0.0011810, 0.0065963, -0.0049529, 0.0053938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040902, upper bound: 0.0043516
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040580, upper bound: 0.0043479
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0153948, 0.0180371, 0.0150178, 0.0179349, -0.0021304, 0.0025051
1: -0.0016837, 0.0002002, -0.0019338, 0.0001350, -0.0015684, 0.0018372
2: 0.0036390, 0.0044925, 0.0036730, 0.0046160, -0.0008015, 0.0006810
3: 0.0013416, 0.0027132, 0.0012895, 0.0026941, -0.0010239, 0.0010760
4: -0.0045349, -0.0026468, -0.0044618, -0.0024138, -0.0014334, 0.0012571
5: -0.0002714, 0.0008675, -0.0002362, 0.0010116, -0.0011204, 0.0009587
6: -0.0050330, -0.0017468, -0.0052777, -0.0018874, -0.0021183, 0.0023056
7: -0.0222496, -0.0113772, -0.0218182, -0.0100149, -0.0083404, 0.0072945
8: 0.9750794, 0.9851004, 0.9755049, 0.9864652, -0.0081648, 0.0070486
9: -0.0002669, 0.0068834, -0.0011810, 0.0065963, -0.0048426, 0.0055567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040902, upper bound: 0.0043516
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040580, upper bound: 0.0043479
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0153367, 0.0179262, 0.0150622, 0.0180513, -0.0023219, 0.0023442
1: -0.0017278, 0.0001183, -0.0019017, 0.0002160, -0.0017054, 0.0017200
2: 0.0036747, 0.0045117, 0.0036350, 0.0046019, -0.0007502, 0.0007424
3: 0.0013604, 0.0027128, 0.0012745, 0.0026965, -0.0010037, 0.0010903
4: -0.0044946, -0.0026214, -0.0045055, -0.0024334, -0.0013772, 0.0013286
5: -0.0002226, 0.0008932, -0.0002841, 0.0009916, -0.0010505, 0.0010402
6: -0.0050262, -0.0017978, -0.0052833, -0.0018351, -0.0021633, 0.0022663
7: -0.0220004, -0.0112250, -0.0220911, -0.0101302, -0.0079952, 0.0077316
8: 0.9753920, 0.9852611, 0.9751749, 0.9863459, -0.0077462, 0.0075660
9: -0.0003713, 0.0067090, -0.0011036, 0.0067883, -0.0051488, 0.0053135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035588, upper bound: 0.0035266
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 118

Time for candidate selection: 4.66 seconds

### Candidate
type: B, layer: 3, pos: 240

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024305, upper bound: 0.0033455
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029421, upper bound: 0.0033384
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0153948, 0.0180371, 0.0150622, 0.0180513, -0.0021778, 0.0023809
1: -0.0016837, 0.0002002, -0.0019017, 0.0002160, -0.0015998, 0.0017440
2: 0.0036390, 0.0044925, 0.0036350, 0.0046019, -0.0007621, 0.0006967
3: 0.0013416, 0.0027132, 0.0012745, 0.0026965, -0.0010244, 0.0010940
4: -0.0045349, -0.0026468, -0.0045055, -0.0024334, -0.0013953, 0.0012810
5: -0.0002714, 0.0008675, -0.0002841, 0.0009916, -0.0010635, 0.0009784
6: -0.0050330, -0.0017468, -0.0052833, -0.0018351, -0.0021869, 0.0023243
7: -0.0222496, -0.0113772, -0.0220911, -0.0101302, -0.0081031, 0.0074395
8: 0.9750794, 0.9851004, 0.9751749, 0.9863459, -0.0078638, 0.0072064
9: -0.0002669, 0.0068834, -0.0011036, 0.0067883, -0.0049423, 0.0053881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035588, upper bound: 0.0035266
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 118

Time for candidate selection: 4.62 seconds

### Candidate
type: B, layer: 3, pos: 240

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024305, upper bound: 0.0033455
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029421, upper bound: 0.0033384
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0153358, 0.0189675, 0.0150874, 0.0179310, -0.0020718, 0.0033952
1: -0.0017159, 0.0008633, -0.0018946, 0.0001215, -0.0015261, 0.0024906
2: 0.0033408, 0.0045125, 0.0036728, 0.0045920, -0.0010846, 0.0006617
3: 0.0013066, 0.0025598, 0.0013332, 0.0027085, -0.0011225, 0.0009495
4: -0.0049372, -0.0026239, -0.0045141, -0.0024998, -0.0017176, 0.0011738
5: -0.0006670, 0.0008809, -0.0002245, 0.0009902, -0.0015154, 0.0009327
6: -0.0050189, -0.0017508, -0.0051237, -0.0017492, -0.0019805, 0.0020958
7: -0.0246837, -0.0112311, -0.0221041, -0.0105005, -0.0101278, 0.0068250
8: 0.9723629, 0.9852887, 0.9753104, 0.9860533, -0.0104065, 0.0066642
9: -0.0003739, 0.0085651, -0.0008684, 0.0067720, -0.0045436, 0.0068362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042253, upper bound: 0.0044071
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041916, upper bound: 0.0041359
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0150471, 0.0179342, 0.0152099, 0.0179310, -0.0024301, 0.0022577
1: -0.0019105, 0.0001341, -0.0018032, 0.0001322, -0.0017691, 0.0016460
2: 0.0036732, 0.0046070, 0.0036744, 0.0045539, -0.0007236, 0.0007787
3: 0.0012930, 0.0026768, 0.0013197, 0.0026667, -0.0009929, 0.0009709
4: -0.0044613, -0.0024215, -0.0044474, -0.0025105, -0.0013604, 0.0014439
5: -0.0002355, 0.0009970, -0.0002345, 0.0009341, -0.0010017, 0.0010743
6: -0.0052567, -0.0019015, -0.0051826, -0.0019301, -0.0021951, 0.0021270
7: -0.0218165, -0.0100595, -0.0217400, -0.0105853, -0.0079026, 0.0083916
8: 0.9755049, 0.9864109, 0.9755719, 0.9858533, -0.0076241, 0.0081454
9: -0.0011490, 0.0065955, -0.0007947, 0.0065454, -0.0055819, 0.0052521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0149659, 0.0179636, 0.0152219, 0.0179307, -0.0026804, 0.0022395
1: -0.0019734, 0.0001561, -0.0017937, 0.0001318, -0.0019703, 0.0016316
2: 0.0036634, 0.0046324, 0.0036745, 0.0045502, -0.0007180, 0.0008563
3: 0.0012950, 0.0027067, 0.0013216, 0.0026611, -0.0009759, 0.0010983
4: -0.0044820, -0.0023963, -0.0044472, -0.0025139, -0.0013681, 0.0014909
5: -0.0002485, 0.0010360, -0.0002342, 0.0009281, -0.0009924, 0.0012017
6: -0.0052371, -0.0018648, -0.0051699, -0.0019353, -0.0021615, 0.0022096
7: -0.0219363, -0.0099024, -0.0217395, -0.0106046, -0.0079381, 0.0087023
8: 0.9753947, 0.9866004, 0.9755719, 0.9858291, -0.0076171, 0.0085833
9: -0.0012596, 0.0066729, -0.0007809, 0.0065453, -0.0058102, 0.0052684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0150471, 0.0179342, 0.0152683, 0.0180416, -0.0025518, 0.0021956
1: -0.0019105, 0.0001341, -0.0017612, 0.0002133, -0.0018546, 0.0015988
2: 0.0036732, 0.0046070, 0.0036390, 0.0045351, -0.0007038, 0.0008180
3: 0.0012930, 0.0026768, 0.0013014, 0.0026677, -0.0009969, 0.0009940
4: -0.0044613, -0.0024215, -0.0044882, -0.0025344, -0.0013355, 0.0014850
5: -0.0002355, 0.0009970, -0.0002824, 0.0009089, -0.0009722, 0.0011248
6: -0.0052567, -0.0019015, -0.0051886, -0.0018781, -0.0022481, 0.0021333
7: -0.0218165, -0.0100595, -0.0219933, -0.0107270, -0.0077510, 0.0086458
8: 0.9755049, 0.9864109, 0.9752560, 0.9856955, -0.0074515, 0.0084584
9: -0.0011490, 0.0065955, -0.0006952, 0.0067256, -0.0057612, 0.0051461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149659, 0.0179636, 0.0152837, 0.0180412, -0.0028009, 0.0021720
1: -0.0019734, 0.0001561, -0.0017492, 0.0002129, -0.0020550, 0.0015802
2: 0.0036634, 0.0046324, 0.0036391, 0.0045303, -0.0006965, 0.0008952
3: 0.0012950, 0.0027067, 0.0013034, 0.0026614, -0.0009790, 0.0011212
4: -0.0044820, -0.0023963, -0.0044880, -0.0025387, -0.0013428, 0.0015320
5: -0.0002485, 0.0010360, -0.0002821, 0.0009013, -0.0009608, 0.0012516
6: -0.0052371, -0.0018648, -0.0051759, -0.0018838, -0.0022150, 0.0022161
7: -0.0219363, -0.0099024, -0.0219928, -0.0107523, -0.0077850, 0.0089565
8: 0.9753947, 0.9866004, 0.9752560, 0.9856628, -0.0074405, 0.0088951
9: -0.0012596, 0.0066729, -0.0006767, 0.0067254, -0.0059895, 0.0051613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151415, 0.0189796, 0.0151548, 0.0179196, -0.0023087, 0.0033305
1: -0.0018513, 0.0008685, -0.0018393, 0.0001243, -0.0016968, 0.0024342
2: 0.0033361, 0.0045752, 0.0036782, 0.0045715, -0.0010652, 0.0007375
3: 0.0012879, 0.0025854, 0.0013192, 0.0026407, -0.0010949, 0.0009844
4: -0.0049622, -0.0025174, -0.0044398, -0.0024923, -0.0017413, 0.0012564
5: -0.0006693, 0.0009620, -0.0002291, 0.0009553, -0.0014790, 0.0010342
6: -0.0051177, -0.0017053, -0.0051841, -0.0019378, -0.0020001, 0.0022475
7: -0.0248291, -0.0105978, -0.0216961, -0.0104713, -0.0102399, 0.0073368
8: 0.9722317, 0.9859463, 0.9756122, 0.9860014, -0.0103930, 0.0073056
9: -0.0008056, 0.0086642, -0.0008759, 0.0065166, -0.0049076, 0.0068902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036626, upper bound: 0.0037094
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034425, upper bound: 0.0031953
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0151374, 0.0179305, 0.0150178, 0.0179349, -0.0023607, 0.0023568
1: -0.0018612, 0.0001210, -0.0019338, 0.0001350, -0.0017399, 0.0017395
2: 0.0036730, 0.0045760, 0.0036730, 0.0046160, -0.0007528, 0.0007547
3: 0.0013335, 0.0027404, 0.0012895, 0.0026941, -0.0010326, 0.0010866
4: -0.0045141, -0.0025186, -0.0044618, -0.0024138, -0.0013644, 0.0013553
5: -0.0002242, 0.0009724, -0.0002362, 0.0010116, -0.0010642, 0.0010637
6: -0.0051239, -0.0017525, -0.0052777, -0.0018874, -0.0021645, 0.0022348
7: -0.0221041, -0.0106164, -0.0218182, -0.0100149, -0.0079161, 0.0078854
8: 0.9753104, 0.9859110, 0.9755049, 0.9864652, -0.0076851, 0.0077032
9: -0.0007858, 0.0067720, -0.0011810, 0.0065963, -0.0052519, 0.0052605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040758, upper bound: 0.0042134
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040392, upper bound: 0.0042060
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0151840, 0.0180473, 0.0150178, 0.0179349, -0.0023050, 0.0024663
1: -0.0018269, 0.0002032, -0.0019338, 0.0001350, -0.0016991, 0.0018177
2: 0.0036350, 0.0045610, 0.0036730, 0.0046160, -0.0007885, 0.0007369
3: 0.0013165, 0.0027414, 0.0012895, 0.0026941, -0.0010550, 0.0010944
4: -0.0045562, -0.0025402, -0.0044618, -0.0024138, -0.0014050, 0.0013317
5: -0.0002732, 0.0009511, -0.0002362, 0.0010116, -0.0011114, 0.0010389
6: -0.0051304, -0.0017003, -0.0052777, -0.0018874, -0.0021712, 0.0022821
7: -0.0223683, -0.0107432, -0.0218182, -0.0100149, -0.0081700, 0.0077430
8: 0.9749826, 0.9857771, 0.9755049, 0.9864652, -0.0079886, 0.0075445
9: -0.0007011, 0.0069603, -0.0011810, 0.0065963, -0.0051533, 0.0054403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040758, upper bound: 0.0042134
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0040392, upper bound: 0.0042060
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0151374, 0.0179305, 0.0150622, 0.0180513, -0.0024803, 0.0023062
1: -0.0018612, 0.0001210, -0.0019017, 0.0002160, -0.0018207, 0.0017024
2: 0.0036730, 0.0045760, 0.0036350, 0.0046019, -0.0007368, 0.0007933
3: 0.0013335, 0.0027404, 0.0012745, 0.0026965, -0.0010382, 0.0011084
4: -0.0045141, -0.0025186, -0.0045055, -0.0024334, -0.0013448, 0.0013984
5: -0.0002242, 0.0009724, -0.0002841, 0.0009916, -0.0010418, 0.0011116
6: -0.0051239, -0.0017525, -0.0052833, -0.0018351, -0.0022182, 0.0022411
7: -0.0221041, -0.0106164, -0.0220911, -0.0101302, -0.0077996, 0.0081527
8: 0.9753104, 0.9859110, 0.9751749, 0.9863459, -0.0075530, 0.0080269
9: -0.0007858, 0.0067720, -0.0011036, 0.0067883, -0.0054398, 0.0051802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035555, upper bound: 0.0034231
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 118

Time for candidate selection: 4.60 seconds

### Candidate
type: B, layer: 3, pos: 240

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024265, upper bound: 0.0032079
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029406, upper bound: 0.0031988
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151840, 0.0180473, 0.0150622, 0.0180513, -0.0023507, 0.0023506
1: -0.0018269, 0.0002032, -0.0019017, 0.0002160, -0.0017278, 0.0017312
2: 0.0036350, 0.0045610, 0.0036350, 0.0046019, -0.0007511, 0.0007518
3: 0.0013165, 0.0027414, 0.0012745, 0.0026965, -0.0010575, 0.0011125
4: -0.0045562, -0.0025402, -0.0045055, -0.0024334, -0.0013634, 0.0013539
5: -0.0002732, 0.0009511, -0.0002841, 0.0009916, -0.0010585, 0.0010559
6: -0.0051304, -0.0017003, -0.0052833, -0.0018351, -0.0022428, 0.0023027
7: -0.0223683, -0.0107432, -0.0220911, -0.0101302, -0.0079123, 0.0078781
8: 0.9749826, 0.9857771, 0.9751749, 0.9863459, -0.0076799, 0.0076957
9: -0.0007011, 0.0069603, -0.0011036, 0.0067883, -0.0052466, 0.0052581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035555, upper bound: 0.0034231
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 240
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 115
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 99
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 118

Time for candidate selection: 4.97 seconds

### Candidate
type: B, layer: 3, pos: 240

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0024265, upper bound: 0.0032079
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029406, upper bound: 0.0031988
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0150471, 0.0179342, 0.0153379, 0.0179382, -0.0024400, 0.0021431
1: -0.0019105, 0.0001341, -0.0017271, 0.0001269, -0.0017628, 0.0015706
2: 0.0036732, 0.0046070, 0.0036708, 0.0045113, -0.0006858, 0.0007840
3: 0.0012930, 0.0026768, 0.0013605, 0.0027138, -0.0010521, 0.0009538
4: -0.0044613, -0.0024215, -0.0045018, -0.0026218, -0.0012821, 0.0015300
5: -0.0002355, 0.0009970, -0.0002283, 0.0008927, -0.0009582, 0.0010676
6: -0.0052567, -0.0019015, -0.0050248, -0.0017873, -0.0024134, 0.0020612
7: -0.0218165, -0.0100595, -0.0220422, -0.0112272, -0.0074451, 0.0088673
8: 0.9755049, 0.9864109, 0.9753515, 0.9852582, -0.0071781, 0.0084844
9: -0.0011490, 0.0065955, -0.0003697, 0.0067371, -0.0058796, 0.0049464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035734, upper bound: 0.0025603
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049768, upper bound: 0.0049031
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0149659, 0.0179636, 0.0153508, 0.0179378, -0.0026904, 0.0021241
1: -0.0019734, 0.0001561, -0.0017169, 0.0001264, -0.0019640, 0.0015565
2: 0.0036634, 0.0046324, 0.0036709, 0.0045074, -0.0006800, 0.0008616
3: 0.0012950, 0.0027067, 0.0013623, 0.0027080, -0.0010346, 0.0010826
4: -0.0044820, -0.0023963, -0.0045017, -0.0026258, -0.0012900, 0.0015771
5: -0.0002485, 0.0010360, -0.0002279, 0.0008863, -0.0009493, 0.0011947
6: -0.0052371, -0.0018648, -0.0050117, -0.0017929, -0.0023761, 0.0021476
7: -0.0219363, -0.0099024, -0.0220418, -0.0112503, -0.0074829, 0.0091781
8: 0.9753947, 0.9866004, 0.9753515, 0.9852301, -0.0071749, 0.0089224
9: -0.0012596, 0.0066729, -0.0003533, 0.0067370, -0.0061077, 0.0049643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035489, upper bound: 0.0025020
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049698, upper bound: 0.0048978
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0150471, 0.0179342, 0.0153958, 0.0180489, -0.0025617, 0.0020758
1: -0.0019105, 0.0001341, -0.0016830, 0.0002090, -0.0018480, 0.0015209
2: 0.0036732, 0.0046070, 0.0036351, 0.0044921, -0.0006645, 0.0008232
3: 0.0012930, 0.0026768, 0.0013417, 0.0027145, -0.0010600, 0.0009789
4: -0.0044613, -0.0024215, -0.0045424, -0.0026472, -0.0012558, 0.0015706
5: -0.0002355, 0.0009970, -0.0002769, 0.0008672, -0.0009280, 0.0011182
6: -0.0052567, -0.0019015, -0.0050318, -0.0017365, -0.0024616, 0.0020690
7: -0.0218165, -0.0100595, -0.0222936, -0.0113798, -0.0072870, 0.0091216
8: 0.9755049, 0.9864109, 0.9750398, 0.9850971, -0.0069986, 0.0088053
9: -0.0011490, 0.0065955, -0.0002650, 0.0069129, -0.0060607, 0.0048364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149659, 0.0179636, 0.0154112, 0.0180486, -0.0028108, 0.0020540
1: -0.0019734, 0.0001561, -0.0016708, 0.0002085, -0.0020486, 0.0015031
2: 0.0036634, 0.0046324, 0.0036352, 0.0044873, -0.0006578, 0.0009003
3: 0.0012950, 0.0027067, 0.0013436, 0.0027083, -0.0010421, 0.0011075
4: -0.0044820, -0.0023963, -0.0045422, -0.0026522, -0.0012631, 0.0016176
5: -0.0002485, 0.0010360, -0.0002765, 0.0008596, -0.0009165, 0.0012450
6: -0.0052371, -0.0018648, -0.0050189, -0.0017422, -0.0024248, 0.0021552
7: -0.0219363, -0.0099024, -0.0222932, -0.0114093, -0.0073191, 0.0094324
8: 0.9753947, 0.9866004, 0.9750398, 0.9850608, -0.0069880, 0.0092432
9: -0.0012596, 0.0066729, -0.0002441, 0.0069127, -0.0062889, 0.0048509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0150471, 0.0179342, 0.0151385, 0.0179420, -0.0023990, 0.0023058
1: -0.0019105, 0.0001341, -0.0018605, 0.0001298, -0.0017403, 0.0016927
2: 0.0036732, 0.0046070, 0.0036692, 0.0045756, -0.0007381, 0.0007706
3: 0.0012930, 0.0026768, 0.0013337, 0.0027414, -0.0010700, 0.0009882
4: -0.0044613, -0.0024215, -0.0045208, -0.0025188, -0.0013539, 0.0015011
5: -0.0002355, 0.0009970, -0.0002298, 0.0009720, -0.0010341, 0.0010577
6: -0.0052567, -0.0019015, -0.0051225, -0.0017424, -0.0023908, 0.0021134
7: -0.0218165, -0.0100595, -0.0221430, -0.0106180, -0.0078775, 0.0086996
8: 0.9755049, 0.9864109, 0.9752718, 0.9859086, -0.0076500, 0.0083213
9: -0.0011490, 0.0065955, -0.0007847, 0.0067982, -0.0057672, 0.0052454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0149659, 0.0179636, 0.0151520, 0.0179416, -0.0026498, 0.0022864
1: -0.0019734, 0.0001561, -0.0018498, 0.0001292, -0.0019420, 0.0016760
2: 0.0036634, 0.0046324, 0.0036693, 0.0045715, -0.0007320, 0.0008479
3: 0.0012950, 0.0027067, 0.0013358, 0.0027351, -0.0010525, 0.0011172
4: -0.0044820, -0.0023963, -0.0045207, -0.0025228, -0.0013606, 0.0015503
5: -0.0002485, 0.0010360, -0.0002294, 0.0009652, -0.0010231, 0.0011843
6: -0.0052371, -0.0018648, -0.0051088, -0.0017474, -0.0023579, 0.0022015
7: -0.0219363, -0.0099024, -0.0221426, -0.0106408, -0.0079051, 0.0090217
8: 0.9753947, 0.9866004, 0.9752718, 0.9858814, -0.0076338, 0.0087740
9: -0.0012596, 0.0066729, -0.0007686, 0.0067980, -0.0060042, 0.0052552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0150471, 0.0179342, 0.0151853, 0.0180589, -0.0025187, 0.0022498
1: -0.0019105, 0.0001341, -0.0018261, 0.0002117, -0.0018219, 0.0016519
2: 0.0036732, 0.0046070, 0.0036313, 0.0045607, -0.0007202, 0.0008091
3: 0.0012930, 0.0026768, 0.0013166, 0.0027426, -0.0010787, 0.0010105
4: -0.0044613, -0.0024215, -0.0045628, -0.0025406, -0.0013303, 0.0015442
5: -0.0002355, 0.0009970, -0.0002786, 0.0009506, -0.0010092, 0.0011060
6: -0.0052567, -0.0019015, -0.0051290, -0.0016902, -0.0024406, 0.0021202
7: -0.0218165, -0.0100595, -0.0224075, -0.0107454, -0.0077349, 0.0089684
8: 0.9755049, 0.9864109, 0.9749451, 0.9857740, -0.0074904, 0.0086483
9: -0.0011490, 0.0065955, -0.0006995, 0.0069861, -0.0059574, 0.0051466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0149659, 0.0179636, 0.0152003, 0.0180585, -0.0027678, 0.0022282
1: -0.0019734, 0.0001561, -0.0018141, 0.0002113, -0.0020231, 0.0016323
2: 0.0036634, 0.0046324, 0.0036313, 0.0045560, -0.0007135, 0.0008860
3: 0.0012950, 0.0027067, 0.0013187, 0.0027362, -0.0010607, 0.0011390
4: -0.0044820, -0.0023963, -0.0045627, -0.0025452, -0.0013371, 0.0015934
5: -0.0002485, 0.0010360, -0.0002782, 0.0009431, -0.0009964, 0.0012325
6: -0.0052371, -0.0018648, -0.0051153, -0.0016953, -0.0024078, 0.0022083
7: -0.0219363, -0.0099024, -0.0224070, -0.0107724, -0.0077637, 0.0092905
8: 0.9753947, 0.9866004, 0.9749451, 0.9857414, -0.0074738, 0.0091002
9: -0.0012596, 0.0066729, -0.0006804, 0.0069859, -0.0061945, 0.0051575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0150926, 0.0179303, 0.0150229, 0.0177066, -0.0021261, 0.0022988
1: -0.0018911, 0.0001205, -0.0019443, -0.0000078, -0.0015843, 0.0016929
2: 0.0036730, 0.0045904, 0.0037482, 0.0046131, -0.0007350, 0.0006764
3: 0.0013366, 0.0027359, 0.0013487, 0.0026756, -0.0009782, 0.0009825
4: -0.0045137, -0.0025002, -0.0042882, -0.0024505, -0.0013479, 0.0011559
5: -0.0002238, 0.0009893, -0.0001555, 0.0010219, -0.0010342, 0.0009714
6: -0.0051051, -0.0017639, -0.0051599, -0.0021201, -0.0018192, 0.0021481
7: -0.0221025, -0.0105018, -0.0208080, -0.0102108, -0.0078368, 0.0067228
8: 0.9753104, 0.9860511, 0.9764539, 0.9863201, -0.0075942, 0.0066235
9: -0.0008679, 0.0067713, -0.0010590, 0.0059264, -0.0044816, 0.0052125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0150086, 0.0179592, 0.0150346, 0.0177062, -0.0023861, 0.0022829
1: -0.0019559, 0.0001426, -0.0019347, -0.0000083, -0.0017938, 0.0016770
2: 0.0036634, 0.0046169, 0.0037483, 0.0046094, -0.0007301, 0.0007571
3: 0.0013383, 0.0027653, 0.0013506, 0.0026693, -0.0009597, 0.0011130
4: -0.0045339, -0.0024746, -0.0042881, -0.0024530, -0.0013561, 0.0012036
5: -0.0002370, 0.0010293, -0.0001551, 0.0010157, -0.0010240, 0.0011036
6: -0.0050850, -0.0017254, -0.0051476, -0.0021258, -0.0017850, 0.0022385
7: -0.0222203, -0.0103407, -0.0208075, -0.0102259, -0.0078745, 0.0070400
8: 0.9751989, 0.9862486, 0.9764539, 0.9862986, -0.0075895, 0.0070670
9: -0.0009824, 0.0068496, -0.0010478, 0.0059262, -0.0047154, 0.0052294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0150926, 0.0179303, 0.0152678, 0.0179266, -0.0022619, 0.0020820
1: -0.0018911, 0.0001205, -0.0017771, 0.0001187, -0.0016730, 0.0015452
2: 0.0036730, 0.0045904, 0.0036745, 0.0045339, -0.0006640, 0.0007219
3: 0.0013366, 0.0027359, 0.0013603, 0.0027242, -0.0009807, 0.0009589
4: -0.0045137, -0.0025002, -0.0044946, -0.0025973, -0.0011818, 0.0012692
5: -0.0002238, 0.0009893, -0.0002228, 0.0009233, -0.0009489, 0.0010236
6: -0.0051051, -0.0017639, -0.0050262, -0.0017959, -0.0020009, 0.0019353
7: -0.0221025, -0.0105018, -0.0220003, -0.0110778, -0.0068689, 0.0073867
8: 0.9753104, 0.9860511, 0.9753920, 0.9854401, -0.0066792, 0.0072419
9: -0.0008679, 0.0067713, -0.0004747, 0.0067089, -0.0049221, 0.0045699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046087, upper bound: 0.0046186
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046047, upper bound: 0.0044305
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0150086, 0.0179592, 0.0152807, 0.0179262, -0.0025209, 0.0020717
1: -0.0019559, 0.0001426, -0.0017669, 0.0001182, -0.0018824, 0.0015348
2: 0.0036634, 0.0046169, 0.0036746, 0.0045299, -0.0006610, 0.0008022
3: 0.0013383, 0.0027653, 0.0013621, 0.0027184, -0.0009634, 0.0010918
4: -0.0045339, -0.0024746, -0.0044945, -0.0026010, -0.0011921, 0.0013148
5: -0.0002370, 0.0010293, -0.0002224, 0.0009169, -0.0009410, 0.0011554
6: -0.0050850, -0.0017254, -0.0050131, -0.0018015, -0.0019704, 0.0020384
7: -0.0222203, -0.0103407, -0.0219999, -0.0110996, -0.0069213, 0.0076887
8: 0.9751989, 0.9862486, 0.9753920, 0.9854133, -0.0067009, 0.0076836
9: -0.0009824, 0.0068496, -0.0004591, 0.0067088, -0.0051475, 0.0045987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045861, upper bound: 0.0045980
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045846, upper bound: 0.0044073
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0151415, 0.0189796, 0.0150909, 0.0179310, -0.0022134, 0.0033599
1: -0.0018513, 0.0008685, -0.0018919, 0.0001215, -0.0016412, 0.0024666
2: 0.0033361, 0.0045752, 0.0036728, 0.0045909, -0.0010729, 0.0007060
3: 0.0012879, 0.0025854, 0.0013332, 0.0027021, -0.0011272, 0.0009760
4: -0.0049622, -0.0025174, -0.0045141, -0.0025008, -0.0016908, 0.0012198
5: -0.0006693, 0.0009620, -0.0002245, 0.0009884, -0.0015013, 0.0010051
6: -0.0051177, -0.0017053, -0.0051234, -0.0017492, -0.0020209, 0.0020642
7: -0.0248291, -0.0105978, -0.0221041, -0.0105069, -0.0099692, 0.0071025
8: 0.9722317, 0.9859463, 0.9753104, 0.9860469, -0.0102462, 0.0069933
9: -0.0008056, 0.0086642, -0.0008642, 0.0067720, -0.0047376, 0.0067294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042334, upper bound: 0.0043119
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0042007, upper bound: 0.0040639
time: 0.98 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0036520, upper bound: 0.0037420
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0035608, upper bound: 0.0034019
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040644, upper bound: 0.0043064
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0039473, upper bound: 0.0042129
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0038717, upper bound: 0.0038995
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0039345, upper bound: 0.0040521
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0038125, upper bound: 0.0036366
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0038333, upper bound: 0.0029709
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0038333, upper bound: 0.0029709
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0048671, upper bound: 0.0047273
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0048671, upper bound: 0.0047273
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0048671, upper bound: 0.0047766
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0048671, upper bound: 0.0047766
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0033623, upper bound: 0.0035713
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0032412, upper bound: 0.0032144
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040902, upper bound: 0.0043516
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040580, upper bound: 0.0043479
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040902, upper bound: 0.0043516
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040580, upper bound: 0.0043479
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0024305, upper bound: 0.0033455
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0029421, upper bound: 0.0033384
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0024305, upper bound: 0.0033455
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0029421, upper bound: 0.0033384
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0042253, upper bound: 0.0044071
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0041916, upper bound: 0.0041359
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0036626, upper bound: 0.0037094
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0034425, upper bound: 0.0031953
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040758, upper bound: 0.0042134
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040392, upper bound: 0.0042060
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040758, upper bound: 0.0042134
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0040392, upper bound: 0.0042060
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0024265, upper bound: 0.0032079
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0029406, upper bound: 0.0031988
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0024265, upper bound: 0.0032079
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0029406, upper bound: 0.0031988
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0035734, upper bound: 0.0025603
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0049768, upper bound: 0.0049031
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0035489, upper bound: 0.0025020
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0049698, upper bound: 0.0048978
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0046087, upper bound: 0.0046186
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0046047, upper bound: 0.0044305
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0045861, upper bound: 0.0045980
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0045846, upper bound: 0.0044073
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0042334, upper bound: 0.0043119
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.29
Output dim: 8, lower bound: -0.0042007, upper bound: 0.0040639

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0151658, 0.0179188, 0.0153397, 0.0179261, -0.0021641, 0.0020052
1: -0.0018330, 0.0001233, -0.0017254, 0.0001182, -0.0015934, 0.0014975
2: 0.0036784, 0.0045681, 0.0036747, 0.0045108, -0.0006381, 0.0006922
3: 0.0013227, 0.0026614, 0.0013607, 0.0027112, -0.0009936, 0.0009291
4: -0.0044393, -0.0024932, -0.0044946, -0.0026223, -0.0010875, 0.0012711
5: -0.0002283, 0.0009516, -0.0002225, 0.0008916, -0.0009211, 0.0009746
6: -0.0051650, -0.0019526, -0.0050243, -0.0017993, -0.0020713, 0.0017967
7: -0.0216943, -0.0104762, -0.0220003, -0.0112306, -0.0063356, 0.0073738
8: 0.9756122, 0.9859844, 0.9753920, 0.9852543, -0.0062286, 0.0071216
9: -0.0008716, 0.0065158, -0.0003674, 0.0067089, -0.0048950, 0.0042263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0152127, 0.0179180, 0.0153975, 0.0180370, -0.0022323, 0.0020014
1: -0.0017978, 0.0001225, -0.0016814, 0.0002001, -0.0016407, 0.0014945
2: 0.0036786, 0.0045533, 0.0036391, 0.0044916, -0.0006368, 0.0007142
3: 0.0013233, 0.0026449, 0.0013419, 0.0027117, -0.0010224, 0.0009403
4: -0.0044392, -0.0025101, -0.0045348, -0.0026477, -0.0010787, 0.0012912
5: -0.0002278, 0.0009297, -0.0002714, 0.0008661, -0.0009188, 0.0010023
6: -0.0051631, -0.0019550, -0.0050312, -0.0017481, -0.0021357, 0.0017940
7: -0.0216942, -0.0105780, -0.0222494, -0.0113830, -0.0062866, 0.0075016
8: 0.9756122, 0.9858668, 0.9750794, 0.9850936, -0.0061904, 0.0072951
9: -0.0008000, 0.0065158, -0.0002628, 0.0068833, -0.0049880, 0.0041962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0150789, 0.0179491, 0.0153526, 0.0179257, -0.0024324, 0.0019968
1: -0.0019011, 0.0001455, -0.0017153, 0.0001177, -0.0018099, 0.0014885
2: 0.0036687, 0.0045957, 0.0036748, 0.0045068, -0.0006360, 0.0007750
3: 0.0013232, 0.0026935, 0.0013625, 0.0027054, -0.0009776, 0.0010654
4: -0.0044591, -0.0024642, -0.0044944, -0.0026263, -0.0011010, 0.0013234
5: -0.0002417, 0.0009934, -0.0002221, 0.0008852, -0.0009145, 0.0011110
6: -0.0051488, -0.0019155, -0.0050113, -0.0018046, -0.0020417, 0.0018966
7: -0.0218081, -0.0102968, -0.0219999, -0.0112535, -0.0064023, 0.0077127
8: 0.9754973, 0.9861987, 0.9753920, 0.9852266, -0.0062527, 0.0075880
9: -0.0009964, 0.0065918, -0.0003511, 0.0067088, -0.0051438, 0.0042628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038717, upper bound: 0.0038995
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038717, upper bound: 0.0038995
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0153358, 0.0189675, 0.0152299, 0.0179191, -0.0020486, 0.0032392
1: -0.0017159, 0.0008633, -0.0017834, 0.0001239, -0.0015159, 0.0023698
2: 0.0033408, 0.0045125, 0.0036783, 0.0045476, -0.0010357, 0.0006531
3: 0.0013066, 0.0025598, 0.0013196, 0.0026239, -0.0010628, 0.0009585
4: -0.0049372, -0.0026239, -0.0044397, -0.0025189, -0.0016789, 0.0010991
5: -0.0006670, 0.0008809, -0.0002288, 0.0009212, -0.0014402, 0.0009271
6: -0.0050189, -0.0017508, -0.0051824, -0.0019410, -0.0018419, 0.0021637
7: -0.0246837, -0.0112311, -0.0216961, -0.0106328, -0.0098807, 0.0064110
8: 0.9723629, 0.9852887, 0.9756122, 0.9858075, -0.0100633, 0.0063736
9: -0.0003739, 0.0085651, -0.0007627, 0.0065166, -0.0042838, 0.0066539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038125, upper bound: 0.0036366
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038125, upper bound: 0.0036366
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0153367, 0.0179262, 0.0153367, 0.0179262, -0.0020579, 0.0020579
1: -0.0017278, 0.0001183, -0.0017278, 0.0001183, -0.0015376, 0.0015376
2: 0.0036747, 0.0045117, 0.0036747, 0.0045117, -0.0006550, 0.0006550
3: 0.0013604, 0.0027128, 0.0013604, 0.0027128, -0.0009576, 0.0009576
4: -0.0044946, -0.0026214, -0.0044946, -0.0026214, -0.0011189, 0.0011189
5: -0.0002226, 0.0008932, -0.0002226, 0.0008932, -0.0009451, 0.0009451
6: -0.0050262, -0.0017978, -0.0050262, -0.0017978, -0.0019008, 0.0019008
7: -0.0220004, -0.0112250, -0.0220004, -0.0112250, -0.0065105, 0.0065105
8: 0.9753920, 0.9852611, 0.9753920, 0.9852611, -0.0064002, 0.0064002
9: -0.0003713, 0.0067090, -0.0003713, 0.0067090, -0.0043372, 0.0043372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048770, upper bound: 0.0045755
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048708, upper bound: 0.0045621
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0153367, 0.0179262, 0.0153948, 0.0180371, -0.0021752, 0.0019947
1: -0.0017278, 0.0001183, -0.0016837, 0.0002002, -0.0016199, 0.0014891
2: 0.0036747, 0.0045117, 0.0036390, 0.0044925, -0.0006349, 0.0006926
3: 0.0013604, 0.0027128, 0.0013416, 0.0027132, -0.0009603, 0.0009835
4: -0.0044946, -0.0026214, -0.0045349, -0.0026468, -0.0010946, 0.0011596
5: -0.0002226, 0.0008932, -0.0002714, 0.0008675, -0.0009148, 0.0009946
6: -0.0050262, -0.0017978, -0.0050330, -0.0017468, -0.0019517, 0.0019092
7: -0.0220004, -0.0112250, -0.0222496, -0.0113772, -0.0063643, 0.0067640
8: 0.9753920, 0.9852611, 0.9750794, 0.9851004, -0.0062334, 0.0067241
9: -0.0003713, 0.0067090, -0.0002669, 0.0068834, -0.0045178, 0.0042360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048770, upper bound: 0.0045755
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048708, upper bound: 0.0045621
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0153948, 0.0180371, 0.0153367, 0.0179262, -0.0019947, 0.0021752
1: -0.0016837, 0.0002002, -0.0017278, 0.0001183, -0.0014891, 0.0016199
2: 0.0036390, 0.0044925, 0.0036747, 0.0045117, -0.0006926, 0.0006349
3: 0.0013416, 0.0027132, 0.0013604, 0.0027128, -0.0009835, 0.0009603
4: -0.0045349, -0.0026468, -0.0044946, -0.0026214, -0.0011596, 0.0010946
5: -0.0002714, 0.0008675, -0.0002226, 0.0008932, -0.0009946, 0.0009148
6: -0.0050330, -0.0017468, -0.0050262, -0.0017978, -0.0019092, 0.0019517
7: -0.0222496, -0.0113772, -0.0220004, -0.0112250, -0.0067640, 0.0063643
8: 0.9750794, 0.9851004, 0.9753920, 0.9852611, -0.0067241, 0.0062334
9: -0.0002669, 0.0068834, -0.0003713, 0.0067090, -0.0042360, 0.0045178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047304, upper bound: 0.0045977
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047022, upper bound: 0.0045722
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0153948, 0.0180371, 0.0153948, 0.0180371, -0.0020465, 0.0020465
1: -0.0016837, 0.0002002, -0.0016837, 0.0002002, -0.0015290, 0.0015290
2: 0.0036390, 0.0044925, 0.0036390, 0.0044925, -0.0006512, 0.0006512
3: 0.0013416, 0.0027132, 0.0013416, 0.0027132, -0.0009810, 0.0009810
4: -0.0045349, -0.0026468, -0.0045349, -0.0026468, -0.0011149, 0.0011149
5: -0.0002714, 0.0008675, -0.0002714, 0.0008675, -0.0009388, 0.0009388
6: -0.0050330, -0.0017468, -0.0050330, -0.0017468, -0.0019749, 0.0019749
7: -0.0222496, -0.0113772, -0.0222496, -0.0113772, -0.0064859, 0.0064859
8: 0.9750794, 0.9851004, 0.9750794, 0.9851004, -0.0063688, 0.0063688
9: -0.0002669, 0.0068834, -0.0002669, 0.0068834, -0.0043200, 0.0043200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047304, upper bound: 0.0045977
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047022, upper bound: 0.0045722
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0153397, 0.0179261, 0.0150471, 0.0179342, -0.0021418, 0.0023045
1: -0.0017254, 0.0001182, -0.0019105, 0.0001341, -0.0015695, 0.0016860
2: 0.0036747, 0.0045108, 0.0036732, 0.0046070, -0.0007382, 0.0006854
3: 0.0013607, 0.0027112, 0.0012930, 0.0026768, -0.0009531, 0.0010308
4: -0.0044946, -0.0026223, -0.0044613, -0.0024215, -0.0013811, 0.0012816
5: -0.0002225, 0.0008916, -0.0002355, 0.0009970, -0.0010281, 0.0009575
6: -0.0050243, -0.0017993, -0.0052567, -0.0019015, -0.0020606, 0.0022043
7: -0.0220003, -0.0112306, -0.0218165, -0.0100595, -0.0080103, 0.0074420
8: 0.9753920, 0.9852543, 0.9755049, 0.9864109, -0.0077187, 0.0071748
9: -0.0003674, 0.0067089, -0.0011490, 0.0065955, -0.0049443, 0.0053173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048978, upper bound: 0.0049699
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048978, upper bound: 0.0049699
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0153526, 0.0179257, 0.0149659, 0.0179636, -0.0021228, 0.0025552
1: -0.0017153, 0.0001177, -0.0019734, 0.0001561, -0.0015554, 0.0018879
2: 0.0036748, 0.0045068, 0.0036634, 0.0046324, -0.0008158, 0.0006796
3: 0.0013625, 0.0027054, 0.0012950, 0.0027067, -0.0010819, 0.0010133
4: -0.0044944, -0.0026263, -0.0044820, -0.0023963, -0.0014252, 0.0012895
5: -0.0002221, 0.0008852, -0.0002485, 0.0010360, -0.0011551, 0.0009486
6: -0.0050113, -0.0018046, -0.0052371, -0.0018648, -0.0021471, 0.0021687
7: -0.0219999, -0.0112535, -0.0219363, -0.0099024, -0.0083024, 0.0074799
8: 0.9753920, 0.9852266, 0.9753947, 0.9866004, -0.0081407, 0.0071717
9: -0.0003511, 0.0067088, -0.0012596, 0.0066729, -0.0049623, 0.0055358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048978, upper bound: 0.0049698
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048978, upper bound: 0.0049698
time: 0.76 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.91 seconds
IS_A1_B1_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
IS_A1_B1_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
IS_A1_B1_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
IS_A1_B1_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0039885, upper bound: 0.0040050
IS_A1_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0038717, upper bound: 0.0038995
IS_A1_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0038717, upper bound: 0.0038995
IS_A1_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0038125, upper bound: 0.0036366
IS_A1_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0038125, upper bound: 0.0036366
IS_A1_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0048770, upper bound: 0.0045755
IS_A1_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0048708, upper bound: 0.0045621
IS_A1_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0048770, upper bound: 0.0045755
IS_A1_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0048708, upper bound: 0.0045621
IS_A1_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0047304, upper bound: 0.0045977
IS_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0047022, upper bound: 0.0045722
IS_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0047304, upper bound: 0.0045977
IS_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0047022, upper bound: 0.0045722
IS_A1_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0048978, upper bound: 0.0049699
IS_A1_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0048978, upper bound: 0.0049699
IS_A1_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0048978, upper bound: 0.0049698
IS_A1_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.91
Output dim: 8, lower bound: -0.0048978, upper bound: 0.0049698
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0040902, upper bound: 0.0043516
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0040580, upper bound: 0.0043479
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0042253, upper bound: 0.0044071
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0041916, upper bound: 0.0041359
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0050340, upper bound: 0.0049623
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0042263, upper bound: 0.0036479
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0040758, upper bound: 0.0042134
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0040392, upper bound: 0.0042060
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0040758, upper bound: 0.0042134
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0040392, upper bound: 0.0042060
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0049768, upper bound: 0.0049031
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0049698, upper bound: 0.0048978
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0044258, upper bound: 0.0041321
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0050347, upper bound: 0.0049674
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0043162, upper bound: 0.0041120
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0043174, upper bound: 0.0032298
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0046087, upper bound: 0.0046186
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0046047, upper bound: 0.0044305
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0045861, upper bound: 0.0045980
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0045846, upper bound: 0.0044073
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0042334, upper bound: 0.0043119
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 8, lower bound: -0.0042007, upper bound: 0.0040639

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.99 + 598.45 = 601.44 seconds
