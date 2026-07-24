## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00046665


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002831, 0.0002831)
1: (-0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0007183, 0.0007183)
2: (0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004456, 0.0004456)
3: (0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008321, 0.0008321)
4: (-0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007306, 0.0007306)
5: (0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002767, 0.0002767)
6: (0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010561, 0.0010561)
7: (0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007390, 0.0007390)
8: (-0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007923, 0.0007923)
9: (-0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005234, 0.0005234)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.73 + 1.36 = 3.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0005018, upper bound: 0.0005018

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004947, upper bound: 0.0004859
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004858, upper bound: 0.0004946
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.98 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.98
Output dim: 7, lower bound: -0.0004947, upper bound: 0.0004859
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.98
Output dim: 7, lower bound: -0.0004858, upper bound: 0.0004946

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002759, 0.0002775
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0007002, 0.0007041
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004344, 0.0004369
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008157, 0.0008111
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007122, 0.0007162
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002698, 0.0002713
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010353, 0.0010294
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007244, 0.0007203
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007767, 0.0007723
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005102, 0.0005131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004937, upper bound: 0.0004810
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004849
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002775, 0.0002759
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0007041, 0.0007002
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004369, 0.0004344
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008111, 0.0008157
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007162, 0.0007122
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002713, 0.0002698
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010294, 0.0010353
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007203, 0.0007244
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007723, 0.0007767
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005131, 0.0005102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004776, upper bound: 0.0004819
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004776, upper bound: 0.0004826
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 7, lower bound: -0.0004937, upper bound: 0.0004810
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004849
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 7, lower bound: -0.0004776, upper bound: 0.0004819
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 7, lower bound: -0.0004776, upper bound: 0.0004826

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002753, 0.0002777
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006986, 0.0007048
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004334, 0.0004373
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008165, 0.0008093
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007106, 0.0007169
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002692, 0.0002715
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010362, 0.0010271
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007251, 0.0007187
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007774, 0.0007706
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005090, 0.0005135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004816, upper bound: 0.0004715
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004809, upper bound: 0.0004715
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002762, 0.0002769
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0007008, 0.0007027
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004348, 0.0004359
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008140, 0.0008119
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007128, 0.0007147
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002700, 0.0002707
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010331, 0.0010304
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007229, 0.0007210
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007751, 0.0007730
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005106, 0.0005120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004714, upper bound: 0.0004652
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004713, upper bound: 0.0004642
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002766, 0.0002754
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0007020, 0.0006988
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004355, 0.0004335
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008095, 0.0008133
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007141, 0.0007108
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002705, 0.0002692
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010274, 0.0010321
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007189, 0.0007222
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007708, 0.0007743
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005115, 0.0005091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004650, upper bound: 0.0004772
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004730, upper bound: 0.0004679
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002769, 0.0002759
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0007028, 0.0007002
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004360, 0.0004344
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008111, 0.0008141
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007148, 0.0007122
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002708, 0.0002698
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010294, 0.0010332
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007203, 0.0007230
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007723, 0.0007752
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005121, 0.0005102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0004786
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004715, upper bound: 0.0004816
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.0004816, upper bound: 0.0004715
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.0004809, upper bound: 0.0004715
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.0004714, upper bound: 0.0004652
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.0004713, upper bound: 0.0004642
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.0004650, upper bound: 0.0004772
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.0004730, upper bound: 0.0004679
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0004786
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.0004715, upper bound: 0.0004816

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002746, 0.0002773
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006967, 0.0007037
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004323, 0.0004365
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008151, 0.0008071
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007087, 0.0007157
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002684, 0.0002711
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010345, 0.0010243
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007239, 0.0007168
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007761, 0.0007685
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005076, 0.0005127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004711, upper bound: 0.0004668
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004770, upper bound: 0.0004608
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002748, 0.0002777
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006975, 0.0007048
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004327, 0.0004373
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008165, 0.0008080
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007094, 0.0007169
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002687, 0.0002715
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010362, 0.0010254
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007251, 0.0007175
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007774, 0.0007693
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005082, 0.0005135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004599, upper bound: 0.0004520
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004598, upper bound: 0.0004507
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002518, 0.0002530
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006390, 0.0006420
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0003965, 0.0003983
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007437, 0.0007403
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0006500, 0.0006530
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002462, 0.0002473
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0009438, 0.0009395
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0006604, 0.0006574
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007081, 0.0007049
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0004656, 0.0004677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004677, upper bound: 0.0004267
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004619
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002522, 0.0002522
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006401, 0.0006400
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0003971, 0.0003971
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007415, 0.0007415
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0006511, 0.0006510
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002466, 0.0002466
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0009410, 0.0009411
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0006585, 0.0006585
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007060, 0.0007061
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0004664, 0.0004663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004575, upper bound: 0.0004559
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004566, upper bound: 0.0004559
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002720, 0.0002699
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006901, 0.0006848
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004281, 0.0004249
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007933, 0.0007995
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007020, 0.0006966
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002659, 0.0002638
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010068, 0.0010146
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007045, 0.0007100
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007554, 0.0007612
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005028, 0.0004990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004641, upper bound: 0.0004729
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004610, upper bound: 0.0004762
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002711, 0.0002708
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006881, 0.0006871
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004269, 0.0004263
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007960, 0.0007971
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0006999, 0.0006989
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002651, 0.0002647
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010102, 0.0010116
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007069, 0.0007079
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007579, 0.0007589
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005013, 0.0005006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004522, upper bound: 0.0004470
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004525, upper bound: 0.0004469
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002764, 0.0002762
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0007015, 0.0007008
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004352, 0.0004348
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008119, 0.0008127
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007136, 0.0007128
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002703, 0.0002700
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010304, 0.0010314
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007210, 0.0007217
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007730, 0.0007738
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005111, 0.0005106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004559, upper bound: 0.0004575
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004562, upper bound: 0.0004576
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002773, 0.0002753
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0007037, 0.0006986
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004365, 0.0004334
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0008093, 0.0008151
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0007157, 0.0007106
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002711, 0.0002692
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010271, 0.0010345
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007187, 0.0007239
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007706, 0.0007761
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005127, 0.0005090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004507, upper bound: 0.0004607
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004520, upper bound: 0.0004607
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004711, upper bound: 0.0004668
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004770, upper bound: 0.0004608
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004599, upper bound: 0.0004520
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004598, upper bound: 0.0004507
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004677, upper bound: 0.0004267
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004318, upper bound: 0.0004619
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004575, upper bound: 0.0004559
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004566, upper bound: 0.0004559
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004641, upper bound: 0.0004729
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004610, upper bound: 0.0004762
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004522, upper bound: 0.0004470
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004525, upper bound: 0.0004469
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004559, upper bound: 0.0004575
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004562, upper bound: 0.0004576
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004507, upper bound: 0.0004607
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 7, lower bound: -0.0004520, upper bound: 0.0004607

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002683, 0.0002699
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006808, 0.0006849
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004224, 0.0004249
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007934, 0.0007887
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0006925, 0.0006966
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002623, 0.0002639
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010069, 0.0010009
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007046, 0.0007004
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007554, 0.0007510
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0004960, 0.0004990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004498, upper bound: 0.0004473
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004494, upper bound: 0.0004460
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002672, 0.0002707
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006780, 0.0006869
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004206, 0.0004261
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007957, 0.0007854
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0006896, 0.0006987
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002612, 0.0002646
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010098, 0.0009968
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007066, 0.0006975
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007576, 0.0007478
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0004940, 0.0005005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004560, upper bound: 0.0004414
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004560, upper bound: 0.0004387
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002471, 0.0002500
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006271, 0.0006344
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0003891, 0.0003936
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007349, 0.0007265
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0006379, 0.0006453
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002416, 0.0002444
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0009327, 0.0009220
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0006527, 0.0006452
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0006998, 0.0006917
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0004569, 0.0004622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004543, upper bound: 0.0004172
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004531, upper bound: 0.0004172
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002698, 0.0002683
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006846, 0.0006809
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004248, 0.0004224
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007888, 0.0007931
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0006964, 0.0006926
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002638, 0.0002623
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0010011, 0.0010066
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0007005, 0.0007044
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007511, 0.0007552
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0004988, 0.0004961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004432, upper bound: 0.0004520
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004438, upper bound: 0.0004520
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0026759, -0.0020847, -0.0026759, -0.0020847, -0.0002704, 0.0002675
1: -0.0111010, -0.0096009, -0.0111010, -0.0096009, -0.0006862, 0.0006787
2: 0.0281429, 0.0290736, 0.0281429, 0.0290736, -0.0004257, 0.0004211
3: 0.0052748, 0.0070126, 0.0052748, 0.0070126, -0.0007862, 0.0007949
4: -0.0101847, -0.0086587, -0.0101847, -0.0086587, -0.0006980, 0.0006904
5: 0.0098805, 0.0104585, 0.0098805, 0.0104585, -0.0002644, 0.0002615
6: 0.0070825, 0.0092880, 0.0070825, 0.0092880, -0.0009978, 0.0010089
7: 0.9830152, 0.9845586, 0.9830152, 0.9845586, -0.0006982, 0.0007060
8: -0.0047746, -0.0031199, -0.0047746, -0.0031199, -0.0007486, 0.0007569
9: -0.0029388, -0.0018457, -0.0029388, -0.0018457, -0.0005000, 0.0004945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004617
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0004482, upper bound: 0.0004556
time: 0.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004498, upper bound: 0.0004473
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004494, upper bound: 0.0004460
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004560, upper bound: 0.0004414
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004560, upper bound: 0.0004387
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004543, upper bound: 0.0004172
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004531, upper bound: 0.0004172
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004432, upper bound: 0.0004520
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004438, upper bound: 0.0004520
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004402, upper bound: 0.0004617
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 7, lower bound: -0.0004482, upper bound: 0.0004556

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.08 + 52.47 = 55.56 seconds
