## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.068045589


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0045828, 0.0243426, 0.0045828, 0.0243426, -0.0197598, 0.0197598)
1: (-0.0056720, 0.0211549, -0.0056720, 0.0211549, -0.0268269, 0.0268269)
2: (0.0024190, 0.0394507, 0.0024190, 0.0394507, -0.0370317, 0.0370317)
3: (-0.0105444, -0.0010774, -0.0105444, -0.0010774, -0.0094671, 0.0094671)
4: (-0.0098809, 0.0091565, -0.0098809, 0.0091565, -0.0190374, 0.0190374)
5: (-0.0112748, 0.0139717, -0.0112748, 0.0139717, -0.0252465, 0.0252465)
6: (0.9184468, 0.9928610, 0.9184468, 0.9928610, -0.0744143, 0.0744143)
7: (-0.0241657, 0.0032827, -0.0241657, 0.0032827, -0.0274484, 0.0274484)
8: (-0.0217363, 0.0139040, -0.0217363, 0.0139040, -0.0356403, 0.0356403)
9: (-0.0066065, 0.0159613, -0.0066065, 0.0159613, -0.0225677, 0.0225677)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.64 + 2.68 = 4.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0731673, upper bound: 0.0731670

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0707569, upper bound: 0.0707569
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0707569, upper bound: 0.0707569
time: 1.57 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.24
Output dim: 6, lower bound: -0.0707569, upper bound: 0.0707569
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.24
Output dim: 6, lower bound: -0.0707569, upper bound: 0.0707569

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0045828, 0.0243426, 0.0045828, 0.0243426, -0.0197598, 0.0197598
1: -0.0056720, 0.0211549, -0.0056720, 0.0211549, -0.0268269, 0.0268269
2: 0.0024190, 0.0394507, 0.0024190, 0.0394507, -0.0370317, 0.0370317
3: -0.0105444, -0.0010774, -0.0105444, -0.0010774, -0.0094671, 0.0094671
4: -0.0098809, 0.0091565, -0.0098809, 0.0091565, -0.0190374, 0.0190374
5: -0.0112748, 0.0139717, -0.0112748, 0.0139717, -0.0252465, 0.0252465
6: 0.9184468, 0.9928610, 0.9184468, 0.9928610, -0.0744143, 0.0744143
7: -0.0241657, 0.0032827, -0.0241657, 0.0032827, -0.0274484, 0.0274484
8: -0.0217363, 0.0139040, -0.0217363, 0.0139040, -0.0356403, 0.0356403
9: -0.0066065, 0.0159613, -0.0066065, 0.0159613, -0.0225677, 0.0225677

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0680275, upper bound: 0.0680288
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0680290, upper bound: 0.0680275
time: 1.33 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0045828, 0.0243426, 0.0045828, 0.0243426, -0.0197598, 0.0197598
1: -0.0056720, 0.0211549, -0.0056720, 0.0211549, -0.0268269, 0.0268269
2: 0.0024190, 0.0394507, 0.0024190, 0.0394507, -0.0370317, 0.0370317
3: -0.0105444, -0.0010774, -0.0105444, -0.0010774, -0.0094671, 0.0094671
4: -0.0098809, 0.0091565, -0.0098809, 0.0091565, -0.0190374, 0.0190374
5: -0.0112748, 0.0139717, -0.0112748, 0.0139717, -0.0252465, 0.0252465
6: 0.9184468, 0.9928610, 0.9184468, 0.9928610, -0.0744143, 0.0744143
7: -0.0241657, 0.0032827, -0.0241657, 0.0032827, -0.0274484, 0.0274484
8: -0.0217363, 0.0139040, -0.0217363, 0.0139040, -0.0356403, 0.0356403
9: -0.0066065, 0.0159613, -0.0066065, 0.0159613, -0.0225677, 0.0225677

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0680275, upper bound: 0.0680288
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0680290, upper bound: 0.0680275
time: 1.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.38 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.38
Output dim: 6, lower bound: -0.0680275, upper bound: 0.0680288
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.38
Output dim: 6, lower bound: -0.0680290, upper bound: 0.0680275
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.38
Output dim: 6, lower bound: -0.0680275, upper bound: 0.0680288
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.38
Output dim: 6, lower bound: -0.0680290, upper bound: 0.0680275

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.32 + 12.02 = 16.34 seconds
