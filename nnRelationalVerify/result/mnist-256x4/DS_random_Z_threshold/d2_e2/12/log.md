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
execution time: IAR + RelationalAnalysis = 0.75 + 2.57 = 3.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0731673, upper bound: 0.0731670

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697326, upper bound: 0.0697329
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0697326, upper bound: 0.0697329
time: 1.29 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 6, lower bound: -0.0697326, upper bound: 0.0697329
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 6, lower bound: -0.0697326, upper bound: 0.0697329

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0694101, upper bound: 0.0694100
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0694101, upper bound: 0.0694096
time: 1.54 seconds

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0599395, upper bound: 0.0599395
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0599395, upper bound: 0.0599395
time: 0.81 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.29 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 6, lower bound: -0.0694101, upper bound: 0.0694100
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 6, lower bound: -0.0694101, upper bound: 0.0694096
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.29
Output dim: 6, lower bound: -0.0599395, upper bound: 0.0599395
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.29
Output dim: 6, lower bound: -0.0599395, upper bound: 0.0599395

## BFS DS instance: DS_DSZ1_DSZ1

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0632786, upper bound: 0.0632784
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0632786, upper bound: 0.0632784
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0686081, upper bound: 0.0686105
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0686101, upper bound: 0.0686085
time: 1.52 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.45
Output dim: 6, lower bound: -0.0632786, upper bound: 0.0632784
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 5.45
Output dim: 6, lower bound: -0.0632786, upper bound: 0.0632784
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.45
Output dim: 6, lower bound: -0.0686081, upper bound: 0.0686105
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.45
Output dim: 6, lower bound: -0.0686101, upper bound: 0.0686085

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0686080, upper bound: 0.0686094
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0686072, upper bound: 0.0686099
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0684945, upper bound: 0.0684984
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0684999, upper bound: 0.0684939
time: 1.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.90 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 6, lower bound: -0.0686080, upper bound: 0.0686094
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 6, lower bound: -0.0686072, upper bound: 0.0686099
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 6, lower bound: -0.0684945, upper bound: 0.0684984
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 6, lower bound: -0.0684999, upper bound: 0.0684939

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0648683, upper bound: 0.0648687
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0648683, upper bound: 0.0648687
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677108, upper bound: 0.0677166
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0677139, upper bound: 0.0677146
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0684949, upper bound: 0.0684981
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0684949, upper bound: 0.0684984
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0648307, upper bound: 0.0648274
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0648307, upper bound: 0.0648274
time: 1.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.99 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 6, lower bound: -0.0648683, upper bound: 0.0648687
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 6, lower bound: -0.0648683, upper bound: 0.0648687
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 6, lower bound: -0.0677108, upper bound: 0.0677166
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 6, lower bound: -0.0677139, upper bound: 0.0677146
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 6, lower bound: -0.0684949, upper bound: 0.0684981
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 6, lower bound: -0.0684949, upper bound: 0.0684984
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 6, lower bound: -0.0648307, upper bound: 0.0648274
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 6, lower bound: -0.0648307, upper bound: 0.0648274

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0667755, upper bound: 0.0667781
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0667755, upper bound: 0.0667781
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682053, upper bound: 0.0682108
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682073, upper bound: 0.0682090
time: 1.62 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.92 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0667755, upper bound: 0.0667781
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0667755, upper bound: 0.0667781
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0682053, upper bound: 0.0682108
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.92
Output dim: 6, lower bound: -0.0682073, upper bound: 0.0682090

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682055, upper bound: 0.0682103
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682055, upper bound: 0.0682103
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682034, upper bound: 0.0682042
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682021, upper bound: 0.0682048
time: 1.70 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.15 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.15
Output dim: 6, lower bound: -0.0682055, upper bound: 0.0682103
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.15
Output dim: 6, lower bound: -0.0682055, upper bound: 0.0682103
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.15
Output dim: 6, lower bound: -0.0682034, upper bound: 0.0682042
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.15
Output dim: 6, lower bound: -0.0682021, upper bound: 0.0682048

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0644928, upper bound: 0.0644952
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0644928, upper bound: 0.0644952
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681188, upper bound: 0.0681266
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681212, upper bound: 0.0681222
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682026, upper bound: 0.0682044
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0682035, upper bound: 0.0682029
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681805, upper bound: 0.0681837
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681803, upper bound: 0.0681837
time: 1.59 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.01 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.01
Output dim: 6, lower bound: -0.0644928, upper bound: 0.0644952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.01
Output dim: 6, lower bound: -0.0644928, upper bound: 0.0644952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.01
Output dim: 6, lower bound: -0.0681188, upper bound: 0.0681266
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.01
Output dim: 6, lower bound: -0.0681212, upper bound: 0.0681222
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.01
Output dim: 6, lower bound: -0.0682026, upper bound: 0.0682044
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.01
Output dim: 6, lower bound: -0.0682035, upper bound: 0.0682029
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.01
Output dim: 6, lower bound: -0.0681805, upper bound: 0.0681837
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.01
Output dim: 6, lower bound: -0.0681803, upper bound: 0.0681837

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681188, upper bound: 0.0681262
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681182, upper bound: 0.0681265
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681207, upper bound: 0.0681205
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681205, upper bound: 0.0681217
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0571238, upper bound: 0.0571236
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0571238, upper bound: 0.0571236
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681188, upper bound: 0.0681204
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681211, upper bound: 0.0681168
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681808, upper bound: 0.0681823
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681801, upper bound: 0.0681839
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0648106, upper bound: 0.0648118
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0648106, upper bound: 0.0648118
time: 1.53 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.80 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0681188, upper bound: 0.0681262
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0681182, upper bound: 0.0681265
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0681207, upper bound: 0.0681205
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0681205, upper bound: 0.0681217
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0571238, upper bound: 0.0571236
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0571238, upper bound: 0.0571236
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0681188, upper bound: 0.0681204
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0681211, upper bound: 0.0681168
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0681808, upper bound: 0.0681823
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0681801, upper bound: 0.0681839
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0648106, upper bound: 0.0648118
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 3.80
Output dim: 6, lower bound: -0.0648106, upper bound: 0.0648118

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0672519, upper bound: 0.0672594
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0672536, upper bound: 0.0672574
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0664086, upper bound: 0.0664151
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0664087, upper bound: 0.0664150
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0672531, upper bound: 0.0672542
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0672554, upper bound: 0.0672523
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 162

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0669435, upper bound: 0.0669466
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0669451, upper bound: 0.0669456
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0647055, upper bound: 0.0647076
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0647055, upper bound: 0.0647076
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681210, upper bound: 0.0681166
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681211, upper bound: 0.0681167
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0648108, upper bound: 0.0648110
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0648108, upper bound: 0.0648110
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0577626, upper bound: 0.0577627
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0577626, upper bound: 0.0577627
time: 0.89 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 2.52 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0672519, upper bound: 0.0672594
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0672536, upper bound: 0.0672574
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0664086, upper bound: 0.0664151
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0664087, upper bound: 0.0664150
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0672531, upper bound: 0.0672542
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0672554, upper bound: 0.0672523
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0669435, upper bound: 0.0669466
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0669451, upper bound: 0.0669456
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0647055, upper bound: 0.0647076
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0647055, upper bound: 0.0647076
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0681210, upper bound: 0.0681166
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0681211, upper bound: 0.0681167
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0648108, upper bound: 0.0648110
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0648108, upper bound: 0.0648110
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0577626, upper bound: 0.0577627
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 2.52
Output dim: 6, lower bound: -0.0577626, upper bound: 0.0577627

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681210, upper bound: 0.0681153
time: 24.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681205, upper bound: 0.0681166
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0643878, upper bound: 0.0643834
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0643878, upper bound: 0.0643834
time: 1.26 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 3.28 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 11, time: 3.28
Output dim: 6, lower bound: -0.0681210, upper bound: 0.0681153
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 11, time: 3.28
Output dim: 6, lower bound: -0.0681205, upper bound: 0.0681166
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 3.28
Output dim: 6, lower bound: -0.0643878, upper bound: 0.0643834
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 3.28
Output dim: 6, lower bound: -0.0643878, upper bound: 0.0643834

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681076, upper bound: 0.0681022
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0681076, upper bound: 0.0681024
time: 7.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0664113, upper bound: 0.0664100
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0664114, upper bound: 0.0664100
time: 1.70 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 3.97 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 6, lower bound: -0.0681076, upper bound: 0.0681022
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 6, lower bound: -0.0681076, upper bound: 0.0681024
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 3.97
Output dim: 6, lower bound: -0.0664113, upper bound: 0.0664100
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 3.97
Output dim: 6, lower bound: -0.0664114, upper bound: 0.0664100

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 160

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0650735, upper bound: 0.0650686
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0650733, upper bound: 0.0650685
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0663816, upper bound: 0.0663785
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0663816, upper bound: 0.0663785
time: 1.54 seconds

## Summary of splitting (split count: 12)
- Time for DS candidates: 3.94 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 13, time: 3.94
Output dim: 6, lower bound: -0.0650735, upper bound: 0.0650686
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 13, time: 3.94
Output dim: 6, lower bound: -0.0650733, upper bound: 0.0650685
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 13, time: 3.94
Output dim: 6, lower bound: -0.0663816, upper bound: 0.0663785
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 13, time: 3.94
Output dim: 6, lower bound: -0.0663816, upper bound: 0.0663785

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.32 + 172.26 = 175.58 seconds
