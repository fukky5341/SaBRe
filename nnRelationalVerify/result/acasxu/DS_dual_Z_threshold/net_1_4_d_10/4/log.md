## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 96.5219627187


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443)
1: (-41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358)
2: (-42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923)
3: (-48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496)
4: (-45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.55 + 1.83 = 2.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -96.6185813, upper bound: 96.6185813

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6152125, upper bound: 96.6152125
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6152125, upper bound: 96.6152409
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.64 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 4, lower bound: -96.6152125, upper bound: 96.6152125
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 4, lower bound: -96.6152125, upper bound: 96.6152409

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6107614, upper bound: 96.6112789
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6113032, upper bound: 96.6107588
time: 0.64 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6107588, upper bound: 96.6113032
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6112789, upper bound: 96.6107614
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.85 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.85
Output dim: 4, lower bound: -96.6107614, upper bound: 96.6112789
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.85
Output dim: 4, lower bound: -96.6113032, upper bound: 96.6107588
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.85
Output dim: 4, lower bound: -96.6107588, upper bound: 96.6113032
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.85
Output dim: 4, lower bound: -96.6112789, upper bound: 96.6107614

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984443
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984685, upper bound: 96.5988841
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984787, upper bound: 96.5984443
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984443
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984787
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5989357
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984685
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984443
time: 0.70 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984443
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 4, lower bound: -96.5984685, upper bound: 96.5988841
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 4, lower bound: -96.5984787, upper bound: 96.5984443
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984443
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984787
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5989357
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984685
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 4, lower bound: -96.5984443, upper bound: 96.5984443

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984624, upper bound: 96.5987805
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5986707, upper bound: 96.5984132
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5988790, upper bound: 96.5984132
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984776, upper bound: 96.5984132
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984776
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5988790
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5986707
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984624
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
time: 0.87 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984624, upper bound: 96.5987805
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5986707, upper bound: 96.5984132
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5988790, upper bound: 96.5984132
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984776, upper bound: 96.5984132
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984776
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5988790
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5986707
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984624
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 4, lower bound: -96.5984132, upper bound: 96.5984132

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984624, upper bound: 96.5987006
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984776, upper bound: 96.5984086
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984776
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5988790
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5987854
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5986707
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984624
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.56 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984624, upper bound: 96.5987006
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984776, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984776
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5988790
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5987854
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5986707
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984624
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.89
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984624, upper bound: 96.5984086
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5987006
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5986707, upper bound: 96.5984086
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5987854, upper bound: 96.5984086
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984776, upper bound: 96.5984086
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984776
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5988790
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5987854
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5986707
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5987006, upper bound: 96.5984374
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984624
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
time: 0.66 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984624, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5987006
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5986707, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5987854, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984776, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984776
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5988790
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5987854
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5986707
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5987006, upper bound: 96.5984374
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984624
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.01
Output dim: 4, lower bound: -96.5984086, upper bound: 96.5984086

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5980183, upper bound: 96.5979788
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5982629
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980199
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5982197, upper bound: 96.5979788
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5980251, upper bound: 96.5979788
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980251
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980324
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5983995
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980929
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5983255
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980911
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5982197
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980789
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979936
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980183
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
time: 0.68 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5980183, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5982629
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980199
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5982197, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5980251, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980251
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980324
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5983995
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980929
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5983255
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980911
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5982197
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980789
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979936
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980183
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.29
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5980136
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5978075
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977701
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5978287, upper bound: 96.5977303
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977322, upper bound: 96.5977303
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5979678, upper bound: 96.5977303
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5980753, upper bound: 96.5977303
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977346, upper bound: 96.5977303
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977767, upper bound: 96.5977303
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443
1: -41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358
2: -42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923
3: -48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496
4: -45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
time: 0.63 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5980136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5978075
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977701
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5978287, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977322, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5979678, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5980753, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977346, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977767, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.14
Output dim: 4, lower bound: -96.5977303, upper bound: 96.5977303
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980251
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980324
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5983995
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980929
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5983255
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980911
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5982197
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980789
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979936
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5980183
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 4, lower bound: -96.5979788, upper bound: 96.5979788

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.38 + 418.80 = 421.17 seconds
