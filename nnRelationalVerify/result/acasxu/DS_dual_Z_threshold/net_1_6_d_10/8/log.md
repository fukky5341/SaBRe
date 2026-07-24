## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 17.96064755562


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208)
1: (-7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301)
2: (-4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755)
3: (-8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169)
4: (-5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.76 + 1.74 = 2.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -17.9624438, upper bound: 17.9624438

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9612123
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450
time: 0.63 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.89 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9612123
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606804, upper bound: 17.9612123
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9607570
time: 0.51 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607570, upper bound: 17.9615450
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9606804
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.88 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.88
Output dim: 3, lower bound: -17.9606804, upper bound: 17.9612123
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.88
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9607570
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.88
Output dim: 3, lower bound: -17.9607570, upper bound: 17.9615450
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.88
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9606804

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606804, upper bound: 17.9611951
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606594, upper bound: 17.9607522
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9607570
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614443, upper bound: 17.9606957
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9614443
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607570, upper bound: 17.9615450
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607522, upper bound: 17.9606594
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607522, upper bound: 17.9606804
time: 0.61 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 3, lower bound: -17.9606804, upper bound: 17.9611951
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 3, lower bound: -17.9606594, upper bound: 17.9607522
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9607570
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 3, lower bound: -17.9614443, upper bound: 17.9606957
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9614443
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 3, lower bound: -17.9607570, upper bound: 17.9615450
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 3, lower bound: -17.9607522, upper bound: 17.9606594
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 3, lower bound: -17.9607522, upper bound: 17.9606804

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606692, upper bound: 17.9611908
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606692, upper bound: 17.9611951
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606989
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606594, upper bound: 17.9607522
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613811, upper bound: 17.9607570
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9607545
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613642, upper bound: 17.9606768
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606594, upper bound: 17.9606957
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9614443
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606768, upper bound: 17.9613642
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607545, upper bound: 17.9615450
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607570, upper bound: 17.9613811
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606989, upper bound: 17.9606594
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606989, upper bound: 17.9606523
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9606692
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606768, upper bound: 17.9606804
time: 0.64 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606692, upper bound: 17.9611908
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606692, upper bound: 17.9611951
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606989
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606594, upper bound: 17.9607522
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9613811, upper bound: 17.9607570
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9607545
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9613642, upper bound: 17.9606768
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606594, upper bound: 17.9606957
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9614443
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606768, upper bound: 17.9613642
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9607545, upper bound: 17.9615450
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9607570, upper bound: 17.9613811
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606989, upper bound: 17.9606594
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606989, upper bound: 17.9606523
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9606692
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 3, lower bound: -17.9606768, upper bound: 17.9606804

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9610877
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9611908
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9606523
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9611951
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606683
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606989
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606523
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606594, upper bound: 17.9607522
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613568, upper bound: 17.9607225
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9607570
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613577, upper bound: 17.9606523
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615395, upper bound: 17.9607545
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613604, upper bound: 17.9606619
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613642, upper bound: 17.9606768
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613780, upper bound: 17.9606523
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614443, upper bound: 17.9606957
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9614443
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9613780
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606619, upper bound: 17.9613642
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9613604
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9615395
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9613577
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607225, upper bound: 17.9613797
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607225, upper bound: 17.9613568
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606594
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606523
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606989, upper bound: 17.9606523
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606683, upper bound: 17.9606523
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9606607
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606692
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611908, upper bound: 17.9606530
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606804
time: 0.52 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9610877
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9611908
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9606523
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9611951
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606683
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606989
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606523
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606594, upper bound: 17.9607522
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9613568, upper bound: 17.9607225
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9607570
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9613577, upper bound: 17.9606523
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9615395, upper bound: 17.9607545
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9613604, upper bound: 17.9606619
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9613642, upper bound: 17.9606768
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9613780, upper bound: 17.9606523
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9614443, upper bound: 17.9606957
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9614443
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9613780
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606619, upper bound: 17.9613642
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9613604
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9615395
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9613577
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9607225, upper bound: 17.9613797
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9607225, upper bound: 17.9613568
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606594
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606523
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606989, upper bound: 17.9606523
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606683, upper bound: 17.9606523
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9606607
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606692
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9611908, upper bound: 17.9606530
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9606804

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9609542
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608415
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610394
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605398, upper bound: 17.9608311
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610416
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9607458
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605580
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605870
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9606334
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605462
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608456, upper bound: 17.9606124
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605683
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9606434
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609914, upper bound: 17.9605676
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606330, upper bound: 17.9605383
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611886, upper bound: 17.9605383
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611932, upper bound: 17.9606405
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613860, upper bound: 17.9605494
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608617, upper bound: 17.9605504
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611931, upper bound: 17.9605383
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608857, upper bound: 17.9605652
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608786, upper bound: 17.9605383
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608786, upper bound: 17.9605383
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610238, upper bound: 17.9605824
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612776, upper bound: 17.9605383
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612776
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610238
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612094
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608786
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611986
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608857
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611931
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605504, upper bound: 17.9608617
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605494, upper bound: 17.9613860
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606405, upper bound: 17.9611932
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611886
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9606330
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605676, upper bound: 17.9612199
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605676, upper bound: 17.9609914
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605683, upper bound: 17.9611879
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608456
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605462
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605870, upper bound: 17.9605383
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605580, upper bound: 17.9605383
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605462, upper bound: 17.9605490
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607458, upper bound: 17.9605383
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605583
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608311, upper bound: 17.9605398
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608311, upper bound: 17.9605383
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608415, upper bound: 17.9605670
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608415, upper bound: 17.9605383
time: 0.57 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9609542
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608415
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610394
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605398, upper bound: 17.9608311
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610416
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9607458
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605580
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605870
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9606334
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605462
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608456, upper bound: 17.9606124
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605683
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9606434
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9609914, upper bound: 17.9605676
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9606330, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9611886, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9611932, upper bound: 17.9606405
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9613860, upper bound: 17.9605494
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608617, upper bound: 17.9605504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9611931, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608857, upper bound: 17.9605652
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608786, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608786, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9610238, upper bound: 17.9605824
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9612776, upper bound: 17.9605383
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612776
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610238
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612094
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608786
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611986
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608857
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611931
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605504, upper bound: 17.9608617
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605494, upper bound: 17.9613860
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9606405, upper bound: 17.9611932
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611886
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9606330
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605676, upper bound: 17.9612199
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605676, upper bound: 17.9609914
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605683, upper bound: 17.9611879
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608456
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605462
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605870, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605580, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605462, upper bound: 17.9605490
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9607458, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605583
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608311, upper bound: 17.9605398
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608311, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608415, upper bound: 17.9605670
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.00
Output dim: 3, lower bound: -17.9608415, upper bound: 17.9605383

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608829
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9609542
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608415
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605616, upper bound: 17.9608403
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610380
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610394
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605398, upper bound: 17.9608311
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9606812
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610415
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610416
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605490, upper bound: 17.9607458
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608456, upper bound: 17.9606124
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607818, upper bound: 17.9605662
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609914, upper bound: 17.9605676
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611886, upper bound: 17.9605383
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606330, upper bound: 17.9605383
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607818, upper bound: 17.9606101
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611932, upper bound: 17.9606405
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611923, upper bound: 17.9605494
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612699, upper bound: 17.9605383
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608617, upper bound: 17.9605504
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611931, upper bound: 17.9605383
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611907, upper bound: 17.9605383
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605652
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608786, upper bound: 17.9605383
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612094, upper bound: 17.9605383
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612016, upper bound: 17.9605383
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610238, upper bound: 17.9605734
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610237, upper bound: 17.9605824
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612776, upper bound: 17.9605383
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612417, upper bound: 17.9605383
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612417
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612776
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610237
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605734, upper bound: 17.9610238
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612016
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612094
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608786
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611952
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611986
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605652, upper bound: 17.9608857
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611907
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611931
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605504, upper bound: 17.9608617
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612699
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605494, upper bound: 17.9613860
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611932
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606101, upper bound: 17.9611923
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611333
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611886
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605676, upper bound: 17.9612025
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612199
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605676, upper bound: 17.9609914
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9607818
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605683, upper bound: 17.9611845
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605551, upper bound: 17.9611879
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606124, upper bound: 17.9608456
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610415, upper bound: 17.9605383
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608311, upper bound: 17.9605398
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606812, upper bound: 17.9605383
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608403, upper bound: 17.9605616
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608415, upper bound: 17.9605670
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
time: 0.54 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608829
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9609542
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608415
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605616, upper bound: 17.9608403
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610380
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610394
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605398, upper bound: 17.9608311
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9606812
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610415
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610416
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605490, upper bound: 17.9607458
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9608456, upper bound: 17.9606124
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9607818, upper bound: 17.9605662
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9609914, upper bound: 17.9605676
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9611886, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9606330, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9607818, upper bound: 17.9606101
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9611932, upper bound: 17.9606405
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9611923, upper bound: 17.9605494
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9612699, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9608617, upper bound: 17.9605504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9611931, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9611907, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605652
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9608786, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9612094, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9612016, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9610238, upper bound: 17.9605734
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9610237, upper bound: 17.9605824
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9612776, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9612417, upper bound: 17.9605383
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612417
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612776
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9610237
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605734, upper bound: 17.9610238
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612016
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612094
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9608786
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611986
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605652, upper bound: 17.9608857
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611907
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611931
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605504, upper bound: 17.9608617
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612699
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605494, upper bound: 17.9613860
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611932
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9606101, upper bound: 17.9611923
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611333
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9611886
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605676, upper bound: 17.9612025
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9612199
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605676, upper bound: 17.9609914
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9607818
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605683, upper bound: 17.9611845
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605551, upper bound: 17.9611879
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9606124, upper bound: 17.9608456
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9610415, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9608311, upper bound: 17.9605398
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9606812, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9608403, upper bound: 17.9605616
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9608415, upper bound: 17.9605670
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.07
Output dim: 3, lower bound: -17.9605383, upper bound: 17.9605383

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608654
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608659
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609290
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609248
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608233
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608244
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605416, upper bound: 17.9608224
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605425, upper bound: 17.9608232
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610200
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610194
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610214
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610201
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9607847
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605231, upper bound: 17.9608089
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9606614
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9606636
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610237
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610227
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610237
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610227
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605266, upper bound: 17.9606933
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605295, upper bound: 17.9607088
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608017, upper bound: 17.9605928
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607567, upper bound: 17.9605920
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611326, upper bound: 17.9605466
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611879, upper bound: 17.9605481
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611595, upper bound: 17.9605482
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611858, upper bound: 17.9605496
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606416, upper bound: 17.9605218
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611538, upper bound: 17.9605218
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611688, upper bound: 17.9605923
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611707, upper bound: 17.9605925
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611712, upper bound: 17.9605988
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611714, upper bound: 17.9605987
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612797, upper bound: 17.9605302
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613451, upper bound: 17.9605317
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612299, upper bound: 17.9605218
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612533, upper bound: 17.9605218
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608184, upper bound: 17.9605295
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607601, upper bound: 17.9605267
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611337, upper bound: 17.9605218
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611745, upper bound: 17.9605218
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611541, upper bound: 17.9605218
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611743, upper bound: 17.9605218
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608348, upper bound: 17.9605218
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607847, upper bound: 17.9605218
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611705, upper bound: 17.9605218
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611891, upper bound: 17.9605218
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611674, upper bound: 17.9605218
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611838, upper bound: 17.9605218
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610039, upper bound: 17.9605495
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610062, upper bound: 17.9605495
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610041, upper bound: 17.9605485
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610052, upper bound: 17.9605457
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612133, upper bound: 17.9605218
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612430, upper bound: 17.9605218
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612043, upper bound: 17.9605218
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612253, upper bound: 17.9605218
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612253
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612043
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612430
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612133
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610052
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610041
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605494, upper bound: 17.9610062
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605495, upper bound: 17.9610039
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611838
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611674
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611891
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611705
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9607847
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608348
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611791
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611575
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611795
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611336
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605273, upper bound: 17.9607733
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605301, upper bound: 17.9608444
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611743
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611541
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611745
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611337
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605267, upper bound: 17.9607601
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605295, upper bound: 17.9608184
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612533
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612299
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605317, upper bound: 17.9613451
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612797
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611714
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605988, upper bound: 17.9611712
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605925, upper bound: 17.9611707
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605923, upper bound: 17.9611688
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610961
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9605218
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611538
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9606416
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611858
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605482, upper bound: 17.9611595
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611879
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605466, upper bound: 17.9611326
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609721
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609648
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605481, upper bound: 17.9607645
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605466, upper bound: 17.9607506
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605507, upper bound: 17.9611660
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605504, upper bound: 17.9611422
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605372, upper bound: 17.9611660
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605360, upper bound: 17.9610579
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605920, upper bound: 17.9607567
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605928, upper bound: 17.9608017
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610227, upper bound: 17.9605218
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610237, upper bound: 17.9605218
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608089, upper bound: 17.9605231
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607847, upper bound: 17.9605218
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610201, upper bound: 17.9605218
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610214, upper bound: 17.9605218
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9605425
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9605416
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608244, upper bound: 17.9605418
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608233, upper bound: 17.9605399
time: 0.93 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608654
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608659
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609290
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609248
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608244
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605416, upper bound: 17.9608224
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605425, upper bound: 17.9608232
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610200
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610194
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610214
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610201
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9607847
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605231, upper bound: 17.9608089
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9606614
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9606636
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610237
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610227
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610237
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610227
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605266, upper bound: 17.9606933
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605295, upper bound: 17.9607088
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9608017, upper bound: 17.9605928
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9607567, upper bound: 17.9605920
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611326, upper bound: 17.9605466
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611879, upper bound: 17.9605481
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611595, upper bound: 17.9605482
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611858, upper bound: 17.9605496
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9606416, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611538, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611688, upper bound: 17.9605923
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611707, upper bound: 17.9605925
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611712, upper bound: 17.9605988
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611714, upper bound: 17.9605987
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9612797, upper bound: 17.9605302
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9613451, upper bound: 17.9605317
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9612299, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9612533, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9608184, upper bound: 17.9605295
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9607601, upper bound: 17.9605267
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611337, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611745, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611541, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611743, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9608348, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9607847, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611705, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611891, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611674, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9611838, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9610039, upper bound: 17.9605495
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9610062, upper bound: 17.9605495
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9610041, upper bound: 17.9605485
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9610052, upper bound: 17.9605457
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9612133, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9612430, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9612043, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9612253, upper bound: 17.9605218
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612043
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612430
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612133
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610052
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610041
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605494, upper bound: 17.9610062
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605495, upper bound: 17.9610039
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611838
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611674
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611891
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611705
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9607847
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608348
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611791
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611795
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611336
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605273, upper bound: 17.9607733
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605301, upper bound: 17.9608444
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611743
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611541
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611337
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605267, upper bound: 17.9607601
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605295, upper bound: 17.9608184
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612533
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605317, upper bound: 17.9613451
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612797
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611714
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605988, upper bound: 17.9611712
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605925, upper bound: 17.9611707
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605923, upper bound: 17.9611688
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610961
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9605218
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9606416
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611858
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605482, upper bound: 17.9611595
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611879
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605466, upper bound: 17.9611326
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609721
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609648
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605481, upper bound: 17.9607645
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605466, upper bound: 17.9607506
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605507, upper bound: 17.9611660
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605504, upper bound: 17.9611422
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605372, upper bound: 17.9611660
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605360, upper bound: 17.9610579
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605920, upper bound: 17.9607567
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605928, upper bound: 17.9608017
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9610227, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9610237, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9608089, upper bound: 17.9605231
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9607847, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9610201, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9610214, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9605425
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9605416
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9608244, upper bound: 17.9605418
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.65
Output dim: 3, lower bound: -17.9608233, upper bound: 17.9605399

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606053
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606281
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606058
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606287
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606722
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606902
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606668
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606861
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605881
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605745
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605893
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605856
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603139, upper bound: 17.9605868
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603151, upper bound: 17.9605876
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9604513
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605530
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607674
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605535
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607668
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606266
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607689
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605950
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607676
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603004, upper bound: 17.9605191
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605525
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603019, upper bound: 17.9605274
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605760
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9604283
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9604304
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607683
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607674
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607691
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607678
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603048, upper bound: 17.9604629
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603083, upper bound: 17.9603001
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603076, upper bound: 17.9604786
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605693, upper bound: 17.9603650
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603125
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603641
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603100
time: 0.84 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606053
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606281
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606058
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606287
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606722
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606902
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606668
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606861
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605881
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605745
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605893
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603139, upper bound: 17.9605868
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603151, upper bound: 17.9605876
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9604513
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605530
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607674
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605535
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607668
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9606266
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607689
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605950
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607676
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603004, upper bound: 17.9605191
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605525
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603019, upper bound: 17.9605274
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605760
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9604283
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9604304
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607683
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607674
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607691
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9607678
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603048, upper bound: 17.9604629
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603083, upper bound: 17.9603001
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603076, upper bound: 17.9604786
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9605693, upper bound: 17.9603650
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603125
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603641
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.53
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9603100
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611326, upper bound: 17.9605466
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611879, upper bound: 17.9605481
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611595, upper bound: 17.9605482
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611858, upper bound: 17.9605496
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611538, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611688, upper bound: 17.9605923
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611707, upper bound: 17.9605925
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611712, upper bound: 17.9605988
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611714, upper bound: 17.9605987
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9612797, upper bound: 17.9605302
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9613451, upper bound: 17.9605317
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9612299, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9612533, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9608184, upper bound: 17.9605295
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9607601, upper bound: 17.9605267
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611337, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611745, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611541, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611743, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9608348, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9607847, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611705, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611891, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611674, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9611838, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9610039, upper bound: 17.9605495
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9610062, upper bound: 17.9605495
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9610041, upper bound: 17.9605485
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9610052, upper bound: 17.9605457
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9612133, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9612430, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9612043, upper bound: 17.9605218
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9612253, upper bound: 17.9605218
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612043
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612430
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612133
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610052
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610041
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605494, upper bound: 17.9610062
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605495, upper bound: 17.9610039
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611838
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611674
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611891
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611705
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9607847
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608348
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611791
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611795
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611336
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605273, upper bound: 17.9607733
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605301, upper bound: 17.9608444
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611743
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611541
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611745
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611337
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605267, upper bound: 17.9607601
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605295, upper bound: 17.9608184
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612533
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612299
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605317, upper bound: 17.9613451
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9612797
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611714
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605988, upper bound: 17.9611712
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605925, upper bound: 17.9611707
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605923, upper bound: 17.9611688
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610961
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611538
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611858
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605482, upper bound: 17.9611595
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9611879
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605466, upper bound: 17.9611326
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609721
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609648
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605481, upper bound: 17.9607645
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605466, upper bound: 17.9607506
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605507, upper bound: 17.9611660
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605504, upper bound: 17.9611422
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605372, upper bound: 17.9611660
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605360, upper bound: 17.9610579
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605920, upper bound: 17.9607567
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9605928, upper bound: 17.9608017
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9610227, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9610237, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9608089, upper bound: 17.9605231
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9607847, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9610201, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9610214, upper bound: 17.9605218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9608244, upper bound: 17.9605418
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.53
Output dim: 3, lower bound: -17.9608233, upper bound: 17.9605399

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.51 + 417.80 = 420.31 seconds
