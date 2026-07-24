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
execution time: IAR + RelationalAnalysis = 1.11 + 1.79 = 2.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -17.9624438, upper bound: 17.9624438

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9622304, upper bound: 17.9620641
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620641, upper bound: 17.9622304
time: 0.55 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.21 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 3, lower bound: -17.9622304, upper bound: 17.9620641
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 3, lower bound: -17.9620641, upper bound: 17.9622304

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620724, upper bound: 17.9620241
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9622006, upper bound: 17.9620342
time: 0.80 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620641, upper bound: 17.9622292
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620561, upper bound: 17.9620758
time: 0.87 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.41 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -17.9620724, upper bound: 17.9620241
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -17.9622006, upper bound: 17.9620342
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -17.9620641, upper bound: 17.9622292
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 3, lower bound: -17.9620561, upper bound: 17.9620758

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612515, upper bound: 17.9620241
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612515, upper bound: 17.9613091
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620939, upper bound: 17.9620298
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620939, upper bound: 17.9619159
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612524, upper bound: 17.9621036
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612524, upper bound: 17.9612857
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612857, upper bound: 17.9620567
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612857, upper bound: 17.9612524
time: 0.50 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 3, lower bound: -17.9612515, upper bound: 17.9620241
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 3, lower bound: -17.9612515, upper bound: 17.9613091
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 3, lower bound: -17.9620939, upper bound: 17.9620298
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 3, lower bound: -17.9620939, upper bound: 17.9619159
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 3, lower bound: -17.9612524, upper bound: 17.9621036
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 3, lower bound: -17.9612524, upper bound: 17.9612857
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 3, lower bound: -17.9612857, upper bound: 17.9620567
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 3, lower bound: -17.9612857, upper bound: 17.9612524

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606692, upper bound: 17.9611908
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9614443
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9619464, upper bound: 17.9613033
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612373, upper bound: 17.9612779
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620672, upper bound: 17.9620168
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620391, upper bound: 17.9619986
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9621941, upper bound: 17.9619159
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9621339, upper bound: 17.9617334
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606769, upper bound: 17.9606327
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606659, upper bound: 17.9613829
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613360, upper bound: 17.9612857
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612524, upper bound: 17.9612336
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608172, upper bound: 17.9616532
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608172, upper bound: 17.9616532
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615332, upper bound: 17.9610984
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618494, upper bound: 17.9610640
time: 0.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9606692, upper bound: 17.9611908
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9606957, upper bound: 17.9614443
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9619464, upper bound: 17.9613033
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9612373, upper bound: 17.9612779
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9620672, upper bound: 17.9620168
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9620391, upper bound: 17.9619986
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9621941, upper bound: 17.9619159
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9621339, upper bound: 17.9617334
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9606769, upper bound: 17.9606327
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9606659, upper bound: 17.9613829
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9613360, upper bound: 17.9612857
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9612524, upper bound: 17.9612336
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9608172, upper bound: 17.9616532
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9608172, upper bound: 17.9616532
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9615332, upper bound: 17.9610984
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 3, lower bound: -17.9618494, upper bound: 17.9610640

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9610877
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9611908
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9614443
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9613780
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612692
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612258, upper bound: 17.9612360
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610638, upper bound: 17.9611322
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618462, upper bound: 17.9610692
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612161, upper bound: 17.9620168
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620672, upper bound: 17.9613172
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618190, upper bound: 17.9610132
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617241, upper bound: 17.9617686
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9619050
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620660, upper bound: 17.9612168
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9610242
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9615130
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604534, upper bound: 17.9604105
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604561, upper bound: 17.9604105
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604453, upper bound: 17.9609550
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604561, upper bound: 17.9611273
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613360, upper bound: 17.9612203
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612203, upper bound: 17.9612857
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610467, upper bound: 17.9610303
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610475, upper bound: 17.9610273
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608451, upper bound: 17.9612224
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608083, upper bound: 17.9616532
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608172, upper bound: 17.9616532
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608162, upper bound: 17.9608799
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610640, upper bound: 17.9610984
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610640, upper bound: 17.9610880
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618145, upper bound: 17.9610568
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618139, upper bound: 17.9610568
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9606530, upper bound: 17.9610877
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9611908
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9614443
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9606523, upper bound: 17.9613780
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612692
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9612258, upper bound: 17.9612360
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9610638, upper bound: 17.9611322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9618462, upper bound: 17.9610692
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9612161, upper bound: 17.9620168
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9620672, upper bound: 17.9613172
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9618190, upper bound: 17.9610132
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9617241, upper bound: 17.9617686
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9619050
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9620660, upper bound: 17.9612168
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9610242
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9615130
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9604534, upper bound: 17.9604105
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9604561, upper bound: 17.9604105
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9604453, upper bound: 17.9609550
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9604561, upper bound: 17.9611273
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9613360, upper bound: 17.9612203
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9612203, upper bound: 17.9612857
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9610467, upper bound: 17.9610303
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9610475, upper bound: 17.9610273
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9608451, upper bound: 17.9612224
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9608083, upper bound: 17.9616532
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9608172, upper bound: 17.9616532
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9608162, upper bound: 17.9608799
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9610640, upper bound: 17.9610984
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9610640, upper bound: 17.9610880
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9618145, upper bound: 17.9610568
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.42
Output dim: 3, lower bound: -17.9618139, upper bound: 17.9610568

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606360, upper bound: 17.9610649
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606438, upper bound: 17.9610616
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606360, upper bound: 17.9611726
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606365, upper bound: 17.9611717
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604486, upper bound: 17.9611590
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604226, upper bound: 17.9611590
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604315, upper bound: 17.9608069
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604315, upper bound: 17.9611122
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612258, upper bound: 17.9612670
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612945
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607239, upper bound: 17.9607616
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607321, upper bound: 17.9607802
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611578, upper bound: 17.9611052
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610638, upper bound: 17.9611322
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611137, upper bound: 17.9610692
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618462, upper bound: 17.9610651
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606390, upper bound: 17.9611734
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606392, upper bound: 17.9613388
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618449, upper bound: 17.9611605
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618507, upper bound: 17.9610641
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618190, upper bound: 17.9610132
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610622, upper bound: 17.9610132
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617241, upper bound: 17.9617686
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610132, upper bound: 17.9616785
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606532, upper bound: 17.9610909
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606601, upper bound: 17.9612069
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612168
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612168
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617548, upper bound: 17.9610242
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618820, upper bound: 17.9610242
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9615130
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9610777
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604005, upper bound: 17.9609371
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604061, upper bound: 17.9609203
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601835, upper bound: 17.9608068
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601478, upper bound: 17.9608294
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606327, upper bound: 17.9606327
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606540, upper bound: 17.9606327
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612969, upper bound: 17.9612793
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612175, upper bound: 17.9612680
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616940, upper bound: 17.9610260
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618107, upper bound: 17.9610212
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607887, upper bound: 17.9604199
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607933, upper bound: 17.9604196
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608115, upper bound: 17.9611881
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608224, upper bound: 17.9609259
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607260, upper bound: 17.9614435
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607308, upper bound: 17.9615674
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608070, upper bound: 17.9615726
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608074, upper bound: 17.9607978
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608056, upper bound: 17.9608336
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608065, upper bound: 17.9607978
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605188, upper bound: 17.9605555
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605188, upper bound: 17.9605602
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613266, upper bound: 17.9608720
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612564, upper bound: 17.9608782
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608563
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608563
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611070, upper bound: 17.9610568
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618139, upper bound: 17.9610568
time: 0.95 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606360, upper bound: 17.9610649
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606438, upper bound: 17.9610616
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606360, upper bound: 17.9611726
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606365, upper bound: 17.9611717
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9604486, upper bound: 17.9611590
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9604226, upper bound: 17.9611590
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9604315, upper bound: 17.9608069
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9604315, upper bound: 17.9611122
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9612258, upper bound: 17.9612670
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612945
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9607239, upper bound: 17.9607616
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9607321, upper bound: 17.9607802
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9611578, upper bound: 17.9611052
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9610638, upper bound: 17.9611322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9611137, upper bound: 17.9610692
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9618462, upper bound: 17.9610651
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606390, upper bound: 17.9611734
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606392, upper bound: 17.9613388
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9618449, upper bound: 17.9611605
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9618507, upper bound: 17.9610641
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9618190, upper bound: 17.9610132
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9610622, upper bound: 17.9610132
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9617241, upper bound: 17.9617686
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9610132, upper bound: 17.9616785
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606532, upper bound: 17.9610909
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606601, upper bound: 17.9612069
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612168
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612168
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9617548, upper bound: 17.9610242
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9618820, upper bound: 17.9610242
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9615130
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9610777
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9604005, upper bound: 17.9609371
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9604061, upper bound: 17.9609203
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9601835, upper bound: 17.9608068
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9601478, upper bound: 17.9608294
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606327, upper bound: 17.9606327
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9606540, upper bound: 17.9606327
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9612969, upper bound: 17.9612793
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9612175, upper bound: 17.9612680
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9616940, upper bound: 17.9610260
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9618107, upper bound: 17.9610212
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9607887, upper bound: 17.9604199
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9607933, upper bound: 17.9604196
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9608115, upper bound: 17.9611881
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9608224, upper bound: 17.9609259
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9607260, upper bound: 17.9614435
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9607308, upper bound: 17.9615674
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9608070, upper bound: 17.9615726
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9608074, upper bound: 17.9607978
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9608056, upper bound: 17.9608336
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9608065, upper bound: 17.9607978
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9605188, upper bound: 17.9605555
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9605188, upper bound: 17.9605602
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9613266, upper bound: 17.9608720
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9612564, upper bound: 17.9608782
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608563
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608563
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9611070, upper bound: 17.9610568
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.83
Output dim: 3, lower bound: -17.9618139, upper bound: 17.9610568

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609290
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608233
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604058, upper bound: 17.9607109
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604059, upper bound: 17.9607381
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606333, upper bound: 17.9611690
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606333, upper bound: 17.9608706
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610201
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605231, upper bound: 17.9608089
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604285, upper bound: 17.9611520
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604493, upper bound: 17.9609601
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604674, upper bound: 17.9609986
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604226, upper bound: 17.9611590
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604287, upper bound: 17.9607975
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604287, upper bound: 17.9604320
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603168, upper bound: 17.9609423
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603168, upper bound: 17.9606420
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612258, upper bound: 17.9612179
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612227, upper bound: 17.9612670
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606300, upper bound: 17.9607329
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606300, upper bound: 17.9606300
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604120, upper bound: 17.9604108
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604120, upper bound: 17.9604038
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612195, upper bound: 17.9607398
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612205, upper bound: 17.9607802
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610951
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613888, upper bound: 17.9610631
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613354, upper bound: 17.9608973
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608642, upper bound: 17.9609107
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613772, upper bound: 17.9605631
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613772, upper bound: 17.9605929
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618071, upper bound: 17.9610531
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618349, upper bound: 17.9610531
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605189, upper bound: 17.9610200
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605189, upper bound: 17.9606905
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605189, upper bound: 17.9611760
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605189, upper bound: 17.9607706
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9611512
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9610510
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616315, upper bound: 17.9608534
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608534, upper bound: 17.9608566
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616211, upper bound: 17.9608534
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615870, upper bound: 17.9608534
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606866, upper bound: 17.9604123
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604123, upper bound: 17.9604123
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604026, upper bound: 17.9605312
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604088, upper bound: 17.9604026
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604026, upper bound: 17.9605212
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604026, upper bound: 17.9604026
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606136, upper bound: 17.9607478
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606206, upper bound: 17.9610578
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9610302
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9608618
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610539
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617984, upper bound: 17.9610539
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618350, upper bound: 17.9610135
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604126, upper bound: 17.9604126
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604126, upper bound: 17.9604126
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617954, upper bound: 17.9610135
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9615130
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9614863
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604126, upper bound: 17.9604215
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604126, upper bound: 17.9604212
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603978, upper bound: 17.9609340
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603912, upper bound: 17.9608616
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601313, upper bound: 17.9602387
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601331, upper bound: 17.9605153
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601520, upper bound: 17.9608018
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601452, upper bound: 17.9607463
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601478, upper bound: 17.9608225
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601478, upper bound: 17.9607669
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607010, upper bound: 17.9606300
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606964, upper bound: 17.9606300
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612793
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612223
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612680
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612227
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616609, upper bound: 17.9610260
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610239, upper bound: 17.9610135
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610212
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604346, upper bound: 17.9604018
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607049, upper bound: 17.9603997
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604271, upper bound: 17.9601548
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603538, upper bound: 17.9601562
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607958, upper bound: 17.9611837
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608126, upper bound: 17.9607958
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604379, upper bound: 17.9603896
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604379, upper bound: 17.9607232
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9604172
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9610805
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9605018
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9610556
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603939, upper bound: 17.9604459
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603932, upper bound: 17.9610211
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603960, upper bound: 17.9603896
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603960, upper bound: 17.9603896
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604794, upper bound: 17.9604710
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604785, upper bound: 17.9605080
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604800, upper bound: 17.9604710
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604773, upper bound: 17.9604710
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602733, upper bound: 17.9602733
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602733, upper bound: 17.9602850
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608563
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608782
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611070, upper bound: 17.9610568
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610568, upper bound: 17.9610568
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615863, upper bound: 17.9608563
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612837, upper bound: 17.9608563
time: 0.60 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9609290
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9608233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604058, upper bound: 17.9607109
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604059, upper bound: 17.9607381
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9606333, upper bound: 17.9611690
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9606333, upper bound: 17.9608706
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605218, upper bound: 17.9610201
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605231, upper bound: 17.9608089
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604285, upper bound: 17.9611520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604493, upper bound: 17.9609601
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604674, upper bound: 17.9609986
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604226, upper bound: 17.9611590
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604287, upper bound: 17.9607975
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604287, upper bound: 17.9604320
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603168, upper bound: 17.9609423
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603168, upper bound: 17.9606420
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612258, upper bound: 17.9612179
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612227, upper bound: 17.9612670
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9606300, upper bound: 17.9607329
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9606300, upper bound: 17.9606300
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604120, upper bound: 17.9604108
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604120, upper bound: 17.9604038
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612195, upper bound: 17.9607398
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612205, upper bound: 17.9607802
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610951
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9613888, upper bound: 17.9610631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9613354, upper bound: 17.9608973
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608642, upper bound: 17.9609107
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9613772, upper bound: 17.9605631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9613772, upper bound: 17.9605929
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9618071, upper bound: 17.9610531
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9618349, upper bound: 17.9610531
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605189, upper bound: 17.9610200
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605189, upper bound: 17.9606905
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605189, upper bound: 17.9611760
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605189, upper bound: 17.9607706
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9611512
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9610510
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9616315, upper bound: 17.9608534
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608534, upper bound: 17.9608566
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9616211, upper bound: 17.9608534
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9615870, upper bound: 17.9608534
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9606866, upper bound: 17.9604123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604123, upper bound: 17.9604123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604026, upper bound: 17.9605312
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604088, upper bound: 17.9604026
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604026, upper bound: 17.9605212
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604026, upper bound: 17.9604026
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9606136, upper bound: 17.9607478
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9606206, upper bound: 17.9610578
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9610302
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9608618
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610539
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9617984, upper bound: 17.9610539
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9618350, upper bound: 17.9610135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604126, upper bound: 17.9604126
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604126, upper bound: 17.9604126
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9617954, upper bound: 17.9610135
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9615130
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610242, upper bound: 17.9614863
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604126, upper bound: 17.9604215
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604126, upper bound: 17.9604212
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603978, upper bound: 17.9609340
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603912, upper bound: 17.9608616
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9601313, upper bound: 17.9602387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9601331, upper bound: 17.9605153
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9601520, upper bound: 17.9608018
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9601452, upper bound: 17.9607463
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9601478, upper bound: 17.9608225
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9601478, upper bound: 17.9607669
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9607010, upper bound: 17.9606300
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9606964, upper bound: 17.9606300
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612793
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612223
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612680
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612227
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9616609, upper bound: 17.9610260
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610239, upper bound: 17.9610135
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610212
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604346, upper bound: 17.9604018
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9607049, upper bound: 17.9603997
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604271, upper bound: 17.9601548
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603538, upper bound: 17.9601562
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9607958, upper bound: 17.9611837
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608126, upper bound: 17.9607958
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604379, upper bound: 17.9603896
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604379, upper bound: 17.9607232
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9604172
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9610805
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9605018
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9610556
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603939, upper bound: 17.9604459
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603932, upper bound: 17.9610211
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603960, upper bound: 17.9603896
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9603960, upper bound: 17.9603896
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604794, upper bound: 17.9604710
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604785, upper bound: 17.9605080
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604800, upper bound: 17.9604710
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9604773, upper bound: 17.9604710
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9602733, upper bound: 17.9602733
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9602733, upper bound: 17.9602850
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608563
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608782
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9611070, upper bound: 17.9610568
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9610568, upper bound: 17.9610568
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9615863, upper bound: 17.9608563
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.54
Output dim: 3, lower bound: -17.9612837, upper bound: 17.9608563

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605022, upper bound: 17.9609105
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605022, upper bound: 17.9605620
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605881
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605745
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604058, upper bound: 17.9606872
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604058, upper bound: 17.9607109
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603963, upper bound: 17.9607270
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603898, upper bound: 17.9604340
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606136, upper bound: 17.9611496
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606136, upper bound: 17.9606235
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604032, upper bound: 17.9606367
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604032, upper bound: 17.9606442
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605022, upper bound: 17.9610002
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605022, upper bound: 17.9605079
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602793, upper bound: 17.9604844
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602793, upper bound: 17.9604844
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604285, upper bound: 17.9609952
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604199, upper bound: 17.9611520
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604032, upper bound: 17.9606858
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604032, upper bound: 17.9609321
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604061, upper bound: 17.9609734
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604058, upper bound: 17.9609500
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9611143
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9604615
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603137, upper bound: 17.9606699
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603137, upper bound: 17.9603721
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603168, upper bound: 17.9609345
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603168, upper bound: 17.9609423
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606386, upper bound: 17.9606683
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606386, upper bound: 17.9606300
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9611153
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610673
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9606189
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9605447
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609572, upper bound: 17.9605356
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610020, upper bound: 17.9605198
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607239, upper bound: 17.9607682
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612069, upper bound: 17.9607802
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610539
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610951
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608153, upper bound: 17.9605252
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9605237
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608864
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608607
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603137, upper bound: 17.9603753
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603137, upper bound: 17.9603137
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609314, upper bound: 17.9602932
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603848, upper bound: 17.9602932
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605212, upper bound: 17.9605852
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605212, upper bound: 17.9605433
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9610432
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9610432
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610471, upper bound: 17.9610432
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9610432
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9609998
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9604993
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602766, upper bound: 17.9603575
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602799, upper bound: 17.9603077
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602970, upper bound: 17.9602970
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9602970, upper bound: 17.9609118
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603034, upper bound: 17.9602970
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603017, upper bound: 17.9605383
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9611512
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9611447
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611760, upper bound: 17.9605546
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605033, upper bound: 17.9605546
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608534, upper bound: 17.9608534
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608534, upper bound: 17.9608534
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602970, upper bound: 17.9603019
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602970, upper bound: 17.9602970
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609525, upper bound: 17.9601927
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601927, upper bound: 17.9601927
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608534, upper bound: 17.9608534
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615565, upper bound: 17.9608534
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604123, upper bound: 17.9604123
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606866, upper bound: 17.9604123
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603912, upper bound: 17.9603912
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603912, upper bound: 17.9605207
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9608886
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9606869
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9606581
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9609994
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9608618
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9605327, upper bound: 17.9608376
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605130, upper bound: 17.9605130
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605130, upper bound: 17.9605130
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605130, upper bound: 17.9605130
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605130, upper bound: 17.9605130
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610026, upper bound: 17.9610026
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610026, upper bound: 17.9610026
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608532
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608532
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608532
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615889, upper bound: 17.9608532
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610132, upper bound: 17.9612532
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610132, upper bound: 17.9614937
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9614787
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9602756, upper bound: 17.9608111
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9602756, upper bound: 17.9607932
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208
1: -7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301
2: -4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755
3: -8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169
4: -5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601288, upper bound: 17.9601288
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601288, upper bound: 17.9605141
time: 0.99 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605022, upper bound: 17.9609105
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605022, upper bound: 17.9605620
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605881
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603001, upper bound: 17.9605745
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604058, upper bound: 17.9606872
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604058, upper bound: 17.9607109
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603963, upper bound: 17.9607270
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603898, upper bound: 17.9604340
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9606136, upper bound: 17.9611496
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9606136, upper bound: 17.9606235
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604032, upper bound: 17.9606367
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604032, upper bound: 17.9606442
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605022, upper bound: 17.9610002
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605022, upper bound: 17.9605079
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602793, upper bound: 17.9604844
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602793, upper bound: 17.9604844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604285, upper bound: 17.9609952
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604199, upper bound: 17.9611520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604032, upper bound: 17.9606858
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604032, upper bound: 17.9609321
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604061, upper bound: 17.9609734
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604058, upper bound: 17.9609500
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9611143
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9604615
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603137, upper bound: 17.9606699
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603137, upper bound: 17.9603721
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603168, upper bound: 17.9609345
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603168, upper bound: 17.9609423
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9606386, upper bound: 17.9606683
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9606386, upper bound: 17.9606300
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9611153
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610673
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9606189
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9605447
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9609572, upper bound: 17.9605356
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610020, upper bound: 17.9605198
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9607239, upper bound: 17.9607682
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9612069, upper bound: 17.9607802
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610539
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610539, upper bound: 17.9610951
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608153, upper bound: 17.9605252
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9605237
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608864
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608607
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603137, upper bound: 17.9603753
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603137, upper bound: 17.9603137
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9609314, upper bound: 17.9602932
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603848, upper bound: 17.9602932
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605212, upper bound: 17.9605852
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605212, upper bound: 17.9605433
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9610432
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9610432
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610471, upper bound: 17.9610432
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9610432
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9609998
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9604993
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602766, upper bound: 17.9603575
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602799, upper bound: 17.9603077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602970, upper bound: 17.9602970
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602970, upper bound: 17.9609118
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603034, upper bound: 17.9602970
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603017, upper bound: 17.9605383
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9611512
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610432, upper bound: 17.9611447
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9611760, upper bound: 17.9605546
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605033, upper bound: 17.9605546
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608534, upper bound: 17.9608534
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608534, upper bound: 17.9608534
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602970, upper bound: 17.9603019
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602970, upper bound: 17.9602970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9609525, upper bound: 17.9601927
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9601927, upper bound: 17.9601927
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608534, upper bound: 17.9608534
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9615565, upper bound: 17.9608534
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604123, upper bound: 17.9604123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9606866, upper bound: 17.9604123
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603912, upper bound: 17.9603912
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9603912, upper bound: 17.9605207
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9608886
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9606869
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9606581
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9604993, upper bound: 17.9609994
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605159, upper bound: 17.9608618
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605327, upper bound: 17.9608376
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605130, upper bound: 17.9605130
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605130, upper bound: 17.9605130
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605130, upper bound: 17.9605130
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9605130, upper bound: 17.9605130
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610026, upper bound: 17.9610026
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610026, upper bound: 17.9610026
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608532
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608532
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9608532, upper bound: 17.9608532
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9615889, upper bound: 17.9608532
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610132, upper bound: 17.9612532
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610132, upper bound: 17.9614937
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9614787
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602756, upper bound: 17.9608111
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9602756, upper bound: 17.9607932
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9601288, upper bound: 17.9601288
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.99
Output dim: 3, lower bound: -17.9601288, upper bound: 17.9605141
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9601520, upper bound: 17.9608018
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9601452, upper bound: 17.9607463
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9601478, upper bound: 17.9608225
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9601478, upper bound: 17.9607669
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9607010, upper bound: 17.9606300
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9606964, upper bound: 17.9606300
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612793
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612223
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612680
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9612168, upper bound: 17.9612227
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9616609, upper bound: 17.9610260
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9610239, upper bound: 17.9610135
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610135
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9610135, upper bound: 17.9610212
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9607049, upper bound: 17.9603997
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9607958, upper bound: 17.9611837
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9608126, upper bound: 17.9607958
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9604379, upper bound: 17.9607232
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9610805
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9604064, upper bound: 17.9610556
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9603932, upper bound: 17.9610211
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608563
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9608563, upper bound: 17.9608782
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9608455, upper bound: 17.9608455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9611070, upper bound: 17.9610568
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9610568, upper bound: 17.9610568
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9615863, upper bound: 17.9608563
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 3, lower bound: -17.9612837, upper bound: 17.9608563

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.90 + 418.61 = 421.51 seconds
