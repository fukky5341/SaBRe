## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 560.5553892585241


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708)
1: (-216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265)
2: (-182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603)
3: (-194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453)
4: (-163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.06 + 2.23 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -560.5890246, upper bound: 560.5890246

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5890246, upper bound: 560.5889047
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5889047, upper bound: 560.5890246
time: 0.89 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.92 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 0, lower bound: -560.5890246, upper bound: 560.5889047
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 0, lower bound: -560.5889047, upper bound: 560.5890246

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5874722, upper bound: 560.5874778
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5874722, upper bound: 560.5875868
time: 0.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5886344, upper bound: 560.5890235
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5886344, upper bound: 560.5886344
time: 0.86 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.87 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 0, lower bound: -560.5874722, upper bound: 560.5874778
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 0, lower bound: -560.5874722, upper bound: 560.5875868
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 0, lower bound: -560.5886344, upper bound: 560.5890235
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 0, lower bound: -560.5886344, upper bound: 560.5886344

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5862543, upper bound: 560.5862543
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5862543, upper bound: 560.5862543
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5874625, upper bound: 560.5874779
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5874625, upper bound: 560.5875285
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5886344, upper bound: 560.5887528
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5886344, upper bound: 560.5890235
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5870966, upper bound: 560.5870966
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5870966, upper bound: 560.5870966
time: 0.84 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.80 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -560.5862543, upper bound: 560.5862543
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -560.5862543, upper bound: 560.5862543
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -560.5874625, upper bound: 560.5874779
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -560.5874625, upper bound: 560.5875285
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -560.5886344, upper bound: 560.5887528
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -560.5886344, upper bound: 560.5890235
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -560.5870966, upper bound: 560.5870966
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -560.5870966, upper bound: 560.5870966

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5856890, upper bound: 560.5856890
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5860313, upper bound: 560.5856890
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799234, upper bound: 560.5799234
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799234, upper bound: 560.5799234
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836808, upper bound: 560.5837077
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836808, upper bound: 560.5837077
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881311, upper bound: 560.5881311
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881311, upper bound: 560.5884574
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5855745, upper bound: 560.5859255
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5855745, upper bound: 560.5859255
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5828374, upper bound: 560.5828374
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5828374, upper bound: 560.5828374
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5848570, upper bound: 560.5848570
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5848570, upper bound: 560.5848570
time: 1.39 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5856890, upper bound: 560.5856890
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5860313, upper bound: 560.5856890
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5799234, upper bound: 560.5799234
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5799234, upper bound: 560.5799234
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5836808, upper bound: 560.5837077
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5836808, upper bound: 560.5837077
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5881311, upper bound: 560.5881311
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5881311, upper bound: 560.5884574
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5855745, upper bound: 560.5859255
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5855745, upper bound: 560.5859255
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5828374, upper bound: 560.5828374
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5828374, upper bound: 560.5828374
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5848570, upper bound: 560.5848570
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -560.5848570, upper bound: 560.5848570

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858402, upper bound: 560.5856513
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5856513, upper bound: 560.5856513
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777710, upper bound: 560.5777710
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777710, upper bound: 560.5777710
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5768683, upper bound: 560.5768683
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5768683, upper bound: 560.5768683
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5832243, upper bound: 560.5832383
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5832243, upper bound: 560.5832243
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5828103, upper bound: 560.5828103
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5828103, upper bound: 560.5828378
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812189, upper bound: 560.5812165
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5865538, upper bound: 560.5869086
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5865538, upper bound: 560.5865538
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881311, upper bound: 560.5884216
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881311, upper bound: 560.5884574
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807928, upper bound: 560.5807928
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807928, upper bound: 560.5810936
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5558767, upper bound: 560.5558767
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5558767, upper bound: 560.5558767
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5823632, upper bound: 560.5823632
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5823632, upper bound: 560.5823632
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5848570, upper bound: 560.5848570
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5848570, upper bound: 560.5848570
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5790025, upper bound: 560.5790025
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5790025, upper bound: 560.5790025
time: 1.03 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5858402, upper bound: 560.5856513
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5856513, upper bound: 560.5856513
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5777710, upper bound: 560.5777710
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5777710, upper bound: 560.5777710
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5768683, upper bound: 560.5768683
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5768683, upper bound: 560.5768683
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5832243, upper bound: 560.5832383
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5832243, upper bound: 560.5832243
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5828103, upper bound: 560.5828103
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5828103, upper bound: 560.5828378
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5812189, upper bound: 560.5812165
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5865538, upper bound: 560.5869086
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5865538, upper bound: 560.5865538
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5881311, upper bound: 560.5884216
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5881311, upper bound: 560.5884574
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5807928, upper bound: 560.5807928
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5807928, upper bound: 560.5810936
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5558767, upper bound: 560.5558767
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5558767, upper bound: 560.5558767
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5823632, upper bound: 560.5823632
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5823632, upper bound: 560.5823632
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5848570, upper bound: 560.5848570
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5848570, upper bound: 560.5848570
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5790025, upper bound: 560.5790025
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -560.5790025, upper bound: 560.5790025

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5762408, upper bound: 560.5762408
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5762408, upper bound: 560.5762408
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5856513, upper bound: 560.5856513
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5857781, upper bound: 560.5856513
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5844719, upper bound: 560.5844719
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5844719, upper bound: 560.5844719
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777710, upper bound: 560.5777710
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777710, upper bound: 560.5777710
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5736787, upper bound: 560.5734424
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5828103, upper bound: 560.5828378
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5828103, upper bound: 560.5828103
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799495, upper bound: 560.5799495
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799495, upper bound: 560.5799495
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801040, upper bound: 560.5801040
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801040, upper bound: 560.5801040
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5853788
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5853788
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5851070
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5851070
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867004, upper bound: 560.5867004
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867004, upper bound: 560.5867414
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5689337, upper bound: 560.5689337
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5689337, upper bound: 560.5689337
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5806999, upper bound: 560.5806999
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5806999, upper bound: 560.5806999
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5791368, upper bound: 560.5796483
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5791368, upper bound: 560.5796483
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5558767, upper bound: 560.5558767
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5558767, upper bound: 560.5558767
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5823632, upper bound: 560.5823632
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5823632, upper bound: 560.5823632
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5519270, upper bound: 560.5519270
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5519270, upper bound: 560.5519270
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5844810, upper bound: 560.5844810
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5844810, upper bound: 560.5844810
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5830406, upper bound: 560.5830406
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5830406, upper bound: 560.5830406
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5774538, upper bound: 560.5774538
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5774538, upper bound: 560.5774538
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
time: 1.01 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5762408, upper bound: 560.5762408
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5762408, upper bound: 560.5762408
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5856513, upper bound: 560.5856513
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5857781, upper bound: 560.5856513
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5844719, upper bound: 560.5844719
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5844719, upper bound: 560.5844719
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5777710, upper bound: 560.5777710
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5777710, upper bound: 560.5777710
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5736787, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5828103, upper bound: 560.5828378
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5828103, upper bound: 560.5828103
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5799495, upper bound: 560.5799495
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5799495, upper bound: 560.5799495
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5812165, upper bound: 560.5812165
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5801040, upper bound: 560.5801040
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5801040, upper bound: 560.5801040
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5853788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5853788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5851070
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5851070
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5867004, upper bound: 560.5867004
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5867004, upper bound: 560.5867414
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5689337, upper bound: 560.5689337
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5689337, upper bound: 560.5689337
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5806999, upper bound: 560.5806999
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5806999, upper bound: 560.5806999
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5791368, upper bound: 560.5796483
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5791368, upper bound: 560.5796483
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5558767, upper bound: 560.5558767
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5558767, upper bound: 560.5558767
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5823632, upper bound: 560.5823632
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5823632, upper bound: 560.5823632
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5519270, upper bound: 560.5519270
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5519270, upper bound: 560.5519270
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5844810, upper bound: 560.5844810
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5844810, upper bound: 560.5844810
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5830406, upper bound: 560.5830406
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5830406, upper bound: 560.5830406
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5774538, upper bound: 560.5774538
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5774538, upper bound: 560.5774538
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5614875, upper bound: 560.5614875
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5614875, upper bound: 560.5614875
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5762408, upper bound: 560.5762408
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5762408, upper bound: 560.5762408
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5443644, upper bound: 560.5443644
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5443644, upper bound: 560.5443644
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5856513, upper bound: 560.5856513
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5856513, upper bound: 560.5856513
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836915, upper bound: 560.5836915
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836915, upper bound: 560.5836915
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788483, upper bound: 560.5788483
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788483, upper bound: 560.5788483
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5739262, upper bound: 560.5734424
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5773532, upper bound: 560.5773532
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5773532, upper bound: 560.5773532
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5714157, upper bound: 560.5710265
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5710265, upper bound: 560.5710265
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684805, upper bound: 560.5684805
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684805, upper bound: 560.5684805
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5760504, upper bound: 560.5760504
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5760504, upper bound: 560.5760504
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601440, upper bound: 560.5601440
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601440, upper bound: 560.5601440
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777205, upper bound: 560.5777205
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777205, upper bound: 560.5777205
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5460401, upper bound: 560.5460401
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5460401, upper bound: 560.5460401
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5827974, upper bound: 560.5827974
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5827974, upper bound: 560.5827974
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5733197, upper bound: 560.5733197
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5733197, upper bound: 560.5733197
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5791639, upper bound: 560.5791639
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5791639, upper bound: 560.5791639
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782533, upper bound: 560.5782533
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782533, upper bound: 560.5782533
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5767176, upper bound: 560.5767176
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5767176, upper bound: 560.5767463
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5626773, upper bound: 560.5626773
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5626773, upper bound: 560.5626773
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5626964, upper bound: 560.5626964
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5626964, upper bound: 560.5626964
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792763, upper bound: 560.5792763
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792763, upper bound: 560.5792763
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5851070
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5853788
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793814, upper bound: 560.5797758
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793814, upper bound: 560.5797756
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5846308, upper bound: 560.5846407
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5846308, upper bound: 560.5846308
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5834860, upper bound: 560.5834860
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5834860, upper bound: 560.5834860
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5845854, upper bound: 560.5845854
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5845854, upper bound: 560.5845854
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5645746, upper bound: 560.5645746
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5645746, upper bound: 560.5645746
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5678716, upper bound: 560.5678716
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5678716, upper bound: 560.5678716
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5791419, upper bound: 560.5791419
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5791419, upper bound: 560.5791419
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5744773, upper bound: 560.5744773
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5744773, upper bound: 560.5744773
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5778076, upper bound: 560.5778076
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5778076, upper bound: 560.5779049
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5735729, upper bound: 560.5736285
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5735729, upper bound: 560.5735729
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5544990, upper bound: 560.5544990
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5544990, upper bound: 560.5544990
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5503482, upper bound: 560.5502500
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5502500, upper bound: 560.5502500
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5509979, upper bound: 560.5509979
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5509979, upper bound: 560.5509979
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5743932, upper bound: 560.5743932
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5743932, upper bound: 560.5743932
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5825744, upper bound: 560.5825744
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5825744, upper bound: 560.5825744
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5802605, upper bound: 560.5802605
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5802605, upper bound: 560.5802605
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5771719, upper bound: 560.5771719
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5771719, upper bound: 560.5771719
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5761257, upper bound: 560.5761257
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5761257, upper bound: 560.5761257
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5715765, upper bound: 560.5715765
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5715765, upper bound: 560.5715765
time: 0.88 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5614875, upper bound: 560.5614875
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5614875, upper bound: 560.5614875
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5762408, upper bound: 560.5762408
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5762408, upper bound: 560.5762408
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5443644, upper bound: 560.5443644
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5443644, upper bound: 560.5443644
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5856513, upper bound: 560.5856513
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5856513, upper bound: 560.5856513
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5836915, upper bound: 560.5836915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5836915, upper bound: 560.5836915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5788483, upper bound: 560.5788483
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5788483, upper bound: 560.5788483
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5739262, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5773532, upper bound: 560.5773532
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5773532, upper bound: 560.5773532
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5714157, upper bound: 560.5710265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5710265, upper bound: 560.5710265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5684805, upper bound: 560.5684805
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5684805, upper bound: 560.5684805
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5760504, upper bound: 560.5760504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5760504, upper bound: 560.5760504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5601440, upper bound: 560.5601440
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5601440, upper bound: 560.5601440
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5777205, upper bound: 560.5777205
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5777205, upper bound: 560.5777205
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5460401, upper bound: 560.5460401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5460401, upper bound: 560.5460401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5827974, upper bound: 560.5827974
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5827974, upper bound: 560.5827974
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5733197, upper bound: 560.5733197
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5733197, upper bound: 560.5733197
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5791639, upper bound: 560.5791639
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5791639, upper bound: 560.5791639
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5782533, upper bound: 560.5782533
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5782533, upper bound: 560.5782533
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5767176, upper bound: 560.5767176
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5767176, upper bound: 560.5767463
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5626773, upper bound: 560.5626773
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5626773, upper bound: 560.5626773
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5626964, upper bound: 560.5626964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5626964, upper bound: 560.5626964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5792763, upper bound: 560.5792763
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5792763, upper bound: 560.5792763
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5851070
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5853788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5793814, upper bound: 560.5797758
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5793814, upper bound: 560.5797756
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5846308, upper bound: 560.5846407
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5846308, upper bound: 560.5846308
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5834860, upper bound: 560.5834860
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5834860, upper bound: 560.5834860
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5845854, upper bound: 560.5845854
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5845854, upper bound: 560.5845854
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5645746, upper bound: 560.5645746
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5645746, upper bound: 560.5645746
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5678716, upper bound: 560.5678716
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5678716, upper bound: 560.5678716
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5791419, upper bound: 560.5791419
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5791419, upper bound: 560.5791419
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5744773, upper bound: 560.5744773
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5744773, upper bound: 560.5744773
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5778076, upper bound: 560.5778076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5778076, upper bound: 560.5779049
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5735729, upper bound: 560.5736285
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5735729, upper bound: 560.5735729
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5548826, upper bound: 560.5548826
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5544990, upper bound: 560.5544990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5544990, upper bound: 560.5544990
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5503482, upper bound: 560.5502500
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5502500, upper bound: 560.5502500
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5509979, upper bound: 560.5509979
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5509979, upper bound: 560.5509979
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5743932, upper bound: 560.5743932
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5743932, upper bound: 560.5743932
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5825744, upper bound: 560.5825744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5825744, upper bound: 560.5825744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5802605, upper bound: 560.5802605
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5802605, upper bound: 560.5802605
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5771719, upper bound: 560.5771719
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5771719, upper bound: 560.5771719
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5761257, upper bound: 560.5761257
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5761257, upper bound: 560.5761257
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5715765, upper bound: 560.5715765
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -560.5715765, upper bound: 560.5715765

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5573465, upper bound: 560.5573465
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5573465, upper bound: 560.5573465
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5508045, upper bound: 560.5508045
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5508045, upper bound: 560.5508045
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5459307, upper bound: 560.5459307
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5459307, upper bound: 560.5459307
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5733820, upper bound: 560.5733820
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5733820, upper bound: 560.5733820
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5706846, upper bound: 560.5706846
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5706846, upper bound: 560.5706846
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788595, upper bound: 560.5788595
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788595, upper bound: 560.5788595
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5614140, upper bound: 560.5614140
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5614140, upper bound: 560.5614140
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5841807, upper bound: 560.5841807
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5841807, upper bound: 560.5841807
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836716, upper bound: 560.5836716
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836716, upper bound: 560.5836716
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815997, upper bound: 560.5815997
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815997, upper bound: 560.5815997
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788483, upper bound: 560.5788483
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788483, upper bound: 560.5788483
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5499000, upper bound: 560.5499000
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5499000, upper bound: 560.5499000
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725767, upper bound: 560.5725767
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5732158, upper bound: 560.5725767
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5773532, upper bound: 560.5773532
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5773532, upper bound: 560.5773532
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5773500, upper bound: 560.5773500
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5773500, upper bound: 560.5773500
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5714157, upper bound: 560.5710265
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5710265, upper bound: 560.5710265
time: 0.92 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5573465, upper bound: 560.5573465
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5573465, upper bound: 560.5573465
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5508045, upper bound: 560.5508045
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5508045, upper bound: 560.5508045
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5800142, upper bound: 560.5800142
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5459307, upper bound: 560.5459307
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5459307, upper bound: 560.5459307
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5733820, upper bound: 560.5733820
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5733820, upper bound: 560.5733820
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5706846, upper bound: 560.5706846
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5706846, upper bound: 560.5706846
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5788595, upper bound: 560.5788595
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5788595, upper bound: 560.5788595
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5614140, upper bound: 560.5614140
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5614140, upper bound: 560.5614140
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5800377, upper bound: 560.5800377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5841807, upper bound: 560.5841807
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5841807, upper bound: 560.5841807
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5836716, upper bound: 560.5836716
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5836716, upper bound: 560.5836716
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5815997, upper bound: 560.5815997
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5815997, upper bound: 560.5815997
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5788483, upper bound: 560.5788483
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5788483, upper bound: 560.5788483
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5499000, upper bound: 560.5499000
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5499000, upper bound: 560.5499000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5725767, upper bound: 560.5725767
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5732158, upper bound: 560.5725767
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5773532, upper bound: 560.5773532
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5773532, upper bound: 560.5773532
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5773500, upper bound: 560.5773500
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5773500, upper bound: 560.5773500
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5714157, upper bound: 560.5710265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.33
Output dim: 0, lower bound: -560.5710265, upper bound: 560.5710265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5710265, upper bound: 560.5710265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5734424, upper bound: 560.5734424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5684805, upper bound: 560.5684805
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5684805, upper bound: 560.5684805
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5718812, upper bound: 560.5718812
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5768486, upper bound: 560.5768486
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5770025, upper bound: 560.5770025
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5760504, upper bound: 560.5760504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5760504, upper bound: 560.5760504
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5601440, upper bound: 560.5601440
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5601440, upper bound: 560.5601440
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5824665, upper bound: 560.5824665
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5777205, upper bound: 560.5777205
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5777205, upper bound: 560.5777205
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5782375, upper bound: 560.5782375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5827974, upper bound: 560.5827974
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5827974, upper bound: 560.5827974
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5733197, upper bound: 560.5733197
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5733197, upper bound: 560.5733197
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5791639, upper bound: 560.5791639
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5791639, upper bound: 560.5791639
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5782533, upper bound: 560.5782533
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5782533, upper bound: 560.5782533
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5767176, upper bound: 560.5767176
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5767176, upper bound: 560.5767463
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5626773, upper bound: 560.5626773
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5626773, upper bound: 560.5626773
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5626964, upper bound: 560.5626964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5626964, upper bound: 560.5626964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5792763, upper bound: 560.5792763
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5792763, upper bound: 560.5792763
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5851070
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5851070, upper bound: 560.5853788
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5793814, upper bound: 560.5797758
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5793814, upper bound: 560.5797756
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5846308, upper bound: 560.5846407
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5846308, upper bound: 560.5846308
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5834860, upper bound: 560.5834860
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5834860, upper bound: 560.5834860
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5845854, upper bound: 560.5845854
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5845854, upper bound: 560.5845854
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5801110, upper bound: 560.5801110
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5645746, upper bound: 560.5645746
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5645746, upper bound: 560.5645746
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5678716, upper bound: 560.5678716
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5678716, upper bound: 560.5678716
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5791419, upper bound: 560.5791419
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5791419, upper bound: 560.5791419
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5744773, upper bound: 560.5744773
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5744773, upper bound: 560.5744773
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5778076, upper bound: 560.5778076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5778076, upper bound: 560.5779049
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5735729, upper bound: 560.5736285
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5735729, upper bound: 560.5735729
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5745145, upper bound: 560.5745145
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5794032, upper bound: 560.5794032
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5743932, upper bound: 560.5743932
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5743932, upper bound: 560.5743932
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5825744, upper bound: 560.5825744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5825744, upper bound: 560.5825744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5802605, upper bound: 560.5802605
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5802605, upper bound: 560.5802605
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5815921, upper bound: 560.5815921
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5771719, upper bound: 560.5771719
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5771719, upper bound: 560.5771719
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5761257, upper bound: 560.5761257
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5761257, upper bound: 560.5761257
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5774230, upper bound: 560.5774230
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5715765, upper bound: 560.5715765
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.33
Output dim: 0, lower bound: -560.5715765, upper bound: 560.5715765

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.29 + 417.81 = 421.10 seconds
