## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
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
execution time: IAR + RelationalAnalysis = 0.70 + 2.19 = 2.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -560.5890246, upper bound: 560.5890246

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871664, upper bound: 560.5871664
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871664, upper bound: 560.5871664
time: 0.85 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.77 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -560.5871664, upper bound: 560.5871664
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -560.5871664, upper bound: 560.5871664

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5553001
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5553001
time: 0.87 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 32

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
time: 0.67 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.04 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5553001
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5553001
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5553001
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5554974, upper bound: 560.5553001
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5553001
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5554974, upper bound: 560.5553001
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
time: 0.97 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.34 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5553001
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -560.5554974, upper bound: 560.5553001
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5553001
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -560.5554974, upper bound: 560.5553001
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
time: 0.74 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.64
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.94 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.94
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.94
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5554974
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.94
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.94
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5556199
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.94
Output dim: 0, lower bound: -560.5553001, upper bound: 560.5553001

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5521646
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5524824
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5525144
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5526451
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5521437
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5524824
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708
1: -216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265
2: -182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603
3: -194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453
4: -163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5525144
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5526451
time: 0.77 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.47 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5521646
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5524824
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5525144
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5526451
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5521437
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5524824
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5525144
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -560.5521437, upper bound: 560.5526451

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.89 + 54.49 = 57.38 seconds
