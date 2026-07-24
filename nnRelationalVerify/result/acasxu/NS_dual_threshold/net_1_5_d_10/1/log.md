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
execution time: IAR + RelationalAnalysis = 0.68 + 2.21 = 2.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -560.5890246, upper bound: 560.5890246

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5796035, upper bound: 560.5577977
time: 1.09 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.97 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 0, lower bound: -560.5796035, upper bound: 560.5577977
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -147.5333099, 450.3576050, -149.7455139, 457.4793701, -605.0126953, 600.1031494
1: -209.3132782, 456.5391235, -212.4545288, 463.6163940, -672.9296875, 668.9936523
2: -176.9316559, 503.9251099, -179.5741730, 511.6823730, -688.6139526, 683.4992676
3: -188.3572540, 631.6350098, -191.1816559, 641.5441284, -829.9013672, 822.8166504
4: -157.8749084, 582.0621338, -160.2293701, 591.1250610, -749.0000000, 742.2915039

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199
time: 0.85 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199
time: 0.93 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -207.2558289, 632.3292847, -144.1444244, 440.0426941, -647.2984009, 776.4736328
1: -293.6619568, 641.0078735, -204.3473206, 445.6375427, -739.2993164, 845.3551636
2: -247.8776855, 708.6886597, -172.7861633, 491.7388916, -739.6165771, 881.4747925
3: -264.6026917, 885.5980835, -183.8718719, 616.6253662, -881.2280273, 1069.4699707
4: -221.8534088, 817.2335815, -154.1671295, 568.0655518, -789.9189453, 971.4006958

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199
time: 0.64 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.28 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -560.5556199, upper bound: 560.5556199

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -147.5333099, 450.3576050, -147.5333099, 450.3576050, -597.8909302, 597.8909302
1: -209.3132782, 456.5391235, -209.3132782, 456.5391235, -665.8523560, 665.8523560
2: -176.9316559, 503.9251099, -176.9316559, 503.9251099, -680.8567505, 680.8567505
3: -188.3572540, 631.6350098, -188.3572540, 631.6350098, -819.9921875, 819.9921875
4: -157.8749084, 582.0621338, -157.8749084, 582.0621338, -739.9370117, 739.9370117

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5612772, upper bound: 560.5350945
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5722162, upper bound: 560.5500290
time: 1.13 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -147.5333099, 450.3576050, -207.2558289, 632.3292847, -779.8625488, 657.6133423
1: -209.3132782, 456.5391235, -293.6619568, 641.0078735, -850.3210449, 750.2010498
2: -176.9316559, 503.9251099, -247.8776855, 708.6886597, -885.6203003, 751.8027344
3: -188.3572540, 631.6350098, -264.6026917, 885.5980835, -1073.9553223, 896.2376709
4: -157.8749084, 582.0621338, -221.8534088, 817.2335815, -975.1085205, 803.9155273

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601164, upper bound: 560.5228756
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5722162, upper bound: 560.5500289
time: 0.74 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -207.2558289, 632.3292847, -147.5333099, 450.3576050, -657.6133423, 779.8625488
1: -293.6619568, 641.0078735, -209.3132782, 456.5391235, -750.2010498, 850.3210449
2: -247.8776855, 708.6886597, -176.9316559, 503.9251099, -751.8027344, 885.6203003
3: -264.6026917, 885.5980835, -188.3572540, 631.6350098, -896.2376709, 1073.9553223
4: -221.8534088, 817.2335815, -157.8749084, 582.0621338, -803.9155273, 975.1085205

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5201941, upper bound: 560.5349188
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5471082, upper bound: 560.5471082
time: 0.80 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -207.2558289, 632.3292847, -207.2558289, 632.3292847, -839.5849609, 839.5849609
1: -293.6619568, 641.0078735, -293.6619568, 641.0078735, -934.2750854, 934.2750854
2: -247.8776855, 708.6886597, -247.8776855, 708.6886597, -955.8807983, 955.8807983
3: -264.6026917, 885.5980835, -264.6026917, 885.5980835, -1150.2008057, 1150.2008057
4: -221.8534088, 817.2335815, -221.8534088, 817.2335815, -1039.0870361, 1039.0870361

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5201941, upper bound: 560.5349188
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5471082, upper bound: 560.5471082
time: 0.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.05 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -560.5612772, upper bound: 560.5350945
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -560.5722162, upper bound: 560.5500290
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -560.5601164, upper bound: 560.5228756
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -560.5722162, upper bound: 560.5500289
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -560.5201941, upper bound: 560.5349188
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -560.5471082, upper bound: 560.5471082
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -560.5201941, upper bound: 560.5349188
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.05
Output dim: 0, lower bound: -560.5471082, upper bound: 560.5471082

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -136.9476013, 418.8932190, -141.9362793, 434.2890015, -571.2365723, 560.8294678
1: -193.9745483, 424.5124207, -201.1533356, 440.0070801, -633.9816284, 625.6657715
2: -164.1404266, 468.5742798, -170.0864258, 485.6631165, -649.8035278, 638.6605835
3: -174.6679230, 587.7494507, -181.0889740, 609.1558228, -783.8237305, 768.8383789
4: -146.5756683, 541.1690674, -151.8472595, 560.9678955, -707.5435791, 693.0162354

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5817993, upper bound: 560.5800723
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5821057, upper bound: 560.5806249
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -139.4508057, 423.7216797, -142.4678802, 433.6678772, -573.1186523, 566.1895752
1: -197.7113037, 430.0884705, -202.0447693, 439.9662170, -637.6773071, 632.1332397
2: -167.1753998, 475.1041565, -170.8153687, 485.8504333, -653.0257568, 645.9195557
3: -177.8843536, 594.8735352, -181.7925110, 608.5966187, -786.4809570, 776.6658936
4: -149.1687775, 548.5308228, -152.4100647, 561.0543823, -710.2230835, 700.9409180

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807598, upper bound: 560.5811983
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5809057, upper bound: 560.5812206
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -141.9362793, 434.2890015, -198.7720490, 607.0061035, -748.9423828, 633.0610352
1: -201.1533356, 440.0070801, -281.4117432, 615.2539062, -816.4072266, 721.4188232
2: -170.0864258, 485.6631165, -237.6312256, 680.3226318, -850.4002686, 723.2943115
3: -181.0889740, 609.1558228, -253.5691071, 850.3156738, -1031.4046631, 862.7249146
4: -151.8472595, 560.9678955, -212.6939850, 784.3975220, -936.2446899, 773.6618652

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5580488, upper bound: 560.5213052
time: 0.92 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5583446, upper bound: 560.5216107
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -142.4678802, 433.6678772, -197.5293884, 602.1398315, -744.6077271, 631.1972656
1: -202.0447693, 439.9662170, -279.6226501, 610.5472412, -812.5920410, 719.5888062
2: -170.8153687, 485.8504333, -236.0990906, 675.2128906, -846.0282593, 721.9495239
3: -181.7925110, 608.5966187, -251.9922943, 843.7869263, -1025.5791016, 860.5888672
4: -152.4100647, 561.0543823, -211.3507996, 778.6357422, -931.0457764, 772.4051514

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5710687, upper bound: 560.5496828
time: 0.86 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5719637, upper bound: 560.5498615
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.22 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -560.5817993, upper bound: 560.5800723
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -560.5821057, upper bound: 560.5806249
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -560.5807598, upper bound: 560.5811983
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -560.5809057, upper bound: 560.5812206
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -560.5580488, upper bound: 560.5213052
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -560.5583446, upper bound: 560.5216107
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -560.5710687, upper bound: 560.5496828
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -560.5719637, upper bound: 560.5498615

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -134.7703247, 412.2543945, -140.0648346, 423.0566101, -557.8269043, 552.3192139
1: -190.8764954, 417.7572937, -198.0234222, 430.1191711, -620.9956665, 615.7807007
2: -161.5099030, 461.0968018, -167.5928802, 475.4104919, -636.9202271, 628.6896973
3: -171.8726959, 578.3549805, -178.3780670, 595.0413818, -766.9140625, 756.7330322
4: -144.2324066, 532.4791260, -149.8056946, 549.0219727, -693.2542114, 682.2847900

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5805024, upper bound: 560.5739056
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5805024, upper bound: 560.5794936
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -135.8740234, 415.5410767, -139.9141693, 427.9280701, -563.8021240, 555.4550781
1: -192.4600830, 421.1465454, -198.2748108, 433.6564636, -626.1165771, 619.4213867
2: -162.8577271, 464.8512573, -167.6568756, 478.6735840, -641.5311890, 632.5078735
3: -173.3024139, 583.0310059, -178.5078583, 600.2459717, -773.5484009, 761.5388184
4: -145.4239655, 536.8195190, -149.6753235, 552.7846069, -698.2085571, 686.4948120

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5817925, upper bound: 560.5795772
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5821057, upper bound: 560.5806249
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -137.4003296, 412.2813110, -140.2733765, 426.9793091, -564.3796387, 552.5546265
1: -194.3476410, 420.1194763, -198.9301147, 433.1492004, -627.4967651, 619.0494995
2: -164.4981232, 464.5071106, -168.1688080, 478.2703857, -642.7684937, 632.6759033
3: -174.9273224, 580.5049438, -178.9854584, 599.0995483, -774.0268555, 759.4904175
4: -146.9221344, 536.2752075, -150.0524292, 552.2630005, -699.1851196, 686.3275757

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5744628, upper bound: 560.5805770
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788010, upper bound: 560.5811983
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -137.5756683, 417.8421326, -141.4378662, 430.4339600, -568.0095215, 559.2800293
1: -195.0400543, 424.2109070, -200.5793610, 436.7359619, -631.7759399, 624.7902222
2: -164.9200134, 468.6439209, -169.5780334, 482.2945557, -647.2144775, 638.2219238
3: -175.4945984, 586.6203613, -180.4797516, 604.0618286, -779.5563965, 767.1000977
4: -147.1547089, 540.9414062, -151.3042755, 556.8835449, -704.0382690, 692.2456055

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800774, upper bound: 560.5808314
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812206, upper bound: 560.5812206
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -137.6110229, 419.9641113, -197.1031189, 601.6599731, -739.2709351, 617.0672607
1: -194.8897095, 425.7550354, -278.9777832, 609.8690796, -804.7586670, 704.7327271
2: -164.8454742, 469.9340820, -235.5810852, 674.3852539, -839.2180176, 705.5150757
3: -175.4532776, 589.0556641, -251.3770142, 842.7697754, -1018.2230225, 840.4326782
4: -147.1806183, 542.4421387, -210.8643188, 777.4183960, -924.5989990, 753.3064575

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5216120, upper bound: 560.5160119
time: 0.84 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5109870, upper bound: 560.4582492
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -138.2086182, 422.1507568, -196.6308136, 600.1162109, -738.3247681, 618.7815552
1: -195.7799988, 427.9135132, -278.3195801, 608.3742065, -804.1541748, 706.2330933
2: -165.5435028, 472.4029236, -235.0291748, 672.7609253, -838.2939453, 707.4321289
3: -176.2510376, 592.2410889, -250.7820740, 840.6701660, -1016.9212036, 843.0231934
4: -147.7921753, 545.4767456, -210.3724060, 775.5031738, -923.2953491, 755.8491211

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5527027, upper bound: 560.5200233
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5138562, upper bound: 560.4586670
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -123.9498520, 379.0828857, -192.4822845, 586.0454712, -709.9951782, 571.5651245
1: -174.7965546, 383.3703918, -272.4774170, 594.3941650, -769.1907349, 655.8477173
2: -147.6246796, 422.9707336, -230.0810852, 657.4426270, -805.0673218, 653.0517578
3: -157.6212311, 530.8798218, -245.5574036, 821.3351440, -978.9562378, 776.4370728
4: -131.8917236, 488.1676636, -205.9928284, 758.0965576, -889.9882202, 694.1604614

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5458676
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5496828
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -139.8050385, 425.8387451, -193.3033295, 589.8495483, -729.6546021, 619.1419678
1: -198.3893127, 431.9798584, -273.7987366, 598.2105103, -796.5997925, 705.7785645
2: -167.7331543, 476.9170837, -231.2171936, 661.5890503, -829.1614380, 708.1342773
3: -178.4602661, 597.4674072, -246.7593231, 826.7421875, -1005.2023926, 844.2267456
4: -149.6310730, 550.7415161, -206.9746857, 762.6762695, -912.3073120, 757.7161865

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5458676
time: 0.99 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5498615
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.53 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5805024, upper bound: 560.5739056
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5805024, upper bound: 560.5794936
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5817925, upper bound: 560.5795772
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5821057, upper bound: 560.5806249
NS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5744628, upper bound: 560.5805770
NS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5788010, upper bound: 560.5811983
NS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5800774, upper bound: 560.5808314
NS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5812206, upper bound: 560.5812206
NS_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5216120, upper bound: 560.5160119
NS_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5109870, upper bound: 560.4582492
NS_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5527027, upper bound: 560.5200233
NS_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5138562, upper bound: 560.4586670
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5458676
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5496828
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5458676
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5498615

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -126.0516281, 389.9027405, -128.2484436, 393.1459656, -519.1975098, 518.1511841
1: -178.6038361, 394.3334351, -181.8162384, 399.4572449, -578.0610962, 576.1496582
2: -151.0581055, 435.3117981, -153.8049316, 441.8794250, -592.9375000, 589.1166992
3: -160.9276123, 546.8748779, -163.9531250, 554.6665649, -715.5941162, 710.8280029
4: -135.1717834, 503.0744629, -137.9818878, 511.9616394, -647.1334229, 641.0561523

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5804837, upper bound: 560.5736679
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5804837, upper bound: 560.5737237
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -131.3639069, 402.0384216, -135.7411957, 409.6784363, -541.0423584, 537.7796021
1: -186.0383301, 407.4203491, -191.9468689, 416.6437073, -602.6819458, 599.3671875
2: -157.4275360, 449.5829773, -162.4503479, 460.4765930, -617.9041138, 612.0330811
3: -167.5415039, 563.9408569, -172.9194794, 576.2484131, -743.7899170, 736.8603516
4: -140.5718842, 518.9539185, -145.2245789, 531.6247559, -672.1966553, 664.1784668

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5758217, upper bound: 560.5782016
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5758217, upper bound: 560.5794936
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -131.9926300, 403.8985291, -135.2697296, 413.8627319, -545.8553467, 539.1682739
1: -187.1132507, 409.4009705, -191.8499756, 419.4821472, -606.5952759, 601.2508545
2: -158.2937317, 451.7450256, -162.1836090, 462.8716125, -621.1653442, 613.9286499
3: -168.4711609, 566.6203613, -172.7080841, 580.4687500, -748.9399414, 739.3284302
4: -141.3221893, 521.5796509, -144.7474365, 534.4697876, -675.7918091, 666.3270874

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5754550, upper bound: 560.5786145
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5813114, upper bound: 560.5795772
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -131.8014679, 403.0802917, -134.5046692, 407.7797852, -539.5812378, 537.5848389
1: -186.7210693, 408.5084534, -190.3895416, 414.5104065, -601.2314453, 598.8980103
2: -157.9656982, 450.8504944, -161.0200653, 458.0421753, -616.0078735, 611.8705444
3: -168.1152039, 565.4299316, -171.5662384, 573.3390503, -741.4542236, 736.9961548
4: -141.0220337, 520.5326538, -143.8100891, 528.7674561, -669.7894287, 664.3427124

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5821057, upper bound: 560.5802112
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5821057, upper bound: 560.5804734
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -127.2216568, 389.0220642, -132.0849304, 406.1372681, -533.3588867, 521.1069946
1: -180.5147705, 395.6608276, -187.3695374, 411.2805786, -591.7952881, 583.0303345
2: -152.7038269, 437.6056519, -158.3393402, 454.1428833, -606.8466797, 595.9450073
3: -162.6417084, 548.8539429, -168.6854401, 569.7296143, -732.3713379, 717.5393066
4: -136.8647308, 506.8750305, -141.5489807, 524.8323975, -661.6970825, 648.4237671

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5733364, upper bound: 560.5792949
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5738832, upper bound: 560.5800970
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -134.2155762, 402.2713013, -136.2275085, 414.6970215, -548.9125977, 538.4987183
1: -189.8871460, 410.0367737, -193.1283264, 420.7638550, -610.6510010, 603.1651001
2: -160.7288055, 453.3092651, -163.2767029, 464.5666809, -625.2954712, 616.5859375
3: -170.9192810, 566.4020996, -173.7966614, 581.8134766, -752.7327881, 740.1987305
4: -143.5718384, 523.2652588, -145.6757965, 536.0880737, -679.6598511, 668.9410400

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5796243, upper bound: 560.5787614
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5798739, upper bound: 560.5805949
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -133.0138397, 403.9935913, -138.2498474, 420.6930237, -553.7068481, 542.2434082
1: -188.6998138, 410.2529602, -196.1518555, 426.9386597, -615.6383057, 606.4047852
2: -159.5364532, 453.1399841, -165.8169708, 471.4218140, -630.9581299, 618.9569702
3: -169.7954102, 567.1499023, -176.4949036, 590.3965454, -760.1919556, 743.6447754
4: -142.3313446, 522.9367676, -147.9328918, 544.2662964, -686.5974121, 670.8696289

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5673508, upper bound: 560.5740001
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5752678, upper bound: 560.5765469
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -129.5399628, 390.1589050, -136.7516022, 416.2782288, -545.8181763, 526.9104004
1: -183.3068390, 397.4616394, -194.0097351, 422.3374939, -605.6443481, 591.4713745
2: -155.0916138, 439.4961548, -163.9595184, 466.3064575, -621.3980713, 603.4556274
3: -165.0343475, 549.2252197, -174.5483398, 583.9780884, -749.0123291, 723.7735596
4: -138.4447327, 506.9289551, -146.2702026, 538.2209473, -676.6656494, 653.1990967

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5210572, upper bound: 560.5078240
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5774222, upper bound: 560.5774222
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -123.9498520, 379.0828857, -180.0895386, 548.6288452, -672.5786743, 559.1723633
1: -174.7965546, 383.3703918, -254.1669312, 555.7836914, -730.5802002, 637.5371704
2: -147.6246796, 422.9707336, -214.3512421, 614.4018555, -762.0265503, 637.3218384
3: -157.6212311, 530.8798218, -229.2562714, 767.7781372, -925.3992310, 760.1359863
4: -131.8917236, 488.1676636, -191.8708801, 708.0519409, -839.9436035, 680.0385132

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5516706, upper bound: 560.5433117
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5656584, upper bound: 560.5458676
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -123.9498520, 379.0828857, -191.2283478, 583.9119873, -707.8616943, 570.3111572
1: -174.7965546, 383.3703918, -270.9284363, 592.2199097, -767.0164795, 654.2987671
2: -147.6246796, 422.9707336, -228.8025360, 654.9713135, -802.5960083, 651.7732544
3: -157.6212311, 530.8798218, -244.1843719, 818.5407104, -976.1618042, 775.0640869
4: -131.8917236, 488.1676636, -204.8168182, 754.9985352, -886.8902588, 692.9844360

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5516706, upper bound: 560.5457502
time: 1.10 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5656584, upper bound: 560.5496828
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -139.8050385, 425.8387451, -180.0895386, 548.6288452, -688.4338989, 605.9282227
1: -198.3893127, 431.9798584, -254.1669312, 555.7836914, -754.1729126, 686.1467896
2: -167.7331543, 476.9170837, -214.3512421, 614.4018555, -781.9368286, 691.2682495
3: -178.4602661, 597.4674072, -229.2562714, 767.7781372, -946.2383423, 826.7236938
4: -149.6310730, 550.7415161, -191.8708801, 708.0519409, -857.6829834, 742.6124268

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5464537, upper bound: 560.5397254
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5429310, upper bound: 560.5433078
time: 1.04 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5458676
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -139.8050385, 425.8387451, -191.2283478, 583.9119873, -723.7169800, 617.0670166
1: -198.3893127, 431.9798584, -270.9284363, 592.2199097, -790.6091919, 702.9083252
2: -167.7331543, 476.9170837, -228.8025360, 654.9713135, -822.5653076, 705.7196045
3: -178.4602661, 597.4674072, -244.1843719, 818.5407104, -997.0009155, 841.6517944
4: -149.6310730, 550.7415161, -204.8168182, 754.9985352, -904.6295776, 755.5583496

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5464537, upper bound: 560.5412198
time: 0.98 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5429310, upper bound: 560.5434727
time: 0.89 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5491019
time: 0.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.55 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5804837, upper bound: 560.5736679
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5804837, upper bound: 560.5737237
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5758217, upper bound: 560.5782016
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5758217, upper bound: 560.5794936
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5754550, upper bound: 560.5786145
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5813114, upper bound: 560.5795772
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5821057, upper bound: 560.5802112
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5821057, upper bound: 560.5804734
NS_A1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5733364, upper bound: 560.5792949
NS_A1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5738832, upper bound: 560.5800970
NS_A1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5796243, upper bound: 560.5787614
NS_A1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5798739, upper bound: 560.5805949
NS_A1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5673508, upper bound: 560.5740001
NS_A1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5752678, upper bound: 560.5765469
NS_A1_B1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5210572, upper bound: 560.5078240
NS_A1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5774222, upper bound: 560.5774222
NS_A1_B2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5516706, upper bound: 560.5433117
NS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5656584, upper bound: 560.5458676
NS_A1_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5516706, upper bound: 560.5457502
NS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5656584, upper bound: 560.5496828
NS_A1_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5429310, upper bound: 560.5433078
NS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5458676
NS_A1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5429310, upper bound: 560.5434727
NS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.55
Output dim: 0, lower bound: -560.5579893, upper bound: 560.5491019

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -104.7572327, 322.2603455, -123.8251495, 379.2772217, -484.0344543, 446.0855103
1: -148.1832581, 326.9640808, -175.4502869, 385.5563049, -533.7394409, 502.4143372
2: -125.2624207, 361.4502563, -148.4057007, 426.6238403, -551.8862305, 509.8559570
3: -133.6519165, 452.9406433, -158.2839508, 535.4919434, -669.1437988, 611.2246094
4: -112.1980591, 417.5491333, -133.2243042, 494.5540161, -606.7518921, 550.7734375

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725711, upper bound: 560.5726206
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725711, upper bound: 560.5736679
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -124.1436539, 383.8518982, -124.9378052, 382.4979248, -506.6415405, 508.7896423
1: -175.9054565, 388.2107544, -177.1563416, 388.8704224, -564.7758789, 565.3670044
2: -148.7681122, 428.5606079, -149.8736877, 430.2654419, -579.0335693, 578.4342041
3: -158.4910126, 538.3789062, -159.7629547, 539.8792725, -698.3703003, 698.1418457
4: -133.1133575, 495.2788696, -134.4683075, 498.4347839, -631.5480957, 629.7471313

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800299, upper bound: 560.5732984
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5796237, upper bound: 560.5733372
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -120.1034164, 370.2868652, -135.7411957, 409.6784363, -529.7818604, 506.0280151
1: -170.2850189, 375.7413635, -191.9468689, 416.6437073, -586.9287109, 567.6882324
2: -144.0552368, 415.5894775, -162.4503479, 460.4765930, -604.5318604, 578.0396118
3: -153.4859314, 522.0560303, -172.9194794, 576.2484131, -729.7343750, 694.9754639
4: -129.1982574, 481.4008484, -145.2245789, 531.6247559, -660.8229980, 626.6254272

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5748858, upper bound: 560.5778032
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5752732, upper bound: 560.5782015
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5752732, upper bound: 560.5782016
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -129.5180969, 396.4279480, -135.7411957, 409.6784363, -539.1965332, 532.1691284
1: -183.3757935, 401.7453308, -191.9468689, 416.6437073, -600.0194702, 593.6921997
2: -155.1768799, 443.3005371, -162.4503479, 460.4765930, -615.6534424, 605.7507935
3: -165.1611023, 556.0195312, -172.9194794, 576.2484131, -741.4093628, 728.9390259
4: -138.5506897, 511.5200806, -145.2245789, 531.6247559, -670.1754150, 656.7446289

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5752732, upper bound: 560.5778963
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5752732, upper bound: 560.5793983
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -117.1212387, 361.2501221, -126.1998749, 390.3705444, -507.4917603, 487.4499817
1: -166.2194977, 366.6256409, -179.0869598, 394.9167175, -561.1362305, 545.7125244
2: -140.5800323, 405.4136047, -151.3122864, 435.8156433, -576.3956909, 556.7258911
3: -149.8028107, 509.3803406, -161.3124237, 547.3679810, -697.1707764, 670.6926880
4: -126.0737228, 469.6578979, -135.3108063, 503.5669556, -629.6406250, 604.9686279

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5747523, upper bound: 560.5772410
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5735832, upper bound: 560.5768363
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -127.3094864, 389.7497253, -131.9980164, 404.0043640, -531.3137817, 521.7477417
1: -180.3452454, 395.0643005, -187.1138458, 409.5249329, -589.8701172, 582.1781616
2: -152.5758362, 435.8907776, -158.1888275, 451.8222046, -604.3979492, 594.0795288
3: -162.4415894, 546.6652222, -168.4966888, 566.5863037, -729.0278931, 715.1618652
4: -136.2126923, 502.8613892, -141.1893311, 521.4242554, -657.6369629, 644.0507202

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5797713, upper bound: 560.5790941
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799540, upper bound: 560.5791872
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -111.7308121, 339.1276855, -132.2907257, 400.8446045, -512.5753784, 471.4183960
1: -158.0385284, 344.8851624, -187.2316742, 407.5532837, -565.5917969, 532.1168213
2: -133.6682587, 381.2792664, -158.3485718, 450.2850342, -583.9532471, 539.6278076
3: -142.4163055, 476.8443909, -168.7184601, 563.5803833, -705.9967041, 645.5628662
4: -119.4174805, 439.9692688, -141.4077301, 519.7090454, -639.1265259, 581.3770142

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5784627
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5802112
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -129.9535065, 397.2473145, -130.1896667, 393.7119751, -523.6654663, 527.4368896
1: -184.1100769, 402.6030579, -184.2492371, 400.4779968, -584.5880737, 586.8522339
2: -155.7455902, 444.3345032, -155.8457031, 442.6605835, -598.4061890, 600.1801758
3: -165.7575226, 557.2296143, -166.0361176, 553.8888550, -719.6463623, 723.2657471
4: -139.0264435, 513.0037231, -139.1911011, 511.0309448, -650.0573730, 652.1948242

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5786494
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5804734
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -125.0242615, 381.8860168, -127.8180542, 392.1546021, -517.1787720, 509.7040710
1: -177.2667236, 388.5055847, -181.1683960, 397.3567200, -574.6234131, 569.6739502
2: -149.9831085, 429.7574768, -153.1545715, 438.7735901, -588.7567139, 582.9119873
3: -159.7153931, 538.8763428, -163.0880280, 550.1427612, -709.8581543, 701.9643555
4: -134.4605560, 497.5956116, -136.9023895, 506.7402954, -641.2008057, 634.4979858

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5731807, upper bound: 560.5781454
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5732613, upper bound: 560.5792785
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -125.6591492, 384.1333618, -128.7819366, 395.3598938, -521.0189819, 512.9152222
1: -178.2677155, 390.7551880, -182.6146698, 400.5396729, -578.8073730, 573.3698120
2: -150.7980042, 432.2301636, -154.3283844, 442.3897095, -593.1874390, 586.5585327
3: -160.6195984, 542.0338745, -164.4046478, 554.7092896, -715.3288574, 706.4384766
4: -135.1707153, 500.6216431, -137.9940186, 511.1076660, -646.2783813, 638.6156616

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725458, upper bound: 560.5757113
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5738832, upper bound: 560.5800970
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -128.3915253, 383.1229248, -118.9887772, 364.3974609, -492.7889404, 502.1116638
1: -181.7096252, 390.9989929, -167.7462006, 368.3889771, -550.0986328, 558.7451782
2: -153.8901215, 432.2569275, -141.6731873, 406.2779236, -560.1680298, 573.9298706
3: -163.5623779, 539.6156006, -151.2870636, 510.0222168, -673.5845337, 690.9026489
4: -137.4849701, 498.8919983, -126.5718002, 468.6476440, -606.1326294, 625.4638062

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5709624, upper bound: 560.5768620
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5709624, upper bound: 560.5787612
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -130.4816742, 392.1405945, -133.3635406, 406.7278442, -537.2095337, 525.5041504
1: -184.8601379, 399.6746521, -189.3709717, 412.6143494, -597.4744873, 589.0454102
2: -156.4749603, 441.8617859, -160.0991516, 455.4154053, -611.8903809, 601.9608154
3: -166.4022064, 552.0841064, -170.3750610, 570.4933472, -736.8955688, 722.4590454
4: -139.7466278, 509.7502747, -142.8167725, 525.5798950, -665.3264771, 652.5670166

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5712594, upper bound: 560.5775144
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5712594, upper bound: 560.5805949
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -131.3517914, 400.9027405, -136.3862762, 414.8444519, -546.1962280, 537.2890015
1: -186.4413300, 406.5832825, -193.4837952, 421.0783997, -607.5197144, 600.0670776
2: -157.5447388, 448.7116089, -163.5585632, 464.9804688, -622.5252075, 612.2701416
3: -167.7609863, 561.8778076, -174.0971069, 582.2504883, -750.0114746, 735.9749146
4: -140.5767822, 517.3845215, -145.9300842, 536.7589722, -677.3357544, 663.3145142

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5206145, upper bound: 560.5085493
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5206145, upper bound: 560.5740001
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -130.0877228, 394.9985352, -136.5563049, 415.5073242, -545.5949707, 531.5547485
1: -184.5915833, 401.0894165, -193.7770386, 421.6492310, -606.2407837, 594.8663940
2: -156.0140839, 442.9841614, -163.7769775, 465.5625305, -621.5765991, 606.7611084
3: -166.0739441, 554.3604736, -174.3424377, 583.0215454, -749.0954590, 728.7028809
4: -139.1658325, 511.0791931, -146.1041565, 537.4318848, -676.5976562, 657.1832275

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5221115, upper bound: 560.5088046
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5221115, upper bound: 560.5765469
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -128.0247498, 385.5955505, -134.4992371, 409.3531799, -537.3779297, 520.0947876
1: -181.1781006, 392.7850952, -190.8240356, 415.2601013, -596.4380493, 583.6091309
2: -153.2630463, 434.3192749, -161.2328644, 458.4798584, -611.7429199, 595.5521240
3: -163.1033936, 542.7240601, -171.6648102, 574.1014404, -737.2048340, 714.3888550
4: -136.8046417, 500.8822632, -143.8209076, 529.0704346, -665.8750610, 644.7031250

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5078240, upper bound: 560.5210572
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5078240, upper bound: 560.5774222
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -123.5631180, 377.8705444, -180.0895386, 548.6288452, -672.1919556, 557.9600830
1: -174.2431030, 382.1484070, -254.1669312, 555.7836914, -730.0267334, 636.3153076
2: -147.1537933, 421.6246643, -214.3512421, 614.4018555, -761.5556641, 635.9758301
3: -157.1236572, 529.1632080, -229.2562714, 767.7781372, -924.9017944, 758.4194946
4: -131.4708557, 486.5885620, -191.8708801, 708.0519409, -839.5226440, 678.4594116

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5562885, upper bound: 560.5291904
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5592168, upper bound: 560.5368259
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -123.5631180, 377.8705444, -191.2283478, 583.9119873, -707.4750977, 569.0988770
1: -174.2431030, 382.1484070, -270.9284363, 592.2199097, -766.4629517, 653.0768433
2: -147.1537933, 421.6246643, -228.8025360, 654.9713135, -802.1250610, 650.4271851
3: -157.1236572, 529.1632080, -244.1843719, 818.5407104, -975.6643677, 773.3475952
4: -131.4708557, 486.5885620, -204.8168182, 754.9985352, -886.4693604, 691.4052734

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5594799, upper bound: 560.5391695
time: 1.19 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601418, upper bound: 560.5379571
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -139.4024811, 424.5623169, -180.0895386, 548.6288452, -688.0313110, 604.6518555
1: -197.8152924, 430.6977539, -254.1669312, 555.7836914, -753.5988159, 684.8646240
2: -167.2440643, 475.5010071, -214.3512421, 614.4018555, -781.3737183, 689.8522339
3: -177.9433289, 595.6674805, -229.2562714, 767.7781372, -945.7213745, 824.9237671
4: -149.1945343, 549.0896606, -191.8708801, 708.0519409, -857.2464600, 740.9605713

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5571054, upper bound: 560.5435788
time: 1.83 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5551304, upper bound: 560.5427613
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -139.4024811, 424.5623169, -191.2283478, 583.9119873, -723.3144531, 615.7906494
1: -197.8152924, 430.6977539, -270.9284363, 592.2199097, -790.0350952, 701.6262207
2: -167.2440643, 475.5010071, -228.8025360, 654.9713135, -822.0021362, 704.3035278
3: -177.9433289, 595.6674805, -244.1843719, 818.5407104, -996.4839478, 839.8518677
4: -149.1945343, 549.0896606, -204.8168182, 754.9985352, -904.1930542, 753.9064941

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5176170, upper bound: 560.5337471
time: 0.94 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5583587, upper bound: 560.5477359
time: 0.83 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.53 seconds
NS_A1_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5725711, upper bound: 560.5726206
NS_A1_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5725711, upper bound: 560.5736679
NS_A1_B1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5800299, upper bound: 560.5732984
NS_A1_B1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5796237, upper bound: 560.5733372
NS_A1_B1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5752732, upper bound: 560.5782015
NS_A1_B1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5752732, upper bound: 560.5782016
NS_A1_B1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5752732, upper bound: 560.5778963
NS_A1_B1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5752732, upper bound: 560.5793983
NS_A1_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5747523, upper bound: 560.5772410
NS_A1_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5735832, upper bound: 560.5768363
NS_A1_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5797713, upper bound: 560.5790941
NS_A1_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5799540, upper bound: 560.5791872
NS_A1_B1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5784627
NS_A1_B1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5802112
NS_A1_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5786494
NS_A1_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5804734
NS_A1_B1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5731807, upper bound: 560.5781454
NS_A1_B1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5732613, upper bound: 560.5792785
NS_A1_B1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5725458, upper bound: 560.5757113
NS_A1_B1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5738832, upper bound: 560.5800970
NS_A1_B1_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5709624, upper bound: 560.5768620
NS_A1_B1_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5709624, upper bound: 560.5787612
NS_A1_B1_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5712594, upper bound: 560.5775144
NS_A1_B1_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5712594, upper bound: 560.5805949
NS_A1_B1_A2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5206145, upper bound: 560.5085493
NS_A1_B1_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5206145, upper bound: 560.5740001
NS_A1_B1_A2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5221115, upper bound: 560.5088046
NS_A1_B1_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5221115, upper bound: 560.5765469
NS_A1_B1_A2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5078240, upper bound: 560.5210572
NS_A1_B1_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5078240, upper bound: 560.5774222
NS_A1_B2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5562885, upper bound: 560.5291904
NS_A1_B2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5592168, upper bound: 560.5368259
NS_A1_B2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5594799, upper bound: 560.5391695
NS_A1_B2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5601418, upper bound: 560.5379571
NS_A1_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5571054, upper bound: 560.5435788
NS_A1_B2_B2_A2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5551304, upper bound: 560.5427613
NS_A1_B2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5176170, upper bound: 560.5337471
NS_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -560.5583587, upper bound: 560.5477359

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -98.6223831, 302.8237915, -123.8251495, 379.2772217, -477.8995972, 426.6489258
1: -139.7735291, 308.2255859, -175.4502869, 385.5563049, -525.3297729, 483.6758423
2: -118.1788864, 341.3947449, -148.4057007, 426.6238403, -544.8026733, 489.8004456
3: -126.0942154, 428.0199280, -158.2839508, 535.4919434, -661.5861206, 586.3038940
4: -106.1004639, 395.5417786, -133.2243042, 494.5540161, -600.6542358, 528.7661133

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5722303
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5726206
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -108.2937851, 328.9898071, -123.8251495, 379.2772217, -487.5710144, 452.8149109
1: -153.0844879, 334.4976501, -175.4502869, 385.5563049, -538.6408081, 509.9479370
2: -129.5347595, 369.7617493, -148.4057007, 426.6238403, -556.1585083, 518.1674805
3: -137.9855347, 462.3987732, -158.2839508, 535.4919434, -673.4774780, 620.6827393
4: -115.6810455, 426.2505798, -133.2243042, 494.5540161, -610.2349854, 559.4748535

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5722303
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5736679
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -120.8804016, 373.0783386, -122.7080536, 375.4029541, -496.2833252, 495.7863770
1: -171.0998688, 377.4435730, -173.8901978, 381.7728882, -552.8727417, 551.3337402
2: -144.7540894, 416.6851501, -147.1342163, 422.4432983, -567.1973877, 563.8193359
3: -154.1907959, 523.2279663, -156.8321838, 529.9494019, -684.1401978, 680.0601807
4: -129.5472412, 481.3460388, -132.0435638, 489.2904053, -618.8376465, 613.3895874

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799243, upper bound: 560.5732984
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799243, upper bound: 560.5732984
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -120.5847168, 372.2359924, -122.8811874, 375.9956055, -496.5802917, 495.1171875
1: -170.7796936, 376.6588135, -174.1894989, 382.3259277, -553.1055908, 550.8483276
2: -144.4543610, 415.9046326, -147.3572388, 423.0785522, -567.5328979, 563.2618408
3: -153.8644562, 522.2062988, -157.0859222, 530.7586670, -684.6231079, 679.2922363
4: -129.2539062, 480.4755859, -132.2209930, 490.0051880, -619.2590332, 612.6965942

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792552, upper bound: 560.5733372
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5781099, upper bound: 560.5733372
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -122.5522003, 375.9010925, -135.7411957, 409.6784363, -532.2306519, 511.6422729
1: -173.7525177, 382.0307617, -191.9468689, 416.6437073, -590.3961792, 573.9776611
2: -147.0411987, 422.7162476, -162.4503479, 460.4765930, -607.5178223, 585.1665649
3: -156.6993103, 530.6627197, -172.9194794, 576.2484131, -732.9476929, 703.5822144
4: -131.9856720, 489.7915649, -145.2245789, 531.6247559, -663.6104126, 635.0161133

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5730099, upper bound: 560.5773489
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5741860, upper bound: 560.5777944
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -120.3175812, 370.8423157, -135.7411957, 409.6784363, -529.9959106, 506.5834961
1: -170.6487732, 376.3403931, -191.9468689, 416.6437073, -587.2924194, 568.2872314
2: -144.3515778, 416.2481995, -162.4503479, 460.4765930, -604.8281860, 578.6983032
3: -153.7984314, 522.8901978, -172.9194794, 576.2484131, -730.0468140, 695.8096924
4: -129.4478607, 482.1914673, -145.2245789, 531.6247559, -661.0726318, 627.4160156

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5753999, upper bound: 560.5771358
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5753999, upper bound: 560.5782015
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -131.6853333, 397.0609436, -135.7411957, 409.6784363, -541.3637695, 532.8021240
1: -186.1695099, 404.0013428, -191.9468689, 416.6437073, -602.8131714, 595.9482422
2: -157.6702118, 446.6024780, -162.4503479, 460.4765930, -618.1467896, 609.0526123
3: -167.7175446, 558.7875366, -172.9194794, 576.2484131, -743.9658813, 731.7070312
4: -140.9789429, 515.5856934, -145.2245789, 531.6247559, -672.6036987, 660.8103027

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5723664
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5784484, upper bound: 560.5777717
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -130.1263123, 398.1080017, -135.7411957, 409.6784363, -539.8046265, 533.8491821
1: -184.2276611, 403.5349121, -191.9468689, 416.6437073, -600.8713379, 595.4816895
2: -155.9013519, 445.3507690, -162.4503479, 460.4765930, -616.3779297, 607.8010864
3: -165.9458466, 558.4691772, -172.9194794, 576.2484131, -742.1942749, 731.3886108
4: -139.1948853, 513.8369141, -145.2245789, 531.6247559, -670.8195190, 659.0615234

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777777, upper bound: 560.5786409
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777491, upper bound: 560.5788016
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -117.0398636, 360.9856567, -126.1998749, 390.3705444, -507.4104004, 487.1855469
1: -166.1050110, 366.3626099, -179.0869598, 394.9167175, -561.0217285, 545.4495850
2: -140.4837799, 405.1240540, -151.3122864, 435.8156433, -576.2994385, 556.4363403
3: -149.6995239, 509.0138245, -161.3124237, 547.3679810, -697.0675049, 670.3262329
4: -125.9877701, 469.3226624, -135.3108063, 503.5669556, -629.5546875, 604.6333008

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5722585, upper bound: 560.5769022
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728966, upper bound: 560.5769726
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -116.9688873, 360.7361755, -126.1998749, 390.3705444, -507.3394165, 486.9360046
1: -166.0164032, 366.1260986, -179.0869598, 394.9167175, -560.9331055, 545.2130737
2: -140.4118805, 404.8716431, -151.3122864, 435.8156433, -576.2275391, 556.1839600
3: -149.6203613, 508.6994324, -161.3124237, 547.3679810, -696.9883423, 670.0118408
4: -125.9255676, 469.0500488, -135.3108063, 503.5669556, -629.4924927, 604.3608398

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5702853, upper bound: 560.5763217
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5727558, upper bound: 560.5766552
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -125.3151016, 383.3699951, -127.8669586, 390.2107239, -515.5258179, 511.2369385
1: -177.4467773, 388.6565857, -181.1293030, 395.7804871, -573.2272949, 569.7857666
2: -150.1492920, 428.8045349, -153.1841583, 436.6302185, -586.7795410, 581.9886475
3: -159.8320770, 537.6771851, -163.0982819, 547.1951294, -707.0272217, 700.7754517
4: -134.0424194, 494.5388794, -136.7225037, 503.5697937, -637.6121826, 631.2612915

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5797713, upper bound: 560.5790941
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5797713, upper bound: 560.5790941
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -125.2029877, 382.8680420, -128.2431183, 391.7998352, -517.0025635, 511.1111450
1: -177.2924957, 388.2133789, -181.6873322, 397.3684998, -574.6610107, 569.9006348
2: -150.0127411, 428.4068909, -153.6097412, 438.5272827, -588.5400391, 582.0166016
3: -159.6968536, 537.0909424, -163.6177216, 549.5948486, -709.2916870, 700.7086792
4: -133.9242859, 494.0805664, -137.1082153, 505.8510132, -639.7752686, 631.1887207

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5798330, upper bound: 560.5791872
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794321, upper bound: 560.5791718
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794321, upper bound: 560.5791718
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -111.7308121, 339.1276855, -119.1938095, 358.0650940, -469.7958984, 458.3215027
1: -158.0385284, 344.8851624, -168.5064545, 365.1320190, -523.1704712, 513.3915405
2: -133.6682587, 381.2792664, -142.6656799, 403.8044128, -537.4726562, 523.9449463
3: -142.4163055, 476.8443909, -151.7751007, 504.1977234, -646.6140137, 628.6195068
4: -119.4174805, 439.9692688, -127.4080200, 465.4279480, -584.8454590, 567.3773193

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5784627
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5784627
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -111.7308121, 339.1276855, -128.6475677, 388.6757812, -500.4065857, 467.7752380
1: -158.0385284, 344.8851624, -182.0438690, 395.4440613, -553.4825439, 526.9290161
2: -133.6682587, 381.2792664, -153.9807739, 437.1323853, -570.8005981, 535.2600098
3: -142.4163055, 476.8443909, -164.0511475, 546.8892822, -689.3056030, 640.8955078
4: -119.4174805, 439.9692688, -137.5259552, 504.6318054, -624.0493164, 577.4952393

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5723564, upper bound: 560.5774360
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5803338, upper bound: 560.5784627
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -129.9535065, 397.2473145, -119.1938095, 358.0650940, -488.0186157, 516.4410400
1: -184.1100769, 402.6030579, -168.5064545, 365.1320190, -549.2420654, 571.1094360
2: -155.7455902, 444.3345032, -142.6656799, 403.8044128, -559.5499878, 587.0001831
3: -165.7575226, 557.2296143, -151.7751007, 504.1977234, -669.9552002, 709.0046997
4: -139.0264435, 513.0037231, -127.4080200, 465.4279480, -604.4544067, 640.4117432

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5786494
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5786494
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -129.9535065, 397.2473145, -128.8227844, 389.2803345, -519.2338257, 526.0700684
1: -184.1100769, 402.6030579, -182.2952118, 396.0438232, -580.1538696, 584.8982544
2: -155.7455902, 444.3345032, -154.1933441, 437.7926331, -593.5382080, 598.5278320
3: -165.7575226, 557.2296143, -164.2790527, 547.7274780, -713.4849854, 721.5086060
4: -139.0264435, 513.0037231, -137.7171936, 505.3930054, -644.4193726, 650.7208862

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793975, upper bound: 560.5795212
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788423, upper bound: 560.5798018
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -119.9195709, 366.2033691, -112.3569946, 342.2013855, -462.1209106, 478.5603638
1: -169.8238831, 372.7666016, -159.0167236, 347.8697815, -517.6936035, 531.7833252
2: -143.6471558, 412.5382385, -134.3604431, 384.7108765, -528.3580322, 546.8986816
3: -153.1659393, 517.2355347, -143.3200684, 481.2460022, -634.4119263, 660.5556030
4: -128.9179382, 477.8540649, -120.2526093, 444.3110962, -573.2290039, 598.1065063

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5731807, upper bound: 560.5781454
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5731807, upper bound: 560.5781454
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -121.6785889, 371.1994629, -124.7545929, 382.5867310, -504.2652283, 495.9540405
1: -172.5568848, 377.8683472, -176.8324890, 387.6929932, -560.2498779, 554.7008057
2: -146.0069733, 418.0887146, -149.4713440, 428.1065063, -574.1133423, 567.5600586
3: -155.4829712, 524.0260620, -159.1892090, 536.6682739, -692.1511841, 683.2152710
4: -130.9170074, 483.9896851, -133.6077576, 494.3518066, -625.2686768, 617.5974121

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5731807, upper bound: 560.5792785
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5732613, upper bound: 560.5792785
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -120.1427917, 367.3106995, -113.3390427, 345.8093567, -465.9521484, 480.6497192
1: -170.2104187, 373.8418274, -160.4661560, 351.4540710, -521.6644897, 534.3078003
2: -143.9357452, 413.7109375, -135.5377655, 388.6962891, -532.6319580, 549.2485352
3: -153.5148163, 518.8136597, -144.6743011, 486.4562073, -639.9710083, 663.4879761
4: -129.1759033, 479.3582153, -121.3654938, 449.1899109, -578.3658447, 600.7236328

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725458, upper bound: 560.5757113
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725458, upper bound: 560.5757113
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -122.6620560, 374.4150696, -125.6413879, 385.3401794, -508.0022278, 500.0564575
1: -174.0574493, 381.1093445, -178.1777802, 390.5015869, -564.5590210, 559.2870483
2: -147.2472534, 421.6554871, -150.5828857, 431.3514404, -578.5986938, 572.2384033
3: -156.8245392, 528.5427856, -160.4143982, 540.6959229, -697.5203247, 688.9571533
4: -131.9986725, 488.2976379, -134.6476746, 498.2774048, -630.2758789, 622.9452515

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5733372, upper bound: 560.5795368
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725458, upper bound: 560.5799749
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -114.9710617, 345.1865234, -118.9887772, 364.3974609, -479.3685303, 464.1752625
1: -161.7427063, 350.6438904, -167.7462006, 368.3889771, -530.1317139, 518.3900757
2: -136.8078613, 387.2987366, -141.6731873, 406.2779236, -543.0858154, 528.9717407
3: -145.8375702, 484.5830383, -151.2870636, 510.0222168, -655.8597412, 635.8700562
4: -122.3209152, 446.8339844, -126.5718002, 468.6476440, -590.9685669, 573.4056396

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5622514, upper bound: 560.5676495
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5634827, upper bound: 560.5706596
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -129.0332336, 387.8454895, -118.9887772, 364.3974609, -493.4306946, 506.8342285
1: -182.9128571, 395.3689880, -167.7462006, 368.3889771, -551.3017578, 563.1151733
2: -154.8321991, 437.1446533, -141.6731873, 406.2779236, -561.1101074, 578.8177490
3: -164.6565552, 546.0996704, -151.2870636, 510.0222168, -674.6787720, 697.3866577
4: -138.2821045, 504.2161560, -126.5718002, 468.6476440, -606.9297485, 630.7879639

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5622514, upper bound: 560.5687905
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5634827, upper bound: 560.5723286
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -114.6262054, 344.1818848, -133.3635406, 406.7278442, -521.3540039, 477.5454102
1: -161.2264404, 349.6030273, -189.3709717, 412.6143494, -573.8408203, 538.9738770
2: -136.3665161, 386.1528015, -160.0991516, 455.4154053, -591.7819214, 546.2519531
3: -145.3856354, 483.1524658, -170.3750610, 570.4933472, -715.8788452, 653.5274048
4: -121.9517899, 445.4833374, -142.8167725, 525.5798950, -647.5316772, 588.3000488

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5603503, upper bound: 560.5695785
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5634827, upper bound: 560.5707289
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -129.2583160, 388.9176025, -133.3635406, 406.7278442, -535.9861450, 522.2811279
1: -183.2346954, 396.3386536, -189.3709717, 412.6143494, -595.8490601, 585.7095947
2: -155.1126709, 438.1524048, -160.0991516, 455.4154053, -610.5280762, 598.2515869
3: -164.9413910, 547.4499512, -170.3750610, 570.4933472, -735.4346313, 717.8248901
4: -138.5189056, 505.3483276, -142.8167725, 525.5798950, -664.0987549, 648.1650391

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5701113, upper bound: 560.5798739
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5701113, upper bound: 560.5805949
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -131.3517914, 400.9027405, -135.4415436, 412.1016541, -543.4534302, 536.3442993
1: -186.4413300, 406.5832825, -192.2095032, 418.1757812, -604.6170654, 598.7927246
2: -157.5447388, 448.7116089, -162.4313049, 461.7128601, -619.2575073, 611.1429443
3: -167.7609863, 561.8778076, -172.9238586, 578.1752930, -745.9362183, 734.8016357
4: -140.5767822, 517.3845215, -144.8974915, 532.9357910, -673.5125732, 662.2818604

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5206145, upper bound: 560.5729733
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5206145, upper bound: 560.5740001
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -130.0877228, 394.9985352, -135.4415436, 412.1016541, -542.1893921, 530.4400635
1: -184.5915833, 401.0894165, -192.2095032, 418.1757812, -602.7672729, 593.2988281
2: -156.0140839, 442.9841614, -162.4313049, 461.7128601, -617.7268066, 605.4154663
3: -166.0739441, 554.3604736, -172.9238586, 578.1752930, -744.2491455, 727.2843018
4: -139.1658325, 511.0791931, -144.8974915, 532.9357910, -672.1016235, 655.9766235

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5221115, upper bound: 560.5744037
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5221115, upper bound: 560.5744037
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -127.0607758, 382.6828918, -134.4992371, 409.3531799, -536.4139404, 517.1821289
1: -179.8215637, 389.7965698, -190.8240356, 415.2601013, -595.0816040, 580.6206055
2: -152.0997467, 431.0142212, -161.2328644, 458.4798584, -610.5794678, 592.2470703
3: -161.8722382, 538.5751953, -171.6648102, 574.1014404, -735.9736938, 710.2399902
4: -135.7593384, 497.0256958, -143.8209076, 529.0704346, -664.8297729, 640.8464966

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5002678, upper bound: 560.5752678
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5002678, upper bound: 560.5774222
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -122.9133987, 376.0119019, -178.3999634, 543.4202271, -666.3334961, 554.4117432
1: -173.3034058, 380.2568054, -251.7177277, 550.5338745, -723.8372803, 631.9745483
2: -146.3646851, 419.5657043, -212.2993317, 608.6469727, -755.0116577, 631.8650513
3: -156.2866211, 526.5829468, -227.0549011, 760.5365601, -916.8231201, 753.6378174
4: -130.7810211, 484.1881409, -190.0547638, 701.3375244, -832.1185303, 674.2429199

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521821, upper bound: 560.5291904
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5521821, upper bound: 560.5291904
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -121.2987213, 370.1004333, -177.8211517, 540.8812256, -662.1799316, 547.9215088
1: -170.9802399, 374.5004578, -250.8834229, 548.1641846, -719.1444092, 625.3839111
2: -144.4260712, 413.2980652, -211.6369324, 606.2115479, -750.6376343, 624.9349365
3: -154.1816711, 518.4381714, -226.3085022, 757.3580933, -911.5396729, 744.7466431
4: -129.0343475, 476.8900757, -189.4899445, 698.7535400, -827.7877197, 666.3800049

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5555145, upper bound: 560.5368259
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5555145, upper bound: 560.5368259
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -121.9825134, 373.3664856, -190.6013489, 581.9680786, -703.9505615, 563.9678345
1: -171.9560394, 377.5604858, -270.0174866, 590.2583618, -762.2143555, 647.5780029
2: -145.2335205, 416.6315002, -228.0382690, 652.8263550, -798.0598145, 644.6697998
3: -155.0872650, 522.9097900, -243.3657837, 815.8471069, -970.9343872, 766.2755127
4: -129.7929230, 480.7670288, -204.1418762, 752.5094604, -882.3023682, 684.9089355

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560725, upper bound: 560.5375299
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560725, upper bound: 560.5391695
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -122.2061234, 372.3715820, -188.8578033, 576.3665771, -698.5726929, 561.2293701
1: -172.2659760, 376.8753662, -267.5223999, 584.7128296, -756.9788208, 644.3977661
2: -145.5442657, 416.0659485, -225.9501801, 646.7679443, -792.3121948, 642.0161133
3: -155.3141479, 521.8527832, -241.1363373, 808.1394653, -963.4535522, 762.9891357
4: -130.0316772, 480.2986145, -202.3009491, 745.4860229, -875.5176392, 682.5995483

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5591957, upper bound: 560.5367163
time: 0.95 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5591957, upper bound: 560.5379571
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -139.3185577, 424.2808838, -180.0895386, 548.6288452, -687.9473877, 604.3703613
1: -197.6965179, 430.4213562, -254.1669312, 555.7836914, -753.4801636, 684.5881348
2: -167.1442261, 475.1976929, -214.3512421, 614.4018555, -781.2739868, 689.5487061
3: -177.8364563, 595.2813110, -229.2562714, 767.7781372, -945.6145630, 824.5375977
4: -149.1055450, 548.7395630, -191.8708801, 708.0519409, -857.1574707, 740.6104736

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5551920, upper bound: 560.5430732
time: 1.02 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5551920, upper bound: 560.5435788
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -139.1152649, 423.6483765, -190.3231659, 580.9854126, -720.1006470, 613.9715576
1: -197.4081421, 429.7830811, -269.6434021, 589.2766724, -786.6846313, 699.4265137
2: -166.9009399, 474.4868469, -227.7259827, 651.6981812, -818.3883667, 702.2128296
3: -177.5759430, 594.3860474, -243.0176849, 814.4506836, -992.0265503, 837.4036865
4: -148.8858185, 547.9083252, -203.8531342, 751.2164917, -900.1022949, 751.7614746

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5528784, upper bound: 560.5471527
time: 0.86 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5528784, upper bound: 560.5477359
time: 0.98 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.67 seconds
NS_A1_B1_A1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5722303
NS_A1_B1_A1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5726206
NS_A1_B1_A1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5722303
NS_A1_B1_A1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5736679
NS_A1_B1_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5799243, upper bound: 560.5732984
NS_A1_B1_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5799243, upper bound: 560.5732984
NS_A1_B1_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5792552, upper bound: 560.5733372
NS_A1_B1_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5781099, upper bound: 560.5733372
NS_A1_B1_A1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5730099, upper bound: 560.5773489
NS_A1_B1_A1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5741860, upper bound: 560.5777944
NS_A1_B1_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5753999, upper bound: 560.5771358
NS_A1_B1_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5753999, upper bound: 560.5782015
NS_A1_B1_A1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5684704, upper bound: 560.5723664
NS_A1_B1_A1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5784484, upper bound: 560.5777717
NS_A1_B1_A1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5777777, upper bound: 560.5786409
NS_A1_B1_A1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5777491, upper bound: 560.5788016
NS_A1_B1_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5722585, upper bound: 560.5769022
NS_A1_B1_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5728966, upper bound: 560.5769726
NS_A1_B1_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5702853, upper bound: 560.5763217
NS_A1_B1_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5727558, upper bound: 560.5766552
NS_A1_B1_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5797713, upper bound: 560.5790941
NS_A1_B1_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5797713, upper bound: 560.5790941
NS_A1_B1_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5794321, upper bound: 560.5791718
NS_A1_B1_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5794321, upper bound: 560.5791718
NS_A1_B1_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5784627
NS_A1_B1_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5784627
NS_A1_B1_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5723564, upper bound: 560.5774360
NS_A1_B1_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5803338, upper bound: 560.5784627
NS_A1_B1_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5786494
NS_A1_B1_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5807811, upper bound: 560.5786494
NS_A1_B1_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5793975, upper bound: 560.5795212
NS_A1_B1_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5788423, upper bound: 560.5798018
NS_A1_B1_A2_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5731807, upper bound: 560.5781454
NS_A1_B1_A2_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5731807, upper bound: 560.5781454
NS_A1_B1_A2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5731807, upper bound: 560.5792785
NS_A1_B1_A2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5732613, upper bound: 560.5792785
NS_A1_B1_A2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5725458, upper bound: 560.5757113
NS_A1_B1_A2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5725458, upper bound: 560.5757113
NS_A1_B1_A2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5733372, upper bound: 560.5795368
NS_A1_B1_A2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5725458, upper bound: 560.5799749
NS_A1_B1_A2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5622514, upper bound: 560.5676495
NS_A1_B1_A2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5634827, upper bound: 560.5706596
NS_A1_B1_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5622514, upper bound: 560.5687905
NS_A1_B1_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5634827, upper bound: 560.5723286
NS_A1_B1_A2_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5603503, upper bound: 560.5695785
NS_A1_B1_A2_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5634827, upper bound: 560.5707289
NS_A1_B1_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5701113, upper bound: 560.5798739
NS_A1_B1_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5701113, upper bound: 560.5805949
NS_A1_B1_A2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5206145, upper bound: 560.5729733
NS_A1_B1_A2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5206145, upper bound: 560.5740001
NS_A1_B1_A2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5221115, upper bound: 560.5744037
NS_A1_B1_A2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5221115, upper bound: 560.5744037
NS_A1_B1_A2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5002678, upper bound: 560.5752678
NS_A1_B1_A2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5002678, upper bound: 560.5774222
NS_A1_B2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5521821, upper bound: 560.5291904
NS_A1_B2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5521821, upper bound: 560.5291904
NS_A1_B2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5555145, upper bound: 560.5368259
NS_A1_B2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5555145, upper bound: 560.5368259
NS_A1_B2_B2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5560725, upper bound: 560.5375299
NS_A1_B2_B2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5560725, upper bound: 560.5391695
NS_A1_B2_B2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5591957, upper bound: 560.5367163
NS_A1_B2_B2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5591957, upper bound: 560.5379571
NS_A1_B2_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5551920, upper bound: 560.5430732
NS_A1_B2_B2_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5551920, upper bound: 560.5435788
NS_A1_B2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5528784, upper bound: 560.5471527
NS_A1_B2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 0, lower bound: -560.5528784, upper bound: 560.5477359

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -99.4085693, 302.6986389, -123.8251495, 379.2772217, -478.6857300, 426.5237427
1: -140.9528198, 308.7158813, -175.4502869, 385.5563049, -526.5091553, 484.1661377
2: -119.4175797, 341.9114685, -148.4057007, 426.6238403, -546.0414429, 490.3171692
3: -127.0970230, 428.3447571, -158.2839508, 535.4919434, -662.5889893, 586.6287231
4: -107.2240524, 395.9542236, -133.2243042, 494.5540161, -601.7778320, 529.1785278

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5717651, upper bound: 560.5717642
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5688093, upper bound: 560.5709699
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -99.7631454, 306.2926941, -123.8251495, 379.2772217, -479.0403748, 430.1177979
1: -141.4544067, 311.7534790, -175.4502869, 385.5563049, -527.0106812, 487.2037659
2: -119.6067886, 345.3048706, -148.4057007, 426.6238403, -546.2305908, 493.7105713
3: -127.5914078, 432.9915161, -158.2839508, 535.4919434, -663.0833740, 591.2754517
4: -107.3646774, 400.1542664, -133.2243042, 494.5540161, -601.9184570, 533.3785400

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5717651, upper bound: 560.5718178
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5688093, upper bound: 560.5712075
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -109.4187851, 325.6903687, -123.8251495, 379.2772217, -488.6960144, 449.5155029
1: -154.5349426, 332.8411255, -175.4502869, 385.5563049, -540.0912476, 508.2913818
2: -131.0408936, 368.3010559, -148.4057007, 426.6238403, -557.6647339, 516.7067261
3: -139.1284943, 459.2321167, -158.2839508, 535.4919434, -674.6204224, 617.5160522
4: -117.1081772, 424.5037537, -133.2243042, 494.5540161, -611.6619873, 557.7280273

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5679047, upper bound: 560.5713988
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683508, upper bound: 560.5722303
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683508, upper bound: 560.5722303
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -109.0713272, 331.1569519, -123.8251495, 379.2772217, -488.3485107, 454.9820862
1: -154.1831207, 336.7716064, -175.4502869, 385.5563049, -539.7394409, 512.2219238
2: -130.4927063, 372.3566589, -148.4057007, 426.6238403, -557.1165771, 520.7623291
3: -138.9854736, 465.5450134, -158.2839508, 535.4919434, -674.4774170, 623.8289795
4: -116.5205078, 429.1947327, -133.2243042, 494.5540161, -611.0743408, 562.4190674

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683508, upper bound: 560.5736679
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683508, upper bound: 560.5736679
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -120.8804016, 373.0783386, -118.2849274, 362.1468811, -483.0272217, 491.3632507
1: -171.0998688, 377.4435730, -167.6328583, 368.2695618, -539.3694458, 545.0762329
2: -144.7540894, 416.6851501, -141.8864288, 407.5787964, -552.3328857, 558.5715942
3: -154.1907959, 523.2279663, -151.1903076, 511.4603882, -665.6511230, 674.4182739
4: -129.5472412, 481.3460388, -127.3892670, 472.1991577, -601.7463989, 608.7352905

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728289, upper bound: 560.5724051
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728289, upper bound: 560.5732984
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -120.8804016, 373.0783386, -121.6785889, 371.1994629, -492.0798035, 494.7568359
1: -171.0998688, 377.4435730, -172.5568848, 377.8683472, -548.9682007, 550.0004883
2: -144.7540894, 416.6851501, -146.0069733, 418.0887146, -562.8427734, 562.6920166
3: -154.1907959, 523.2279663, -155.4829712, 524.0260620, -678.2168579, 678.7109375
4: -129.5472412, 481.3460388, -130.9170074, 483.9896851, -613.5369263, 612.2629395

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728289, upper bound: 560.5724051
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728289, upper bound: 560.5724051
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -120.5847168, 372.2359924, -118.8608017, 363.8561401, -484.4407959, 491.0968018
1: -170.7796936, 376.6588135, -168.5130463, 369.9801331, -540.7598267, 545.1718750
2: -144.4543610, 415.9046326, -142.6161957, 409.4685059, -553.9228516, 558.5206909
3: -153.8644562, 522.2062988, -151.9530029, 513.8200684, -667.6845093, 674.1593018
4: -129.2539062, 480.4755859, -127.9972839, 474.3628845, -603.6168213, 608.4729004

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5769399, upper bound: 560.5732324
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5769399, upper bound: 560.5733372
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -120.5847168, 372.2359924, -122.6620560, 374.4150696, -494.9997253, 494.8980408
1: -170.7796936, 376.6588135, -174.0574493, 381.1093445, -551.8889160, 550.7162476
2: -144.4543610, 415.9046326, -147.2472534, 421.6554871, -566.1098633, 563.1518555
3: -153.8644562, 522.2062988, -156.8245392, 528.5427856, -682.4072266, 679.0308228
4: -129.2539062, 480.4755859, -131.9986725, 488.2976379, -617.5515137, 612.4742432

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5769399, upper bound: 560.5732324
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5769399, upper bound: 560.5733372
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -99.6262665, 303.3445435, -131.2355499, 395.3599548, -494.9862061, 434.5800781
1: -141.2650452, 309.3683777, -185.4496613, 402.3600159, -543.6250610, 494.8180542
2: -119.6821518, 342.6448059, -156.9603271, 444.8258667, -564.5079956, 499.6051331
3: -127.3772964, 429.2673950, -167.1299133, 556.5081177, -683.8852539, 596.3971558
4: -107.4619675, 396.8184814, -140.3726349, 513.7339478, -621.1959229, 537.1911011

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5722295, upper bound: 560.5745454
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5688093, upper bound: 560.5737273
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -120.0360336, 367.5689087, -132.4397278, 399.0668640, -519.1028442, 500.0086365
1: -170.2012939, 373.7297363, -187.3000793, 406.1049500, -576.3060913, 561.0297852
2: -144.0362396, 413.5958862, -158.5325470, 448.9275818, -592.9636841, 572.1283569
3: -153.4891968, 519.0938110, -168.7416840, 561.5283813, -715.0175781, 687.8354492
4: -129.2781982, 479.2603149, -141.7186890, 518.1704102, -647.4485474, 620.9790039

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5754846, upper bound: 560.5773466
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5754846, upper bound: 560.5790257
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -120.3175812, 370.8423157, -131.9770660, 398.0668640, -518.3844604, 502.8193665
1: -170.6487732, 376.3403931, -186.6004181, 404.9904175, -575.6390991, 562.9407959
2: -144.3515778, 416.2481995, -158.0350342, 447.6813965, -592.0329590, 574.2832031
3: -153.7984314, 522.8901978, -168.1001434, 560.1605225, -713.9589844, 690.9903564
4: -129.4478607, 482.1914673, -141.2967529, 516.8365479, -646.2844238, 623.4882202

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5741046, upper bound: 560.5766286
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5733226, upper bound: 560.5736972
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -120.3175812, 370.8423157, -134.2155762, 402.2713013, -522.5886841, 505.0578918
1: -170.6487732, 376.3403931, -189.8871460, 410.0367737, -580.6855469, 566.2275391
2: -144.3515778, 416.2481995, -160.7288055, 453.3092651, -597.6608276, 576.9768677
3: -153.7984314, 522.8901978, -170.9192810, 566.4020996, -720.2005005, 693.8094482
4: -129.4478607, 482.1914673, -143.5718384, 523.2652588, -652.7131348, 625.7633057

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5741046, upper bound: 560.5781030
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5733226, upper bound: 560.5753254
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -109.1940842, 324.9040527, -131.2355499, 395.3599548, -504.5540466, 456.1395874
1: -154.2060547, 332.0718689, -185.4496613, 402.3600159, -556.5660400, 517.5214233
2: -130.7622986, 367.4578247, -156.9603271, 444.8258667, -575.5881348, 524.4180908
3: -138.8350067, 458.1616211, -167.1299133, 556.5081177, -695.3431396, 625.2915039
4: -116.8646240, 423.5288086, -140.3726349, 513.7339478, -630.5985107, 563.9013672

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5679047, upper bound: 560.5714424
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683508, upper bound: 560.5723664
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683508, upper bound: 560.5723664
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -129.1638794, 388.7749023, -132.4397278, 399.0668640, -528.2307129, 521.2145386
1: -182.6036224, 395.7469177, -187.3000793, 406.1049500, -588.7084961, 583.0469971
2: -154.6542358, 437.5384216, -158.5325470, 448.9275818, -603.5816040, 596.0709229
3: -164.4934235, 547.2797852, -168.7416840, 561.5283813, -726.0217896, 716.0214844
4: -138.2512360, 505.0944519, -141.7186890, 518.1704102, -656.4215698, 646.8131104

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5769734, upper bound: 560.5764385
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777491, upper bound: 560.5772197
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -126.9126892, 387.2245789, -133.7818604, 403.4608459, -530.3735352, 521.0064697
1: -179.5383453, 392.6753540, -189.0951233, 410.3890991, -589.9274292, 581.7705078
2: -151.9948578, 433.3672485, -160.0552521, 453.5818176, -605.5766602, 593.4224854
3: -161.7007294, 543.1607056, -170.3536835, 567.5003052, -729.2010498, 713.5144043
4: -135.6943665, 499.8034973, -143.0878754, 523.5402222, -659.2345581, 642.8913574

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807978, upper bound: 560.5779162
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807978, upper bound: 560.5785196
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -126.5283279, 386.3662720, -133.6807556, 402.9498596, -529.4782104, 520.0468750
1: -179.0304413, 391.8453674, -188.9752197, 409.9585876, -588.9890137, 580.8205566
2: -151.5244446, 432.5765076, -159.9429626, 453.1469116, -604.6713867, 592.5193481
3: -161.2650909, 542.1322632, -170.2390137, 566.9044800, -728.1695557, 712.3712769
4: -135.2827759, 498.8507996, -142.9900208, 523.0442505, -658.3270264, 641.8408203

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5791267, upper bound: 560.5779383
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5803237, upper bound: 560.5787034
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -96.2877121, 295.7979431, -121.0678635, 374.2596436, -470.5473328, 416.8658142
1: -136.6418304, 301.1189575, -171.5790863, 378.8587036, -515.5004272, 472.6980591
2: -115.5163803, 333.4515381, -144.9213104, 418.2716064, -533.7879639, 478.3728638
3: -123.2514038, 418.1851501, -154.6978455, 525.1723022, -648.4237061, 572.8829956
4: -103.6779709, 386.4305725, -129.7096252, 483.4325867, -587.1105347, 516.1401978

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5721817, upper bound: 560.5769022
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5721817, upper bound: 560.5769022
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -114.6248932, 353.1755066, -122.5349503, 378.8967285, -493.5216064, 475.7104492
1: -162.7025604, 358.5023804, -173.9224701, 383.3767700, -546.0792847, 532.4248047
2: -137.6036835, 396.4537354, -146.9243011, 423.0957947, -560.6994629, 543.3780518
3: -146.6180573, 498.0721130, -156.6787872, 531.2985229, -677.9165649, 654.7507324
4: -123.3942490, 459.2930908, -131.4108124, 488.7672729, -612.1614380, 590.7039185

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728045, upper bound: 560.5769726
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728045, upper bound: 560.5769726
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -96.1648254, 295.3631897, -121.0678635, 374.2596436, -470.4244690, 416.4310608
1: -136.4778290, 300.7017822, -171.5790863, 378.8587036, -515.3364868, 472.2808533
2: -115.3803406, 332.9954834, -144.9213104, 418.2716064, -533.6518555, 477.9168091
3: -123.1041336, 417.6282654, -154.6978455, 525.1723022, -648.2764282, 572.3261108
4: -103.5581360, 385.9321289, -129.7096252, 483.4325867, -586.9907227, 515.6417236

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5702139, upper bound: 560.5763217
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5702139, upper bound: 560.5763217
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -114.5463333, 352.9139709, -122.5349503, 378.8967285, -493.4430542, 475.4488831
1: -162.6037445, 358.2490845, -173.9224701, 383.3767700, -545.9805298, 532.1715698
2: -137.5226593, 396.1813660, -146.9243011, 423.0957947, -560.6184692, 543.1054688
3: -146.5297241, 497.7347107, -156.6787872, 531.2985229, -677.8282471, 654.4133911
4: -123.3238602, 458.9965515, -131.4108124, 488.7672729, -612.0910034, 590.4073486

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5726610, upper bound: 560.5766552
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5726610, upper bound: 560.5766552
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -125.3151016, 383.3699951, -123.0931473, 375.6898193, -501.0048828, 506.4631348
1: -177.4467773, 388.6565857, -174.3089600, 381.0484314, -558.4952393, 562.9654541
2: -150.1492920, 428.8045349, -147.5238495, 420.3610535, -570.5103760, 576.3283691
3: -159.8320770, 537.6771851, -156.9694366, 526.9136353, -686.7457275, 694.6466064
4: -134.0424194, 494.5388794, -131.6811676, 484.8074951, -618.8499146, 626.2199097

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792577, upper bound: 560.5790787
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792577, upper bound: 560.5790941
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -125.3151016, 383.3699951, -125.8798065, 381.1787109, -506.4938049, 509.2498169
1: -177.4467773, 388.6565857, -178.3494415, 387.3388367, -564.7856445, 567.0060425
2: -150.1492920, 428.8045349, -150.8644562, 427.7771606, -577.9263916, 579.6689453
3: -159.8320770, 537.6771851, -160.4801941, 534.9985962, -694.8306885, 698.1573486
4: -134.0424194, 494.5388794, -134.5471344, 493.0236206, -627.0660400, 629.0859985

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792577, upper bound: 560.5790787
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792577, upper bound: 560.5790941
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -123.4416428, 377.6062927, -128.2431183, 391.7998352, -515.2413940, 505.8494263
1: -174.8338013, 382.8805542, -181.6873322, 397.3684998, -572.2022705, 564.5678711
2: -147.9176788, 422.4621277, -153.6097412, 438.5272827, -586.4449463, 576.0718384
3: -157.4863892, 529.6583252, -163.6177216, 549.5948486, -707.0811768, 693.2760620
4: -132.0468597, 487.1866150, -137.1082153, 505.8510132, -637.8978271, 624.2947388

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5762863, upper bound: 560.5782209
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794321, upper bound: 560.5791382
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -124.2083588, 375.0057068, -128.2431183, 391.7998352, -516.0081177, 503.2488403
1: -175.5164948, 381.9026184, -181.6873322, 397.3684998, -572.8850098, 563.5899658
2: -148.6136475, 422.0887146, -153.6097412, 438.5272827, -587.1408081, 575.6982422
3: -158.2032166, 527.7643433, -163.6177216, 549.5948486, -707.7980957, 691.3820801
4: -132.7045746, 486.6079102, -137.1082153, 505.8510132, -638.5556030, 623.7158203

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792980, upper bound: 560.5787840
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794321, upper bound: 560.5791455
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -111.7308121, 339.1276855, -114.2450638, 342.8109436, -454.5417480, 453.3727417
1: -158.0385284, 344.8851624, -161.5024261, 349.8391724, -507.8776855, 506.3875732
2: -133.6682587, 381.2792664, -136.8422546, 386.9328003, -520.6010132, 518.1215210
3: -142.4163055, 476.8443909, -145.4195404, 482.9165955, -625.3328857, 622.2639160
4: -119.4174805, 439.9692688, -122.1577530, 445.7084045, -565.1258545, 562.1270142

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788191, upper bound: 560.5777011
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5756415, upper bound: 560.5774058
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5750202, upper bound: 560.5742845
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -111.7308121, 339.1276855, -117.0018158, 349.1543884, -460.8851929, 456.1295166
1: -158.0385284, 344.8851624, -165.4091034, 356.7609253, -514.7994385, 510.2942200
2: -133.6682587, 381.2792664, -140.0563660, 394.8855286, -528.5537720, 521.3355713
3: -142.4163055, 476.8443909, -148.9973450, 492.6399536, -635.0562134, 625.8417358
4: -119.4174805, 439.9692688, -125.0952988, 455.4996948, -574.9171753, 565.0645752

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5796727, upper bound: 560.5750653
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5750202, upper bound: 560.5742845
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -97.4426956, 299.3106384, -120.5417557, 368.2868652, -465.7295532, 419.8523865
1: -138.2017670, 304.6515503, -170.6495056, 373.9450378, -512.1467896, 475.3009949
2: -116.8124924, 337.3945312, -144.2661591, 413.1957092, -530.0081787, 481.6607056
3: -124.6575012, 423.0793762, -153.8923950, 518.0905762, -642.7480469, 576.9718018
4: -104.8619003, 390.9336548, -129.1324921, 477.7371216, -582.5989990, 520.0660400

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5730099, upper bound: 560.5792216
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5730099, upper bound: 560.5780612
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -106.5353012, 323.5266724, -124.8903198, 377.1516724, -483.6869812, 448.4169617
1: -150.5850067, 328.9982605, -176.6421661, 383.9039917, -534.4888916, 505.6404419
2: -127.3968277, 363.7165833, -149.4244537, 424.3194885, -551.7161255, 513.1409912
3: -135.7500610, 454.7708130, -159.2208557, 530.6939087, -666.4439697, 613.9916992
4: -113.7740707, 419.2009277, -133.4423523, 489.5030823, -603.2771606, 552.6433105

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5809294, upper bound: 560.5799038
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5809294, upper bound: 560.5802020
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -129.9535065, 397.2473145, -114.2450638, 342.8109436, -472.7644348, 511.4923096
1: -184.1100769, 402.6030579, -161.5024261, 349.8391724, -533.9492188, 564.1054688
2: -155.7455902, 444.3345032, -136.8422546, 386.9328003, -542.6782227, 581.1767578
3: -165.7575226, 557.2296143, -145.4195404, 482.9165955, -648.6741333, 702.6491699
4: -139.0264435, 513.0037231, -122.1577530, 445.7084045, -584.7348633, 635.1614380

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5796425, upper bound: 560.5786449
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5796425, upper bound: 560.5786494
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -129.9535065, 397.2473145, -117.0018158, 349.1543884, -479.1078796, 514.2490845
1: -184.1100769, 402.6030579, -165.4091034, 356.7609253, -540.8709717, 568.0121460
2: -155.7455902, 444.3345032, -140.0563660, 394.8855286, -550.6311035, 584.3908691
3: -165.7575226, 557.2296143, -148.9973450, 492.6399536, -658.3973999, 706.2269287
4: -139.0264435, 513.0037231, -125.0952988, 455.4996948, -594.5261230, 638.0989990

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788191, upper bound: 560.5776921
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5796727, upper bound: 560.5751778
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788423, upper bound: 560.5752280
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -128.0226135, 391.0363770, -125.7067261, 379.1259766, -507.1485901, 516.7429810
1: -181.2680206, 396.3690796, -177.7718353, 385.8962097, -567.1641846, 574.1409302
2: -153.3480988, 437.4625244, -150.3794403, 426.5780945, -579.9259644, 587.8419800
3: -163.2168579, 548.4954834, -160.2034149, 533.3610229, -696.5778198, 708.6989136
4: -136.9131927, 504.9101868, -134.3050079, 492.1593628, -629.0723267, 639.2152100

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5798886, upper bound: 560.5795212
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5798886, upper bound: 560.5795212
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -127.9034958, 390.5495605, -124.9170685, 376.4293213, -504.3328247, 515.4666138
1: -181.1608582, 395.9506226, -176.6444855, 383.2922668, -564.4531250, 572.5950928
2: -153.2670898, 437.0494995, -149.4490662, 423.8316650, -577.0987549, 586.4985352
3: -163.0983887, 547.9187012, -159.1657257, 529.9244385, -693.0227661, 707.0844116
4: -136.8083954, 504.4762573, -133.4847717, 489.0381470, -625.8464355, 637.9610596

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777875, upper bound: 560.5797933
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777875, upper bound: 560.5797571
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -119.9195709, 366.2033691, -101.0618286, 309.9206848, -429.8402100, 467.2651978
1: -169.8238831, 372.7666016, -142.8033905, 314.6390991, -484.4629517, 515.5699463
2: -143.6471558, 412.5382385, -120.7682037, 347.8873901, -491.5344849, 533.3063965
3: -153.1659393, 517.2355347, -128.7658386, 435.6432495, -588.8092041, 646.0013428
4: -128.9179382, 477.8540649, -108.1679916, 401.6184998, -530.5363770, 586.0219727

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5718027, upper bound: 560.5775637
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5712141, upper bound: 560.5770344
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -119.9195709, 366.2033691, -110.5318527, 336.0343933, -455.9539490, 476.7351685
1: -169.8238831, 372.7666016, -156.3856964, 341.7556458, -511.5794983, 529.1522827
2: -143.6471558, 412.5382385, -132.1538239, 378.0199890, -521.6671143, 544.6920166
3: -153.1659393, 517.2355347, -140.9421234, 472.7878723, -625.9537964, 658.1776733
4: -128.9179382, 477.8540649, -118.2759857, 436.5998840, -565.5176392, 596.1300659

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5730866, upper bound: 560.5777952
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5730866, upper bound: 560.5781454
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -121.6785889, 371.1994629, -120.7274399, 372.5253906, -494.2038879, 491.9269104
1: -172.5568848, 377.8683472, -170.8585815, 376.9385071, -549.4953613, 548.7269287
2: -146.0069733, 418.0887146, -144.5494232, 416.1507874, -562.1577759, 562.6380005
3: -155.4829712, 524.0260620, -153.9804077, 522.5287476, -678.0117188, 678.0063477
4: -130.9170074, 483.9896851, -129.3731079, 480.7271729, -611.6440430, 613.3627319

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729775, upper bound: 560.5765915
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729775, upper bound: 560.5792785
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -121.6785889, 371.1994629, -121.9174347, 373.2305908, -494.9090881, 493.1168823
1: -172.5568848, 377.8683472, -172.7810822, 378.3530579, -550.9099121, 550.6494141
2: -146.0069733, 418.0887146, -146.0597229, 417.8842773, -563.8911743, 564.1484375
3: -155.4829712, 524.0260620, -155.5123138, 523.6802368, -679.1632080, 679.5383911
4: -130.9170074, 483.9896851, -130.5527496, 482.4863586, -613.4031982, 614.5424194

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729775, upper bound: 560.5765915
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729775, upper bound: 560.5792785
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -120.1427917, 367.3106995, -102.0341721, 313.4450684, -433.5878601, 469.3448181
1: -170.2104187, 373.8418274, -144.2632599, 318.1817322, -488.3920898, 518.1049805
2: -143.9357452, 413.7109375, -121.9566574, 351.8126831, -495.7484131, 535.6675415
3: -153.5148163, 518.8136597, -130.1275635, 440.6820068, -594.1968384, 648.9411621
4: -129.1759033, 479.3582153, -109.2724915, 406.3101196, -535.4859619, 588.6306763

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5710220, upper bound: 560.5744010
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5723308, upper bound: 560.5743205
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5723308, upper bound: 560.5757113
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -120.1427917, 367.3106995, -111.3329849, 339.1594543, -459.3022461, 478.6436462
1: -170.2104187, 373.8418274, -157.5732269, 344.8350525, -515.0454712, 531.4150391
2: -143.9357452, 413.7109375, -133.1096191, 381.4466553, -525.3823853, 546.8203735
3: -153.5148163, 518.8136597, -142.0659485, 477.3213196, -630.8361206, 660.8796387
4: -129.1759033, 479.3582153, -119.1969452, 440.8366394, -570.0125732, 598.5549927

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5723308, upper bound: 560.5743205
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5723308, upper bound: 560.5757113
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -122.6620560, 374.4150696, -120.5239029, 372.0434875, -494.7055359, 494.9389648
1: -174.0574493, 381.1093445, -170.6943207, 376.4648743, -550.5221558, 551.8036499
2: -147.2472534, 421.6554871, -144.3832092, 415.6890564, -562.9362793, 566.0386963
3: -156.8245392, 528.5427856, -153.7874146, 521.9312744, -678.7557373, 682.3302002
4: -131.9986725, 488.2976379, -129.1891479, 480.2225952, -612.2211304, 617.4868164

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5670380, upper bound: 560.5777972
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5723308, upper bound: 560.5788842
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5723308, upper bound: 560.5795368
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -122.6620560, 374.4150696, -123.0435486, 376.7595215, -499.4215698, 497.4585876
1: -174.0574493, 381.1093445, -174.4557190, 381.9606628, -556.0180054, 555.5648804
2: -147.2472534, 421.6554871, -147.4536438, 422.0023193, -569.2495728, 569.1090698
3: -156.8245392, 528.5427856, -157.0422516, 528.8126221, -685.6371460, 685.5850220
4: -131.9986725, 488.2976379, -131.8476105, 487.4066162, -619.4052734, 620.1452637

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5732324, upper bound: 560.5784961
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5732324, upper bound: 560.5799749
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -113.0328445, 338.5581055, -120.0344696, 366.4639282, -479.4967651, 458.5925903
1: -158.9756317, 344.1450806, -168.9707031, 370.7972717, -529.7728882, 513.1157227
2: -134.4783783, 380.1960449, -142.7505951, 409.2299194, -543.7082520, 522.9466553
3: -143.3408508, 475.4277954, -152.4058228, 513.4776001, -656.8184814, 627.8336182
4: -120.2272491, 438.5757446, -127.5335159, 472.2354431, -592.4626465, 566.1092529

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5622514, upper bound: 560.5613327
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5622514, upper bound: 560.5676495
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -114.6040115, 344.2165833, -117.2706833, 359.5879822, -474.1919861, 461.4872742
1: -161.2019653, 349.6532288, -165.2231903, 363.5224609, -524.7244263, 514.8762817
2: -136.3444061, 386.2251587, -139.5133514, 401.0580139, -537.4023438, 525.7385254
3: -145.3687744, 483.2318115, -149.0971222, 503.3690491, -648.7377930, 632.3288574
4: -121.9228745, 445.5721741, -124.7022018, 462.5181580, -584.4410400, 570.2742920

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5620243, upper bound: 560.5698004
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5620243, upper bound: 560.5706596
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -127.2526550, 381.8688965, -120.0344696, 366.4639282, -493.7165833, 501.9033813
1: -180.3879852, 389.4684143, -168.9707031, 370.7972717, -551.1852417, 558.4390869
2: -152.6966553, 430.6887817, -142.7505951, 409.2299194, -561.9265747, 573.4393921
3: -162.3751068, 537.8438110, -152.4058228, 513.4776001, -675.8527222, 690.2495728
4: -136.3590698, 496.7335205, -127.5335159, 472.2354431, -608.5944824, 624.2670288

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683150, upper bound: 560.5623616
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683150, upper bound: 560.5687905
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -128.6148682, 386.6873779, -117.2706833, 359.5879822, -488.2028198, 503.9580688
1: -182.2928619, 394.1914368, -165.2231903, 363.5224609, -545.8153076, 559.4144897
2: -154.3034668, 435.8630981, -139.5133514, 401.0580139, -555.3614502, 575.3764648
3: -164.1164246, 544.4848022, -149.0971222, 503.3690491, -667.4854736, 693.5819092
4: -137.8252563, 502.7117310, -124.7022018, 462.5181580, -600.3433838, 627.4139404

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5603503, upper bound: 560.5697855
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5603503, upper bound: 560.5723287
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -115.4690552, 345.0193787, -131.4072418, 400.2196350, -515.6887207, 476.4266357
1: -161.9754639, 350.8452148, -186.5669556, 406.1948242, -568.1702271, 537.4121704
2: -137.0639343, 387.8125610, -157.7274628, 448.3967896, -585.4606934, 545.5400391
3: -146.0861053, 484.9868164, -167.8526306, 561.5205078, -707.6066284, 652.8394775
4: -122.6019363, 447.6185913, -140.7045593, 517.4260864, -640.0278931, 588.3230591

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605413, upper bound: 560.5581990
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605413, upper bound: 560.5695831
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -112.9905243, 339.8621521, -132.9510345, 405.5420532, -518.5325928, 472.8131714
1: -158.8236237, 345.1923828, -188.7662201, 411.4202881, -570.2438965, 533.9586182
2: -134.3068542, 381.3728943, -159.5817871, 454.1270752, -588.4338989, 540.9547119
3: -143.3074646, 477.1377563, -169.8479004, 568.8524170, -712.1598511, 646.9856567
4: -120.1810074, 439.8717957, -142.3677826, 524.0730591, -644.2539673, 582.2395630

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5618852, upper bound: 560.5585451
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5618852, upper bound: 560.5707289
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -129.2583160, 388.9176025, -133.9061584, 403.7178345, -532.9761353, 522.8236694
1: -183.2346954, 396.3386536, -189.7881927, 411.1179199, -594.3525391, 586.1268311
2: -155.1126709, 438.1524048, -160.6450043, 454.4287109, -609.5413818, 598.7974243
3: -164.9413910, 547.4499512, -170.8900299, 568.0398560, -732.9811401, 718.3399048
4: -138.5189056, 505.3483276, -143.5208435, 524.2348633, -662.7537231, 648.8691406

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5778166, upper bound: 560.5784220
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5778165, upper bound: 560.5798528
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -129.2583160, 388.9176025, -133.3831940, 406.5639648, -535.8222656, 522.3007812
1: -183.2346954, 396.3386536, -189.3949585, 412.5765686, -595.8112793, 585.7336426
2: -155.1126709, 438.1524048, -160.1429749, 455.4507141, -610.5633545, 598.2954102
3: -164.9413910, 547.4499512, -170.4139099, 570.3533325, -735.2945557, 717.8638916
4: -138.5189056, 505.3483276, -142.8468781, 525.5026855, -664.0215454, 648.1951904

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5764361, upper bound: 560.5796674
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793055, upper bound: 560.5801849
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -131.3517914, 400.9027405, -133.6826935, 406.7755432, -538.1273193, 534.5853882
1: -186.4413300, 406.5832825, -189.7682800, 412.8060913, -599.2474365, 596.3515625
2: -157.5447388, 448.7116089, -160.3578339, 455.7443848, -613.2890625, 609.0693359
3: -167.7609863, 561.8778076, -170.7280121, 570.6857910, -738.4467773, 732.6057739
4: -140.5767822, 517.3845215, -143.0380249, 526.0008545, -666.5776367, 660.4223022

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5668655, upper bound: 560.5729733
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5668655, upper bound: 560.5729733
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -131.3517914, 400.9027405, -131.6188660, 397.3190002, -528.6707764, 532.5215454
1: -186.4413300, 406.5832825, -186.3482666, 404.3933411, -590.8346558, 592.9315186
2: -157.5447388, 448.7116089, -157.5971680, 447.0506592, -604.5952759, 606.3087158
3: -167.7609863, 561.8778076, -167.7583618, 558.9235840, -726.6845703, 729.6361694
4: -140.5767822, 517.3845215, -140.6687012, 515.6995239, -656.2761841, 658.0532227

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5636744, upper bound: 560.5740001
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5664015, upper bound: 560.5728020
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5665284, upper bound: 560.5730962
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -130.0877228, 394.9985352, -133.6826935, 406.7755432, -536.8632202, 528.6811523
1: -184.5915833, 401.0894165, -189.7682800, 412.8060913, -597.3976440, 590.8576660
2: -156.0140839, 442.9841614, -160.3578339, 455.7443848, -611.7583618, 603.3419189
3: -166.0739441, 554.3604736, -170.7280121, 570.6857910, -736.7597046, 725.0884399
4: -139.1658325, 511.0791931, -143.0380249, 526.0008545, -665.1666870, 654.1170654

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5720846, upper bound: 560.5697299
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5685405, upper bound: 560.5685405
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -130.0877228, 394.9985352, -131.6188660, 397.3190002, -527.4067383, 526.6173096
1: -184.5915833, 401.0894165, -186.3482666, 404.3933411, -588.9848633, 587.4376831
2: -156.0140839, 442.9841614, -157.5971680, 447.0506592, -603.0645752, 600.5812988
3: -166.0739441, 554.3604736, -167.7583618, 558.9235840, -724.9974976, 722.1188354
4: -139.1658325, 511.0791931, -140.6687012, 515.6995239, -654.8652344, 651.7479248

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5685766, upper bound: 560.5691994
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5685405, upper bound: 560.5685405
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -127.0607758, 382.6828918, -133.6826935, 406.7755432, -533.8363037, 516.3654785
1: -179.8215637, 389.7965698, -189.7682800, 412.8060913, -592.6276855, 579.5648193
2: -152.0997467, 431.0142212, -160.3578339, 455.7443848, -607.8439941, 591.3720703
3: -161.8722382, 538.5751953, -170.7280121, 570.6857910, -732.5580444, 709.3031616
4: -135.7593384, 497.0256958, -143.0380249, 526.0008545, -661.7601929, 640.0635376

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5750828, upper bound: 560.5746727
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745537, upper bound: 560.5732715
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -127.0607758, 382.6828918, -131.6188660, 397.3190002, -524.3797607, 514.3016968
1: -179.8215637, 389.7965698, -186.3482666, 404.3933411, -584.2149048, 576.1448364
2: -152.0997467, 431.0142212, -157.5971680, 447.0506592, -599.1502075, 588.6113892
3: -161.8722382, 538.5751953, -167.7583618, 558.9235840, -720.7958374, 706.3335571
4: -135.7593384, 497.0256958, -140.6687012, 515.6995239, -651.4588013, 637.6943970

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5737409, upper bound: 560.5756566
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5726325, upper bound: 560.5701980
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -121.9825134, 373.3664856, -177.8211517, 540.8812256, -662.8637695, 551.1876221
1: -171.9560394, 377.5604858, -250.8834229, 548.1641846, -720.1201782, 628.4439087
2: -145.2335205, 416.6315002, -211.6369324, 606.2115479, -751.4450073, 628.2684326
3: -155.0872650, 522.9097900, -226.3085022, 757.3580933, -912.4453125, 749.2182617
4: -129.7929230, 480.7670288, -189.4899445, 698.7535400, -828.5464478, 670.2569580

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -122.2061234, 372.3715820, -177.8211517, 540.8812256, -663.0873413, 550.1927490
1: -172.2659760, 376.8753662, -250.8834229, 548.1641846, -720.4301147, 627.7587891
2: -145.5442657, 416.0659485, -211.6369324, 606.2115479, -751.7557983, 627.7028809
3: -155.3141479, 521.8527832, -226.3085022, 757.3580933, -912.6721802, 748.1612549
4: -130.0316772, 480.2986145, -189.4899445, 698.7535400, -828.7851562, 669.7885132

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -121.9825134, 373.3664856, -198.2688141, 603.6936646, -725.6761475, 571.6353149
1: -171.9560394, 377.5604858, -281.1198730, 612.5306396, -784.4866943, 658.6803589
2: -145.2335205, 416.6315002, -237.3463593, 677.2596436, -822.4931030, 653.9778442
3: -155.0872650, 522.9097900, -253.3197174, 845.5316162, -1000.6188965, 776.2294922
4: -129.7929230, 480.7670288, -212.3357697, 780.1259155, -909.9188232, 693.1027832

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560725, upper bound: 560.5310706
time: 0.92 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560725, upper bound: 560.5375299
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -121.9825134, 373.3664856, -190.2652740, 580.8919067, -702.8743896, 563.6317139
1: -171.9560394, 377.5604858, -269.5406494, 589.1798706, -761.1358643, 647.1011353
2: -145.2335205, 416.6315002, -227.6355896, 651.6405640, -796.8740845, 644.2670898
3: -155.0872650, 522.9097900, -242.9345245, 814.3419800, -969.4292603, 765.8442993
4: -129.7929230, 480.7670288, -203.7815399, 751.1313477, -880.9242554, 684.5485840

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560725, upper bound: 560.5391695
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560725, upper bound: 560.5391695
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -122.2061234, 372.3715820, -196.3585205, 597.5297852, -719.7359009, 568.7300415
1: -172.2659760, 376.8753662, -278.3824158, 606.4171753, -778.6831055, 655.2578125
2: -145.5442657, 416.0659485, -235.0536194, 670.5921021, -816.1363525, 651.1195679
3: -155.3141479, 521.8527832, -250.8645630, 837.0253906, -992.3395386, 772.7172852
4: -130.0316772, 480.2986145, -210.3136292, 772.3630981, -902.3946533, 690.6122437

Time for backsubstitution: 0.88 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.89 + 417.14 = 420.03 seconds
