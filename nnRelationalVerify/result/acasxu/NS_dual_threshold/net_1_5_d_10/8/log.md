## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.5202488034


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706)
1: (-6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965)
2: (-5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000)
3: (-7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002)
4: (-5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.70 + 1.65 = 2.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -27.5477966, upper bound: 27.5477966

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5466365, upper bound: 27.5440437
time: 0.56 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5476940, upper bound: 27.5476940
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.24 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 3, lower bound: -27.5466365, upper bound: 27.5440437
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 3, lower bound: -27.5476940, upper bound: 27.5476940

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.1535349, 14.4982834, -4.8926516, 16.6545238, -20.8080578, 19.3909321
1: -6.0371099, 14.8871717, -6.9941020, 17.1100998, -23.1472054, 21.8812714
2: -5.1117692, 16.7495937, -5.9263554, 19.1902447, -24.3020134, 22.6759491
3: -6.0486369, 21.3488541, -7.0618343, 24.5106659, -30.5593033, 28.4106884
4: -4.9890728, 19.7949219, -5.7611084, 22.7441978, -27.7332706, 25.5560303

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386160, upper bound: 27.5375745
time: 0.57 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5460602, upper bound: 27.5435831
time: 0.83 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.8032727, 16.3772182, -4.8926516, 16.6545238, -21.4577904, 21.2698650
1: -6.8679352, 16.8314762, -6.9941020, 17.1100998, -23.9780331, 23.8255749
2: -5.8185549, 18.8867245, -5.9263554, 19.1902447, -25.0087986, 24.8130779
3: -6.9354429, 24.1179543, -7.0618343, 24.5106659, -31.4461098, 31.1797886
4: -5.6641207, 22.3921146, -5.7611084, 22.7441978, -28.4083176, 28.1532230

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5440437, upper bound: 27.5466365
time: 0.60 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5440437, upper bound: 27.5476940
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.92 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 3, lower bound: -27.5386160, upper bound: 27.5375745
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 3, lower bound: -27.5460602, upper bound: 27.5435831
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 3, lower bound: -27.5440437, upper bound: 27.5466365
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 3, lower bound: -27.5440437, upper bound: 27.5476940

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.1341019, 14.4451771, -3.2905438, 11.8368683, -15.9709682, 17.7357216
1: -6.0095778, 14.8314266, -4.6927733, 12.0747938, -18.0843716, 19.5241985
2: -5.0889492, 16.6874809, -3.9797940, 13.5427437, -18.6316929, 20.6672726
3: -6.0211139, 21.2716236, -4.6745629, 17.4246578, -23.4457722, 25.9461861
4: -4.9685078, 19.7226830, -3.9391065, 16.1370792, -21.1055851, 23.6617889

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5328998
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5363881
time: 0.44 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.1535349, 14.4982834, -4.8764224, 16.6146393, -20.7681713, 19.3747044
1: -6.0371099, 14.8871717, -6.9735188, 17.0683918, -23.1054974, 21.8606892
2: -5.1117692, 16.7495937, -5.9087296, 19.1438313, -24.2556000, 22.6583233
3: -6.0486369, 21.3488541, -7.0417824, 24.4534302, -30.5020676, 28.3906326
4: -4.9890728, 19.7949219, -5.7452898, 22.6896019, -27.6786747, 25.5402107

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386160, upper bound: 27.5423827
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5423827, upper bound: 27.5435831
time: 0.56 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.8032727, 16.3772182, -4.1535349, 14.4982834, -19.3015556, 20.5307503
1: -6.8679352, 16.8314762, -6.0371099, 14.8871717, -21.7551079, 22.8685818
2: -5.8185549, 18.8867245, -5.1117692, 16.7495937, -22.5681496, 23.9984932
3: -6.9354429, 24.1179543, -6.0486369, 21.3488541, -28.2842979, 30.1665916
4: -5.6641207, 22.3921146, -4.9890728, 19.7949219, -25.4590416, 27.3811874

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375745, upper bound: 27.5386160
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5375745, upper bound: 27.5460602
time: 0.61 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.8032727, 16.3772182, -4.8032727, 16.3772182, -21.1804867, 21.1804867
1: -6.8679352, 16.8314762, -6.8679352, 16.8314762, -23.6994114, 23.6994114
2: -5.8185549, 18.8867245, -5.8185549, 18.8867245, -24.7052765, 24.7052746
3: -6.9354429, 24.1179543, -6.9354429, 24.1179543, -31.0533962, 31.0533962
4: -5.6641207, 22.3921146, -5.6641207, 22.3921146, -28.0562363, 28.0562363

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398164, upper bound: 27.5412519
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5435831, upper bound: 27.5472606
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.97 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5328998
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5363881
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -27.5386160, upper bound: 27.5423827
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -27.5423827, upper bound: 27.5435831
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -27.5375745, upper bound: 27.5386160
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -27.5375745, upper bound: 27.5460602
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -27.5398164, upper bound: 27.5412519
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 1.97
Output dim: 3, lower bound: -27.5435831, upper bound: 27.5472606

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.8787930, 13.7537832, -3.2905438, 11.8368683, -15.7156591, 17.0443268
1: -5.6302629, 14.0933819, -4.6927733, 12.0747938, -17.7050533, 18.7861538
2: -4.7595100, 15.8591976, -3.9797940, 13.5427437, -18.3022537, 19.8389893
3: -5.6521206, 20.2543964, -4.6745629, 17.4246578, -23.0767784, 24.9289589
4: -4.6689615, 18.7457523, -3.9391065, 16.1370792, -20.8060417, 22.6848583

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5328998
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5328998
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.2500224, 14.9438887, -3.2905438, 11.8368683, -16.0868893, 18.2344284
1: -6.1566515, 15.2967548, -4.6927733, 12.0747938, -18.2314453, 19.9895267
2: -5.2083526, 17.1389999, -3.9797940, 13.5427437, -18.7510967, 21.1187935
3: -6.1825032, 21.9082413, -4.6745629, 17.4246578, -23.6071606, 26.5828037
4: -5.0796795, 20.2783566, -3.9391065, 16.1370792, -21.2167587, 24.2174625

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5365861, upper bound: 27.5363881
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5363881
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -4.1535349, 14.4982834, -4.1433725, 14.4690485, -18.6225834, 18.6416531
1: -6.0371099, 14.8871717, -6.0221658, 14.8564215, -20.8935299, 20.9093361
2: -5.1117692, 16.7495937, -5.0990467, 16.7156181, -21.8273849, 21.8486404
3: -6.0486369, 21.3488541, -6.0337214, 21.3067570, -27.3553944, 27.3825760
4: -4.9890728, 19.7949219, -4.9775858, 19.7547321, -24.7438030, 24.7725067

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5369818, upper bound: 27.5369047
time: 0.61 seconds

## Relational analysis of NS_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317780, upper bound: 27.5374895
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317780, upper bound: 27.5374895
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -4.1535349, 14.4982834, -4.7884359, 16.3409023, -20.4944382, 19.2867184
1: -6.0371099, 14.8871717, -6.8493519, 16.7934227, -22.8305302, 21.7365208
2: -5.1117692, 16.7495937, -5.8027220, 18.8445301, -23.9562988, 22.5523148
3: -6.0486369, 21.3488541, -6.9172134, 24.0655479, -30.1141853, 28.2660675
4: -4.9890728, 19.7949219, -5.6499076, 22.3423691, -27.3314419, 25.4448299

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317780, upper bound: 27.5384215
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5409776, upper bound: 27.5419098
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -4.1341019, 14.4451771, -17.6682663, 15.7628689
1: -4.5962033, 11.8650751, -6.0095778, 14.8314266, -19.4276295, 17.8746529
2: -3.8960257, 13.3130264, -5.0889492, 16.6874809, -20.5835075, 18.4019737
3: -4.5771689, 17.1330128, -6.0211139, 21.2716236, -25.8487930, 23.1541271
4: -3.8639328, 15.8723202, -4.9685078, 19.7226830, -23.5866165, 20.8408222

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328998, upper bound: 27.5273865
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328998, upper bound: 27.5365861
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7884359, 16.3409023, -4.1535349, 14.4982834, -19.2867203, 20.4944382
1: -6.8493519, 16.7934227, -6.0371099, 14.8871717, -21.7365208, 22.8305302
2: -5.8027220, 18.8445301, -5.1117692, 16.7495937, -22.5523148, 23.9562988
3: -6.9172134, 24.0655479, -6.0486369, 21.3488541, -28.2660675, 30.1141853
4: -5.6499076, 22.3423691, -4.9890728, 19.7949219, -25.4448299, 27.3314419

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328998, upper bound: 27.5354531
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328998, upper bound: 27.5446527
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -4.7838039, 16.3224258, -3.2230899, 11.6287680, -16.4125710, 19.5455132
1: -6.8397312, 16.7739010, -4.5962033, 11.8650751, -18.7048073, 21.3701038
2: -5.7952309, 18.8222408, -3.8960257, 13.3130264, -19.1082573, 22.7182655
3: -6.9071493, 24.0379963, -4.5771689, 17.1330128, -24.0401592, 28.6151657
4: -5.6429553, 22.3172417, -3.8639328, 15.8723202, -21.5152683, 26.1811714

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5328998
time: 0.61 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5400632
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -4.8032727, 16.3772182, -4.7884359, 16.3409023, -21.1441727, 21.1656532
1: -6.8679352, 16.8314762, -6.8493519, 16.7934227, -23.6613579, 23.6808243
2: -5.8185549, 18.8867245, -5.8027220, 18.8445301, -24.6630840, 24.6894436
3: -6.9354429, 24.1179543, -6.9172134, 24.0655479, -31.0009918, 31.0351658
4: -5.6641207, 22.3921146, -5.6499076, 22.3423691, -28.0064888, 28.0420227

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5276802, upper bound: 27.5381153
time: 0.51 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5438369, upper bound: 27.5452615
time: 0.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.88 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5328998
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5328998
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5365861, upper bound: 27.5363881
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5363881
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5317780, upper bound: 27.5374895
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5317780, upper bound: 27.5374895
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5317780, upper bound: 27.5384215
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5409776, upper bound: 27.5419098
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5328998, upper bound: 27.5273865
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5328998, upper bound: 27.5365861
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5328998, upper bound: 27.5354531
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5328998, upper bound: 27.5446527
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5328998
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5273865, upper bound: 27.5400632
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5276802, upper bound: 27.5381153
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 1.88
Output dim: 3, lower bound: -27.5438369, upper bound: 27.5452615

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.8787930, 13.7537832, -2.6493955, 10.1003323, -13.9791222, 16.4031773
1: -5.6302629, 14.0933819, -3.8986821, 10.2718744, -15.9021358, 17.9920635
2: -4.7595100, 15.8591976, -3.3160415, 11.5687923, -16.3283024, 19.1752357
3: -5.6521206, 20.2543964, -3.8553152, 14.8965769, -20.5486984, 24.1097107
4: -4.6689615, 18.7457523, -3.3364959, 13.8042555, -18.4732170, 22.0822487

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5073330, upper bound: 27.5201165
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5220844, upper bound: 27.5273486
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.8787930, 13.7537832, -3.2230899, 11.6287680, -15.5075598, 16.9768734
1: -5.6302629, 14.0933819, -4.5962033, 11.8650751, -17.4953384, 18.6895847
2: -4.7595100, 15.8591976, -3.8960257, 13.3130264, -18.0725327, 19.7552223
3: -5.6521206, 20.2543964, -4.5771689, 17.1330128, -22.7851315, 24.8315659
4: -4.6689615, 18.7457523, -3.8639328, 15.8723202, -20.5412788, 22.6096859

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5073330, upper bound: 27.5201165
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5220844, upper bound: 27.5273486
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.2500224, 14.9438887, -2.6493955, 10.1003323, -14.3503532, 17.5932846
1: -6.1566515, 15.2967548, -3.8986821, 10.2718744, -16.4285259, 19.1954365
2: -5.2083526, 17.1389999, -3.3160415, 11.5687923, -16.7771454, 20.4550400
3: -6.1825032, 21.9082413, -3.8553152, 14.8965769, -21.0790787, 25.7635574
4: -5.0796795, 20.2783566, -3.3364959, 13.8042555, -18.8839340, 23.6148529

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5228559, upper bound: 27.5353453
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5345493, upper bound: 27.5354791
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.2500224, 14.9438887, -3.2230899, 11.6287680, -15.8787899, 18.1669788
1: -6.1566515, 15.2967548, -4.5962033, 11.8650751, -18.0217266, 19.8929577
2: -5.2083526, 17.1389999, -3.8960257, 13.3130264, -18.5213776, 21.0350266
3: -6.1825032, 21.9082413, -4.5771689, 17.1330128, -23.3155155, 26.4854107
4: -5.0796795, 20.2783566, -3.8639328, 15.8723202, -20.9519958, 24.1422882

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5039373, upper bound: 27.5353453
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5345493, upper bound: 27.5354791
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3.8966453, 13.8043938, -4.1433725, 14.4690485, -18.3656940, 17.9477654
1: -5.6560292, 14.1464357, -6.0221658, 14.8564215, -20.5124512, 20.1685963
2: -4.7809715, 15.9183598, -5.0990467, 16.7156181, -21.4965897, 21.0174065
3: -5.6777368, 20.3282623, -6.0337214, 21.3067570, -26.9844933, 26.3619843
4: -4.6884899, 18.8143234, -4.9775858, 19.7547321, -24.4432220, 23.7919083

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5125097, upper bound: 27.5282895
time: 0.62 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5282895, upper bound: 27.5282895
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.1433725, 14.4690485, -18.7361202, 19.1363068
1: -6.1816254, 15.3483210, -6.0221658, 14.8564215, -21.0380478, 21.3704777
2: -5.2292953, 17.1967506, -5.0990467, 16.7156181, -21.9449139, 22.2957954
3: -6.2076254, 21.9800053, -6.0337214, 21.3067570, -27.5143814, 28.0137253
4: -5.0987215, 20.3454666, -4.9775858, 19.7547321, -24.8534527, 25.3230515

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5374895, upper bound: 27.5317779
time: 0.61 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5374895, upper bound: 27.5317779
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3.8966453, 13.8043938, -4.7884359, 16.3409023, -20.2375469, 18.5928307
1: -5.6560292, 14.1464357, -6.8493519, 16.7934227, -22.4494514, 20.9957829
2: -4.7809715, 15.9183598, -5.8027220, 18.8445301, -23.6255016, 21.7210808
3: -5.6777368, 20.3282623, -6.9172134, 24.0655479, -29.7432842, 27.2454758
4: -4.6884899, 18.8143234, -5.6499076, 22.3423691, -27.0308590, 24.4642315

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258604, upper bound: 27.5325953
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5337995, upper bound: 27.5330461
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.7884359, 16.3409023, -20.6079750, 19.7813721
1: -6.1816254, 15.3483210, -6.8493519, 16.7934227, -22.9750481, 22.1976643
2: -5.2292953, 17.1967506, -5.8027220, 18.8445301, -24.0738258, 22.9994717
3: -6.2076254, 21.9800053, -6.9172134, 24.0655479, -30.2731743, 28.8972187
4: -5.0987215, 20.3454666, -5.6499076, 22.3423691, -27.4410896, 25.9953747

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5434660, upper bound: 27.5325372
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5434660, upper bound: 27.5419099
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -3.8787930, 13.7537832, -16.9768734, 15.5075598
1: -4.5962033, 11.8650751, -5.6302629, 14.0933819, -18.6895847, 17.4953384
2: -3.8960257, 13.3130264, -4.7595100, 15.8591976, -19.7552223, 18.0725307
3: -4.5771689, 17.1330128, -5.6521206, 20.2543964, -24.8315659, 22.7851315
4: -3.8639328, 15.8723202, -4.6689615, 18.7457523, -22.6096859, 20.5412788

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5201165, upper bound: 27.5073330
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5201165, upper bound: 27.5220844
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -4.2500224, 14.9438887, -18.1669788, 15.8787899
1: -4.5962033, 11.8650751, -6.1566515, 15.2967548, -19.8929558, 18.0217266
2: -3.8960257, 13.3130264, -5.2083526, 17.1389999, -21.0350266, 18.5213757
3: -4.5771689, 17.1330128, -6.1825032, 21.9082413, -26.4854107, 23.3155155
4: -3.8639328, 15.8723202, -5.0796795, 20.2783566, -24.1422882, 20.9519958

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5278178, upper bound: 27.5039373
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5354791, upper bound: 27.5345493
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.7884359, 16.3409023, -3.8966453, 13.8043938, -18.5928307, 20.2375469
1: -6.8493519, 16.7934227, -5.6560292, 14.1464357, -20.9957829, 22.4494514
2: -5.8027220, 18.8445301, -4.7809715, 15.9183598, -21.7210808, 23.6255016
3: -6.9172134, 24.0655479, -5.6777368, 20.3282623, -27.2454758, 29.7432842
4: -5.6499076, 22.3423691, -4.6884899, 18.8143234, -24.4642315, 27.0308590

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5325953, upper bound: 27.5340464
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330461, upper bound: 27.5337995
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.7884359, 16.3409023, -4.2670722, 14.9929380, -19.7813740, 20.6079750
1: -6.8493519, 16.7934227, -6.1816254, 15.3483210, -22.1976662, 22.9750481
2: -5.8027220, 18.8445301, -5.2292953, 17.1967506, -22.9994736, 24.0738258
3: -6.9172134, 24.0655479, -6.2076254, 21.9800053, -28.8972187, 30.2731743
4: -5.6499076, 22.3423691, -5.0987215, 20.3454666, -25.9953728, 27.4410896

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5325372, upper bound: 27.5434660
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5325372, upper bound: 27.5446530
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -4.4854693, 15.5458088, -3.2230899, 11.6287680, -16.1142349, 18.7688961
1: -6.4143353, 15.9462843, -4.5962033, 11.8650751, -18.2794113, 20.5424862
2: -5.4284940, 17.8944187, -3.8960257, 13.3130264, -18.7415199, 21.7904434
3: -6.4913116, 22.9025860, -4.5771689, 17.1330128, -23.6243229, 27.4797554
4: -5.3140855, 21.2266712, -3.8639328, 15.8723202, -21.1864052, 25.0906029

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5073330, upper bound: 27.5265403
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5229487, upper bound: 27.5348086
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -4.8927426, 16.7637806, -3.2230899, 11.6287680, -16.5215092, 19.9868698
1: -6.9896717, 17.1797619, -4.5962033, 11.8650751, -18.8547478, 21.7759647
2: -5.8971844, 19.2085629, -3.8960257, 13.3130264, -19.2102108, 23.1045876
3: -7.0501633, 24.5901051, -4.5771689, 17.1330128, -24.1831741, 29.1672745
4: -5.7360182, 22.7893887, -3.8639328, 15.8723202, -21.6083355, 26.6533203

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4959917, upper bound: 27.5114814
time: 0.63 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340057, upper bound: 27.5364650
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -4.3976188, 15.3468742, -4.7884359, 16.3409023, -20.7385216, 20.1353073
1: -6.2940626, 15.7216339, -6.8493519, 16.7934227, -23.0874825, 22.5709820
2: -5.3095002, 17.6520042, -5.8027220, 18.8445301, -24.1540298, 23.4547272
3: -6.3730483, 22.6263657, -6.9172134, 24.0655479, -30.4385967, 29.5435791
4: -5.2157865, 20.9694633, -5.6499076, 22.3423691, -27.5581551, 26.6193695

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5016667, upper bound: 27.5074979
time: 0.64 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5042102, upper bound: 27.5185584
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.7884359, 16.3409023, -21.0615120, 21.0207634
1: -6.7455416, 16.6543579, -6.8493519, 16.7934227, -23.5389633, 23.5037098
2: -5.7037611, 18.6685505, -5.8027220, 18.8445301, -24.5482883, 24.4712715
3: -6.8226271, 23.8968391, -6.9172134, 24.0655479, -30.8881760, 30.8140526
4: -5.5644245, 22.1408634, -5.6499076, 22.3423691, -27.9067936, 27.7907696

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5393670, upper bound: 27.5324849
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5421455, upper bound: 27.5435419
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.19 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5073330, upper bound: 27.5201165
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5220844, upper bound: 27.5273486
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5073330, upper bound: 27.5201165
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5220844, upper bound: 27.5273486
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5228559, upper bound: 27.5353453
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5345493, upper bound: 27.5354791
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5039373, upper bound: 27.5353453
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5345493, upper bound: 27.5354791
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5125097, upper bound: 27.5282895
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5282895, upper bound: 27.5282895
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5374895, upper bound: 27.5317779
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5374895, upper bound: 27.5317779
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5258604, upper bound: 27.5325953
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5337995, upper bound: 27.5330461
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5434660, upper bound: 27.5325372
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5434660, upper bound: 27.5419099
NS_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5201165, upper bound: 27.5073330
NS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5201165, upper bound: 27.5220844
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5278178, upper bound: 27.5039373
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5354791, upper bound: 27.5345493
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5325953, upper bound: 27.5340464
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5330461, upper bound: 27.5337995
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5325372, upper bound: 27.5434660
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5325372, upper bound: 27.5446530
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5073330, upper bound: 27.5265403
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5229487, upper bound: 27.5348086
NS_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.4959917, upper bound: 27.5114814
NS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5340057, upper bound: 27.5364650
NS_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5016667, upper bound: 27.5074979
NS_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5042102, upper bound: 27.5185584
NS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5393670, upper bound: 27.5324849
NS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.19
Output dim: 3, lower bound: -27.5421455, upper bound: 27.5435419

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.8199701, 13.6537867, -2.6493955, 10.1003323, -13.9203014, 16.3031826
1: -5.5330939, 13.9627600, -3.8986821, 10.2718744, -15.8049679, 17.8614426
2: -4.6683445, 15.6982718, -3.3160415, 11.5687923, -16.2371349, 19.0143108
3: -5.5629125, 20.1039257, -3.8553152, 14.8965769, -20.4594879, 23.9592400
4: -4.5901971, 18.5590172, -3.3364959, 13.8042555, -18.3944530, 21.8955135

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4930094, upper bound: 27.5208456
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5215962, upper bound: 27.5265891
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8199701, 13.6537867, -3.2230899, 11.6287680, -15.4487371, 16.8768768
1: -5.5330939, 13.9627600, -4.5962033, 11.8650751, -17.3981686, 18.5589638
2: -4.6683445, 15.6982718, -3.8960257, 13.3130264, -17.9813690, 19.5942974
3: -5.5629125, 20.1039257, -4.5771689, 17.1330128, -22.6959209, 24.6810951
4: -4.5901971, 18.5590172, -3.8639328, 15.8723202, -20.4625111, 22.4229488

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4823696, upper bound: 27.5202765
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5206469, upper bound: 27.5260200
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.2170448, 14.8499813, -2.6493955, 10.1003323, -14.3173761, 17.4993763
1: -6.0874748, 15.1848240, -3.8986821, 10.2718744, -16.3593483, 19.0835056
2: -5.1462145, 16.9952316, -3.3160415, 11.5687923, -16.7150040, 20.3112736
3: -6.1174002, 21.7649803, -3.8553152, 14.8965769, -21.0139771, 25.6202965
4: -5.0203681, 20.1216583, -3.3364959, 13.8042555, -18.8246212, 23.4581547

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4992940, upper bound: 27.5321894
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4980277, upper bound: 27.5207970
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.1861086, 14.7579079, -2.6493955, 10.1003323, -14.2864408, 17.4073029
1: -6.0617056, 15.1005745, -3.8986821, 10.2718744, -16.3335800, 18.9992561
2: -5.1281843, 16.9207191, -3.3160415, 11.5687923, -16.6969757, 20.2367554
3: -6.0870829, 21.6393185, -3.8553152, 14.8965769, -20.9836597, 25.4946327
4: -5.0075822, 20.0201283, -3.3364959, 13.8042555, -18.8118362, 23.3566246

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5321858, upper bound: 27.5324076
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5311005, upper bound: 27.5312373
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.2170448, 14.8499813, -3.2230899, 11.6287680, -15.8458118, 18.0730705
1: -6.0874748, 15.1848240, -4.5962033, 11.8650751, -17.9525490, 19.7810268
2: -5.1462145, 16.9952316, -3.8960257, 13.3130264, -18.4592400, 20.8912582
3: -6.1174002, 21.7649803, -4.5771689, 17.1330128, -23.2504120, 26.3421497
4: -5.0203681, 20.1216583, -3.8639328, 15.8723202, -20.8926811, 23.9855919

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4816465, upper bound: 27.5076902
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5182523, upper bound: 27.5313274
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.1861086, 14.7579079, -3.2230899, 11.6287680, -15.8148766, 17.9809971
1: -6.0617056, 15.1005745, -4.5962033, 11.8650751, -17.9267807, 19.6967773
2: -5.1281843, 16.9207191, -3.8960257, 13.3130264, -18.4412079, 20.8167419
3: -6.0870829, 21.6393185, -4.5771689, 17.1330128, -23.2200947, 26.2164879
4: -5.0075822, 20.0201283, -3.8639328, 15.8723202, -20.8798943, 23.8840599

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4816465, upper bound: 27.5076902
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5316324, upper bound: 27.5314612
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.8966453, 13.8043938, -3.8869870, 13.7760038, -17.6726494, 17.6913815
1: -5.6560292, 14.1464357, -5.6416559, 14.1165829, -19.7726116, 19.7880878
2: -4.7809715, 15.9183598, -4.7686691, 15.8853073, -20.6662788, 20.6870289
3: -5.6777368, 20.3282623, -5.6634254, 20.2872963, -25.9650326, 25.9916878
4: -4.6884899, 18.8143234, -4.6772971, 18.7753258, -23.4638157, 23.4916210

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4964652, upper bound: 27.4682796
time: 0.62 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4376996, upper bound: 27.4376996
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.8966453, 13.8043938, -4.2554588, 14.9586115, -18.8552570, 18.0598526
1: -5.6560292, 14.1464357, -6.1641321, 15.3125620, -20.9685917, 20.3105640
2: -4.7809715, 15.9183598, -5.2145309, 17.1568089, -21.9377804, 21.1328907
3: -5.6777368, 20.3282623, -6.1905026, 21.9307594, -27.6084957, 26.5187607
4: -4.6884899, 18.8143234, -5.0852590, 20.2982121, -24.9867020, 23.8995819

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4682796, upper bound: 27.5161001
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4376996, upper bound: 27.4573345
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -3.8869870, 13.7760038, -18.0430756, 18.8799229
1: -6.1816254, 15.3483210, -5.6416559, 14.1165829, -20.2982082, 20.9899693
2: -5.2292953, 17.1967506, -4.7686691, 15.8853073, -21.1146030, 21.9654198
3: -6.2076254, 21.9800053, -5.6634254, 20.2872963, -26.4949226, 27.6434307
4: -5.0987215, 20.3454666, -4.6772971, 18.7753258, -23.8740463, 25.0227642

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5160999, upper bound: 27.4888279
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4573343, upper bound: 27.4582478
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.2554588, 14.9586115, -19.2256832, 19.2483978
1: -6.1816254, 15.3483210, -6.1641321, 15.3125620, -21.4941864, 21.5124474
2: -5.2292953, 17.1967506, -5.2145309, 17.1568089, -22.3861046, 22.4112816
3: -6.2076254, 21.9800053, -6.1905026, 21.9307594, -28.1383858, 28.1705017
4: -5.0987215, 20.3454666, -5.0852590, 20.2982121, -25.3969326, 25.4307251

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4962463, upper bound: 27.5041744
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5337099, upper bound: 27.5279455
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.8966453, 13.8043938, -4.2947645, 15.0208149, -18.9174595, 18.0991592
1: -5.6560292, 14.1464357, -6.1987748, 15.4000883, -21.0561123, 20.3452091
2: -4.7809715, 15.9183598, -5.2384167, 17.3021488, -22.0831203, 21.1567764
3: -5.6777368, 20.3282623, -6.2558775, 22.1487484, -27.8264847, 26.5841408
4: -4.6884899, 18.8143234, -5.1477151, 20.5328636, -25.2213535, 23.9620361

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5042966, upper bound: 27.4890451
time: 0.64 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5007862, upper bound: 27.5292833
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.8966453, 13.8043938, -4.6833072, 16.0610256, -19.9576664, 18.4877014
1: -5.6560292, 14.1464357, -6.7087297, 16.5008717, -22.1569004, 20.8551655
2: -4.7809715, 15.9183598, -5.6831689, 18.5184383, -23.2994099, 21.6015282
3: -5.6777368, 20.3282623, -6.7763977, 23.6576977, -29.3354340, 27.1046600
4: -4.6884899, 18.8143234, -5.5424156, 21.9553661, -26.6438560, 24.3567390

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5191132, upper bound: 27.5204707
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4894897, upper bound: 27.4658629
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.4899883, 15.5635271, -19.8305988, 19.4829254
1: -6.1816254, 15.3483210, -6.4238963, 15.9648161, -22.1464424, 21.7722111
2: -5.2292953, 17.1967506, -5.4360805, 17.9155674, -23.1448631, 22.6328278
3: -6.2076254, 21.9800053, -6.5009260, 22.9288368, -29.1364632, 28.4809303
4: -5.0987215, 20.3454666, -5.3208542, 21.2503090, -26.3490295, 25.6663208

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5417239, upper bound: 27.5218896
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5416908, upper bound: 27.5274600
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.8903704, 16.7688160, -21.0358887, 19.8833084
1: -6.1816254, 15.3483210, -6.9869814, 17.1864223, -23.3680477, 22.3353004
2: -5.2292953, 17.1967506, -5.8964348, 19.2150192, -24.4443150, 23.0931835
3: -6.2076254, 21.9800053, -7.0507531, 24.6006584, -30.8082848, 29.0307541
4: -5.0987215, 20.3454666, -5.7372026, 22.7977104, -27.8964310, 26.0826683

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5418298, upper bound: 27.5316136
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5416911, upper bound: 27.5274601
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -3.8199701, 13.6537867, -16.8768768, 15.4487371
1: -4.5962033, 11.8650751, -5.5330939, 13.9627600, -18.5589638, 17.3981686
2: -3.8960257, 13.3130264, -4.6683445, 15.6982718, -19.5942974, 17.9813690
3: -4.5771689, 17.1330128, -5.5629125, 20.1039257, -24.6810951, 22.6959229
4: -3.8639328, 15.8723202, -4.5901971, 18.5590172, -22.4229488, 20.4625111

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5202765, upper bound: 27.4920600
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5260200, upper bound: 27.5206469
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -4.2170448, 14.8499813, -18.0730705, 15.8458109
1: -4.5962033, 11.8650751, -6.0874748, 15.1848240, -19.7810268, 17.9525490
2: -3.8960257, 13.3130264, -5.1462145, 16.9952316, -20.8912582, 18.4592400
3: -4.5771689, 17.1330128, -6.1174002, 21.7649803, -26.3421497, 23.2504120
4: -3.8639328, 15.8723202, -5.0203681, 20.1216583, -23.9855919, 20.8926811

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B1_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5076902, upper bound: 27.4820904
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_B2

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5313274, upper bound: 27.5182523
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -4.1861086, 14.7579079, -17.9809971, 15.8148766
1: -4.5962033, 11.8650751, -6.0617056, 15.1005745, -19.6967773, 17.9267807
2: -3.8960257, 13.3130264, -5.1281843, 16.9207191, -20.8167419, 18.4412060
3: -4.5771689, 17.1330128, -6.0870829, 21.6393185, -26.2164879, 23.2200947
4: -3.8639328, 15.8723202, -5.0075822, 20.0201283, -23.8840599, 20.8798943

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5076902, upper bound: 27.4943692
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5314612, upper bound: 27.5316324
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.2947645, 15.0208149, -3.8966453, 13.8043938, -18.0991592, 18.9174595
1: -6.1987748, 15.4000883, -5.6560292, 14.1464357, -20.3452091, 21.0561123
2: -5.2384167, 17.3021488, -4.7809715, 15.9183598, -21.1567764, 22.0831203
3: -6.2558775, 22.1487484, -5.6777368, 20.3282623, -26.5841408, 27.8264847
4: -5.1477151, 20.5328636, -4.6884899, 18.8143234, -23.9620342, 25.2213535

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4884310, upper bound: 27.5042966
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5292833, upper bound: 27.5299744
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.6833072, 16.0610256, -3.8966453, 13.8043938, -18.4877014, 19.9576664
1: -6.7087297, 16.5008717, -5.6560292, 14.1464357, -20.8551655, 22.1569004
2: -5.6831689, 18.5184383, -4.7809715, 15.9183598, -21.6015282, 23.2994099
3: -6.7763977, 23.6576977, -5.6777368, 20.3282623, -27.1046600, 29.3354340
4: -5.5424156, 21.9553661, -4.6884899, 18.8143234, -24.3567390, 26.6438560

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5204707, upper bound: 27.5191132
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4658629, upper bound: 27.4894897
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.4899883, 15.5635271, -4.2670722, 14.9929380, -19.4829254, 19.8305988
1: -6.4238963, 15.9648161, -6.1816254, 15.3483210, -21.7722092, 22.1464424
2: -5.4360805, 17.9155674, -5.2292953, 17.1967506, -22.6328278, 23.1448631
3: -6.5009260, 22.9288368, -6.2076254, 21.9800053, -28.4809303, 29.1364632
4: -5.3208542, 21.2503090, -5.0987215, 20.3454666, -25.6663208, 26.3490295

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5184051, upper bound: 27.5417239
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5239756, upper bound: 27.5416908
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.8903704, 16.7688160, -4.2670722, 14.9929380, -19.8833084, 21.0358887
1: -6.9869814, 17.1864223, -6.1816254, 15.3483210, -22.3353004, 23.3680477
2: -5.8964348, 19.2150192, -5.2292953, 17.1967506, -23.0931816, 24.4443150
3: -7.0507531, 24.6006584, -6.2076254, 21.9800053, -29.0307541, 30.8082848
4: -5.7372026, 22.7977104, -5.0987215, 20.3454666, -26.0826683, 27.8964310

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5280380, upper bound: 27.5352379
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5239756, upper bound: 27.5328149
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -4.1121082, 14.5780020, -3.2230899, 11.6287680, -15.7408762, 17.8010902
1: -5.8801484, 14.9022427, -4.5962033, 11.8650751, -17.7452240, 19.4984455
2: -4.9502902, 16.7479477, -3.8960257, 13.3130264, -18.2633133, 20.6439743
3: -5.9691157, 21.5021744, -4.5771689, 17.1330128, -23.1021233, 26.0793419
4: -4.8908606, 19.8969631, -3.8639328, 15.8723202, -20.7631741, 23.7608929

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4986815, upper bound: 27.5251594
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5093733, upper bound: 27.5252117
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -4.4178681, 15.4399347, -3.2230899, 11.6287680, -16.0466347, 18.6630249
1: -6.3143077, 15.8106079, -4.5962033, 11.8650751, -18.1793823, 20.4068069
2: -5.3327088, 17.7244244, -3.8960257, 13.3130264, -18.6457348, 21.6204491
3: -6.3986797, 22.7401009, -4.5771689, 17.1330128, -23.5316925, 27.3172703
4: -5.2309446, 21.0288792, -3.8639328, 15.8723202, -21.1032639, 24.8928108

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4823696, upper bound: 27.5323671
time: 0.55 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5214012, upper bound: 27.5334800
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -4.7580576, 16.5340042, -3.2230899, 11.6287680, -16.3868256, 19.7570915
1: -6.8066359, 16.9095268, -4.5962033, 11.8650751, -18.6717110, 21.5057278
2: -5.7204256, 18.8901978, -3.8960257, 13.3130264, -19.0334511, 22.7862244
3: -6.8873081, 24.2441006, -4.5771689, 17.1330128, -24.0203171, 28.8212700
4: -5.5860939, 22.4078159, -3.8639328, 15.8723202, -21.4584045, 26.2717476

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5218180, upper bound: 27.5351364
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5324031, upper bound: 27.5351364
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.4899883, 15.5635271, -20.2841377, 20.7223186
1: -6.7455416, 16.6543579, -6.4238963, 15.9648161, -22.7103577, 23.0782547
2: -5.7037611, 18.6685505, -5.4360805, 17.9155674, -23.6193237, 24.1046314
3: -6.8226271, 23.8968391, -6.5009260, 22.9288368, -29.7514629, 30.3977661
4: -5.5644245, 22.1408634, -5.3208542, 21.2503090, -26.8147316, 27.4617176

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5385126, upper bound: 27.5227264
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5385056, upper bound: 27.5314100
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.8903704, 16.7688160, -21.4894218, 21.1226978
1: -6.7455416, 16.6543579, -6.9869814, 17.1864223, -23.9319649, 23.6413383
2: -5.7037611, 18.6685505, -5.8964348, 19.2150192, -24.9187775, 24.5649853
3: -6.8226271, 23.8968391, -7.0507531, 24.6006584, -31.4232845, 30.9475861
4: -5.5644245, 22.1408634, -5.7372026, 22.7977104, -28.3621330, 27.8780670

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5385308, upper bound: 27.5420784
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5399181, upper bound: 27.5418315
time: 0.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.22 seconds
NS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4930094, upper bound: 27.5208456
NS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5215962, upper bound: 27.5265891
NS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4823696, upper bound: 27.5202765
NS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5206469, upper bound: 27.5260200
NS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4992940, upper bound: 27.5321894
NS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4980277, upper bound: 27.5207970
NS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5321858, upper bound: 27.5324076
NS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5311005, upper bound: 27.5312373
NS_A1_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4816465, upper bound: 27.5076902
NS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5182523, upper bound: 27.5313274
NS_A1_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4816465, upper bound: 27.5076902
NS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5316324, upper bound: 27.5314612
NS_A1_B2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4964652, upper bound: 27.4682796
NS_A1_B2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4376996, upper bound: 27.4376996
NS_A1_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4682796, upper bound: 27.5161001
NS_A1_B2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4376996, upper bound: 27.4573345
NS_A1_B2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5160999, upper bound: 27.4888279
NS_A1_B2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4573343, upper bound: 27.4582478
NS_A1_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4962463, upper bound: 27.5041744
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5337099, upper bound: 27.5279455
NS_A1_B2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5042966, upper bound: 27.4890451
NS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5007862, upper bound: 27.5292833
NS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5191132, upper bound: 27.5204707
NS_A1_B2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4894897, upper bound: 27.4658629
NS_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5417239, upper bound: 27.5218896
NS_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5416908, upper bound: 27.5274600
NS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5418298, upper bound: 27.5316136
NS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5416911, upper bound: 27.5274601
NS_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5202765, upper bound: 27.4920600
NS_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5260200, upper bound: 27.5206469
NS_A2_B1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5076902, upper bound: 27.4820904
NS_A2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5313274, upper bound: 27.5182523
NS_A2_B1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5076902, upper bound: 27.4943692
NS_A2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5314612, upper bound: 27.5316324
NS_A2_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4884310, upper bound: 27.5042966
NS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5292833, upper bound: 27.5299744
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5204707, upper bound: 27.5191132
NS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4658629, upper bound: 27.4894897
NS_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5184051, upper bound: 27.5417239
NS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5239756, upper bound: 27.5416908
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5280380, upper bound: 27.5352379
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5239756, upper bound: 27.5328149
NS_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4986815, upper bound: 27.5251594
NS_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5093733, upper bound: 27.5252117
NS_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.4823696, upper bound: 27.5323671
NS_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5214012, upper bound: 27.5334800
NS_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5218180, upper bound: 27.5351364
NS_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5324031, upper bound: 27.5351364
NS_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5385126, upper bound: 27.5227264
NS_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5385056, upper bound: 27.5314100
NS_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5385308, upper bound: 27.5420784
NS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 3, lower bound: -27.5399181, upper bound: 27.5418315

## BFS NS instance: NS_A1_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -3.7937629, 13.5773649, -2.6493955, 10.1003323, -13.8940935, 16.2267609
1: -5.4732571, 13.8674803, -3.8986821, 10.2718744, -15.7451305, 17.7661629
2: -4.6151047, 15.5798492, -3.3160415, 11.5687923, -16.1838970, 18.8958893
3: -5.5074983, 19.9871922, -3.8553152, 14.8965769, -20.4040718, 23.8425045
4: -4.5403013, 18.4348907, -3.3364959, 13.8042555, -18.3445568, 21.7713852

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4869831, upper bound: 27.5158242
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4847860, upper bound: 27.5130146
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -3.7561455, 13.4668655, -2.6493955, 10.1003323, -13.8564749, 16.1162605
1: -5.4376044, 13.7654448, -3.8986821, 10.2718744, -15.7094784, 17.6641254
2: -4.5877476, 15.4795847, -3.3160415, 11.5687923, -16.1565380, 18.7956257
3: -5.4666896, 19.8337669, -3.8553152, 14.8965769, -20.3632660, 23.6890831
4: -4.5176473, 18.3004112, -3.3364959, 13.8042555, -18.3219013, 21.6369076

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4769126, upper bound: 27.5218345
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4847860, upper bound: 27.5124774
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -3.7937629, 13.5773649, -3.2230899, 11.6287680, -15.4225302, 16.8004551
1: -5.4732571, 13.8674803, -4.5962033, 11.8650751, -17.3383331, 18.4636822
2: -4.6151047, 15.5798492, -3.8960257, 13.3130264, -17.9281311, 19.4758759
3: -5.5074983, 19.9871922, -4.5771689, 17.1330128, -22.6405087, 24.5643597
4: -4.5403013, 18.4348907, -3.8639328, 15.8723202, -20.4126186, 22.2988205

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4866340, upper bound: 27.5157480
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4844369, upper bound: 27.5129384
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3.7561455, 13.4668655, -3.2230899, 11.6287680, -15.3849125, 16.6899548
1: -5.4376044, 13.7654448, -4.5962033, 11.8650751, -17.3026791, 18.3616447
2: -4.5877476, 15.4795847, -3.8960257, 13.3130264, -17.9007721, 19.3756104
3: -5.4666896, 19.8337669, -4.5771689, 17.1330128, -22.5997028, 24.4109364
4: -4.5176473, 18.3004112, -3.8639328, 15.8723202, -20.3899612, 22.1643429

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5176003, upper bound: 27.5217584
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4958082, upper bound: 27.5124013
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -3.6922674, 13.4013319, -2.6493955, 10.1003323, -13.7925978, 16.0507278
1: -5.2974415, 13.6524391, -3.8986821, 10.2718744, -15.5693159, 17.5511208
2: -4.4381924, 15.2915983, -3.3160415, 11.5687923, -16.0069847, 18.6076374
3: -5.3446608, 19.6805878, -3.8553152, 14.8965769, -20.2412357, 23.5359039
4: -4.3785963, 18.1139355, -3.3364959, 13.8042555, -18.1828518, 21.4504318

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4706088, upper bound: 27.5024798
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5147899, upper bound: 27.5282608
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -4.2060833, 14.8103809, -2.6493955, 10.1003323, -14.3064156, 17.4597759
1: -6.0292349, 15.1190653, -3.8986821, 10.2718744, -16.3011093, 19.0177479
2: -5.0859008, 16.9169827, -3.3160415, 11.5687923, -16.6546917, 20.2330189
3: -6.0686307, 21.6695690, -3.8553152, 14.8965769, -20.9652042, 25.5248833
4: -4.9631276, 20.0034618, -3.3364959, 13.8042555, -18.7673817, 23.3399582

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4768208, upper bound: 27.5041867
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5150622, upper bound: 27.5272430
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -3.6635847, 13.3049660, -2.6493955, 10.1003323, -13.7639160, 15.9543610
1: -5.2731128, 13.5671997, -3.8986821, 10.2718744, -15.5449858, 17.4658813
2: -4.4237947, 15.2094803, -3.3160415, 11.5687923, -15.9925871, 18.5255203
3: -5.3172736, 19.5534477, -3.8553152, 14.8965769, -20.2138462, 23.4087620
4: -4.3698683, 18.0040493, -3.3364959, 13.8042555, -18.1741238, 21.3405457

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4706088, upper bound: 27.5025066
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5291952, upper bound: 27.5287901
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -4.1477933, 14.6454697, -2.6493955, 10.1003323, -14.2481222, 17.2948647
1: -5.9666500, 14.9592514, -3.8986821, 10.2718744, -16.2385254, 18.8579311
2: -5.0351315, 16.7523537, -3.3160415, 11.5687923, -16.6039219, 20.0683918
3: -6.0022440, 21.4368610, -3.8553152, 14.8965769, -20.8988171, 25.2921753
4: -4.9214268, 19.7988586, -3.3364959, 13.8042555, -18.7256813, 23.1353550

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4888332, upper bound: 27.5041867
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271296, upper bound: 27.5269979
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -4.1170735, 14.6569815, -3.2230899, 11.6287680, -15.7458410, 17.8800716
1: -5.9346251, 14.9519119, -4.5962033, 11.8650751, -17.7996998, 19.5481148
2: -4.9900007, 16.7207108, -3.8960257, 13.3130264, -18.3030224, 20.6167336
3: -5.9762187, 21.4692192, -4.5771689, 17.1330128, -23.1092262, 26.0463886
4: -4.8850317, 19.7887573, -3.8639328, 15.8723202, -20.7573471, 23.6526871

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4702597, upper bound: 27.5281846
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5147132, upper bound: 27.5271669
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -4.0811014, 14.5612078, -3.2230899, 11.6287680, -15.7098675, 17.7842979
1: -5.9024377, 14.8641195, -4.5962033, 11.8650751, -17.7675133, 19.4603233
2: -4.9668932, 16.6364784, -3.8960257, 13.3130264, -18.2799187, 20.5325050
3: -5.9401455, 21.3393211, -4.5771689, 17.1330128, -23.0731583, 25.9164886
4: -4.8693376, 19.6772442, -3.8639328, 15.8723202, -20.7416515, 23.5411758

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5288461, upper bound: 27.5287139
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5267806, upper bound: 27.5269218
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.1616135, 14.7958717, -4.2554588, 14.9586115, -19.1202221, 19.0513306
1: -6.0217419, 15.1112652, -6.1641321, 15.3125620, -21.3343048, 21.2753944
2: -5.0678825, 16.9106216, -5.2145309, 17.1568089, -22.2246914, 22.1251526
3: -6.0606642, 21.6794033, -6.1905026, 21.9307594, -27.9914246, 27.8698997
4: -4.9605665, 20.0012703, -5.0852590, 20.2982121, -25.2587776, 25.0865288

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5273737, upper bound: 27.5335334
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4667987, upper bound: 27.5376803
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3.8966453, 13.8043938, -4.2013507, 14.8456116, -18.7422562, 18.0057449
1: -5.6560292, 14.1464357, -6.0548878, 15.1893959, -20.8454247, 20.2013206
2: -4.7809715, 15.9183598, -5.1086121, 17.0443916, -21.8253632, 21.0269718
3: -5.6777368, 20.3282623, -6.1252065, 21.8839588, -27.5616951, 26.4534683
4: -4.6884899, 18.8143234, -5.0339437, 20.2378387, -24.9263287, 23.8482666

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5132512, upper bound: 27.5212161
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5132512, upper bound: 27.5292835
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8465097, 13.6745157, -4.6833072, 16.0610256, -19.9075298, 18.3578224
1: -5.5826459, 14.0074348, -6.7087297, 16.5008717, -22.0835171, 20.7161636
2: -4.7157102, 15.7597399, -5.6831689, 18.5184383, -23.2341480, 21.4429092
3: -5.6075134, 20.1370182, -6.7763977, 23.6576977, -29.2652092, 26.9134159
4: -4.6303511, 18.6289520, -5.5424156, 21.9553661, -26.5857162, 24.1713676

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5051627, upper bound: 27.5015179
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4640054, upper bound: 27.4594100
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.0129819, 14.2722778, -18.5393505, 19.0059185
1: -6.1816254, 15.3483210, -5.7867179, 14.6021156, -20.7837410, 21.1350327
2: -5.2292953, 17.1967506, -4.8859429, 16.4078541, -21.6371498, 22.0826931
3: -6.2076254, 21.9800053, -5.8579979, 21.0537167, -27.2613411, 27.8380013
4: -5.0987215, 20.3454666, -4.8296361, 19.4827900, -24.5815105, 25.1751022

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5215937, upper bound: 27.4825641
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5309146, upper bound: 27.5029662
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5391862, upper bound: 27.5154066
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.3852854, 15.2826328, -19.5497055, 19.3782234
1: -6.1816254, 15.3483210, -6.2835150, 15.6714468, -21.8530731, 21.6318283
2: -5.2292953, 17.1967506, -5.3162026, 17.5887966, -22.8180923, 22.5129528
3: -6.2076254, 21.9800053, -6.3600073, 22.5200157, -28.7276421, 28.3400116
4: -5.0987215, 20.3454666, -5.2129455, 20.8621902, -25.9609108, 25.5584126

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5216402, upper bound: 27.4887660
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5076469, upper bound: 27.4825937
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.8483875, 13.8090839, -4.8903704, 16.7688160, -20.6172009, 18.6994553
1: -5.6051860, 14.0970688, -6.9869814, 17.1864223, -22.7916088, 21.0840492
2: -4.7509747, 15.8073130, -5.8964348, 19.2150192, -23.9659939, 21.7037468
3: -5.6255231, 20.2453957, -7.0507531, 24.6006584, -30.2261810, 27.2961445
4: -4.6657948, 18.7160625, -5.7372026, 22.7977104, -27.4635048, 24.4532661

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5282164, upper bound: 27.5281763
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5360026, upper bound: 27.5133874
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4989281, upper bound: 27.5032751
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5408930, upper bound: 27.5384039
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.2006721, 14.8031902, -4.8903704, 16.7688160, -20.9694843, 19.6935616
1: -6.0869379, 15.1494980, -6.9869814, 17.1864223, -23.2733555, 22.1364784
2: -5.1482048, 16.9752731, -5.8964348, 19.2150192, -24.3632240, 22.8717060
3: -6.1115742, 21.7016144, -7.0507531, 24.6006584, -30.7122326, 28.7523651
4: -5.0243750, 20.0811024, -5.7372026, 22.7977104, -27.8220863, 25.8183060

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5281908, upper bound: 27.5240909
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5091244, upper bound: 27.4864111
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -3.7937629, 13.5773649, -16.8004551, 15.4225302
1: -4.5962033, 11.8650751, -5.4732571, 13.8674803, -18.4636822, 17.3383331
2: -3.8960257, 13.3130264, -4.6151047, 15.5798492, -19.4758759, 17.9281311
3: -4.5771689, 17.1330128, -5.5074983, 19.9871922, -24.5643616, 22.6405087
4: -3.8639328, 15.8723202, -4.5403013, 18.4348907, -22.2988205, 20.4126186

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_B2_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5157480, upper bound: 27.4866340
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5129384, upper bound: 27.4844369
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -3.7561455, 13.4668655, -16.6899548, 15.3849125
1: -4.5962033, 11.8650751, -5.4376044, 13.7654448, -18.3616447, 17.3026791
2: -3.8960257, 13.3130264, -4.5877476, 15.4795847, -19.3756104, 17.9007721
3: -4.5771689, 17.1330128, -5.4666896, 19.8337669, -24.4109364, 22.5997028
4: -3.8639328, 15.8723202, -4.5176473, 18.3004112, -22.1643429, 20.3899612

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217584, upper bound: 27.5176003
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5124013, upper bound: 27.4958082
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -4.1170735, 14.6569815, -17.8800716, 15.7458410
1: -4.5962033, 11.8650751, -5.9346251, 14.9519119, -19.5481148, 17.7996998
2: -3.8960257, 13.3130264, -4.9900007, 16.7207108, -20.6167336, 18.3030205
3: -4.5771689, 17.1330128, -5.9762187, 21.4692192, -26.0463867, 23.1092262
4: -3.8639328, 15.8723202, -4.8850317, 19.7887573, -23.6526871, 20.7573490

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_B1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5281846, upper bound: 27.5144408
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271669, upper bound: 27.5147132
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -4.0811014, 14.5612078, -17.7842979, 15.7098675
1: -4.5962033, 11.8650751, -5.9024377, 14.8641195, -19.4603214, 17.7675133
2: -3.8960257, 13.3130264, -4.9668932, 16.6364784, -20.5325050, 18.2799187
3: -4.5771689, 17.1330128, -5.9401455, 21.3393211, -25.9164886, 23.0731583
4: -3.8639328, 15.8723202, -4.8693376, 19.6772442, -23.5411758, 20.7416515

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5287139, upper bound: 27.5288461
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5269218, upper bound: 27.5267806
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -4.2013507, 14.8456116, -3.8966453, 13.8043938, -18.0057449, 18.7422562
1: -6.0548878, 15.1893959, -5.6560292, 14.1464357, -20.2013187, 20.8454247
2: -5.1086121, 17.0443916, -4.7809715, 15.9183598, -21.0269718, 21.8253632
3: -6.1252065, 21.8839588, -5.6777368, 20.3282623, -26.4534683, 27.5616951
4: -5.0339437, 20.2378387, -4.6884899, 18.8143234, -23.8482666, 24.9263287

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5129343, upper bound: 27.5132512
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5210520, upper bound: 27.5299744
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.6833072, 16.0610256, -3.8465097, 13.6745157, -18.3578224, 19.9075298
1: -6.7087297, 16.5008717, -5.5826459, 14.0074348, -20.7161636, 22.0835171
2: -5.6831689, 18.5184383, -4.7157102, 15.7597399, -21.4429092, 23.2341480
3: -6.7763977, 23.6576977, -5.6075134, 20.1370182, -26.9134159, 29.2652092
4: -5.5424156, 21.9553661, -4.6303511, 18.6289520, -24.1713676, 26.5857162

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5015179, upper bound: 27.5051627
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4594100, upper bound: 27.4640054
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -4.0129819, 14.2722778, -4.2670722, 14.9929380, -19.0059185, 18.5393505
1: -5.7867179, 14.6021156, -6.1816254, 15.3483210, -21.1350327, 20.7837410
2: -4.8859429, 16.4078541, -5.2292953, 17.1967506, -22.0826931, 21.6371498
3: -5.8579979, 21.0537167, -6.2076254, 21.9800053, -27.8380013, 27.2613411
4: -4.8296361, 19.4827900, -5.0987215, 20.3454666, -25.1751022, 24.5815105

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4825641, upper bound: 27.5215936
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4802686, upper bound: 27.5196484
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5154066, upper bound: 27.5391863
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -4.3852854, 15.2826328, -4.2670722, 14.9929380, -19.3782215, 19.5497055
1: -6.2835150, 15.6714468, -6.1816254, 15.3483210, -21.6318283, 21.8530731
2: -5.3162026, 17.5887966, -5.2292953, 17.1967506, -22.5129528, 22.8180923
3: -6.3600073, 22.5200157, -6.2076254, 21.9800053, -28.3400116, 28.7276421
4: -5.2129455, 20.8621902, -5.0987215, 20.3454666, -25.5584106, 25.9609108

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4887660, upper bound: 27.5216402
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4825937, upper bound: 27.5076469
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.8903704, 16.7688160, -3.8483875, 13.8090839, -18.6994553, 20.6172009
1: -6.9869814, 17.1864223, -5.6051860, 14.0970688, -21.0840492, 22.7916088
2: -5.8964348, 19.2150192, -4.7509747, 15.8073130, -21.7037468, 23.9659939
3: -7.0507531, 24.6006584, -5.6255231, 20.2453957, -27.2961445, 30.2261810
4: -5.7372026, 22.7977104, -4.6657948, 18.7160625, -24.4532661, 27.4635048

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5268826, upper bound: 27.5162570
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5104238, upper bound: 27.5189991
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5136965, upper bound: 27.5002052
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386653, upper bound: 27.5319472
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.8903704, 16.7688160, -4.2006721, 14.8031902, -19.6935616, 20.9694843
1: -6.9869814, 17.1864223, -6.0869379, 15.1494980, -22.1364784, 23.2733555
2: -5.8964348, 19.2150192, -5.1482048, 16.9752731, -22.8717060, 24.3632240
3: -7.0507531, 24.6006584, -6.1115742, 21.7016144, -28.7523651, 30.7122326
4: -5.7372026, 22.7977104, -5.0243750, 20.0811024, -25.8183060, 27.8220863

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5225435, upper bound: 27.5218573
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826782, upper bound: 27.5011701
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -4.1388130, 14.6604109, -3.2230899, 11.6287680, -15.7675810, 17.8835011
1: -5.9016218, 14.9764757, -4.5962033, 11.8650751, -17.7666969, 19.5726795
2: -4.9702492, 16.8187313, -3.8960257, 13.3130264, -18.2832756, 20.7147560
3: -5.9883618, 21.6023979, -4.5771689, 17.1330128, -23.1213722, 26.1795673
4: -4.9062600, 19.9984188, -3.8639328, 15.8723202, -20.7785721, 23.8623505

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4777494, upper bound: 27.5130376
time: 0.55 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4971038, upper bound: 27.5223832
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -4.0310955, 14.3473444, -3.2230899, 11.6287680, -15.6598616, 17.5704346
1: -5.7621255, 14.6591921, -4.5962033, 11.8650751, -17.6271992, 19.2553959
2: -4.8494329, 16.4783955, -3.8960257, 13.3130264, -18.1624546, 20.3744202
3: -5.8512173, 21.1671925, -4.5771689, 17.1330128, -22.9842300, 25.7443619
4: -4.8007731, 19.5782337, -3.8639328, 15.8723202, -20.6730881, 23.4421654

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5047887, upper bound: 27.5205857
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5059324, upper bound: 27.5224305
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -4.4659195, 15.5693312, -3.2230899, 11.6287680, -16.0946884, 18.7924213
1: -6.3684106, 15.9348230, -4.5962033, 11.8650751, -18.2334843, 20.5310211
2: -5.3735919, 17.8539467, -3.8960257, 13.3130264, -18.6866169, 21.7499733
3: -6.4544306, 22.9067936, -4.5771689, 17.1330128, -23.5874443, 27.4839630
4: -5.2639656, 21.1983070, -3.8639328, 15.8723202, -21.1362820, 25.0622368

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4936334, upper bound: 27.4987556
time: 0.64 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4936335, upper bound: 27.5315420
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -4.3332319, 15.2043400, -3.2230899, 11.6287680, -15.9619999, 18.4274292
1: -6.1916523, 15.5625191, -4.5962033, 11.8650751, -18.0567265, 20.1587219
2: -5.2286344, 17.4494247, -3.8960257, 13.3130264, -18.5416584, 21.3454514
3: -6.2767315, 22.3976593, -4.5771689, 17.1330128, -23.4097443, 26.9748287
4: -5.1382856, 20.7020950, -3.8639328, 15.8723202, -21.0106010, 24.5660267

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5027442, upper bound: 27.4998684
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5027443, upper bound: 27.5333412
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -4.8102126, 16.6795921, -3.2230899, 11.6287680, -16.4389782, 19.9026814
1: -6.8670340, 17.0531311, -4.5962033, 11.8650751, -18.7321091, 21.6493340
2: -5.7668905, 19.0408039, -3.8960257, 13.3130264, -19.0799141, 22.9368286
3: -6.9508572, 24.4394665, -4.5771689, 17.1330128, -24.0838699, 29.0166359
4: -5.6242981, 22.6034756, -3.8639328, 15.8723202, -21.4966164, 26.4674072

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5031611, upper bound: 27.5015248
time: 0.50 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4734031, upper bound: 27.5351364
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -4.6728821, 16.2983379, -3.2230899, 11.6287680, -16.3016491, 19.5214252
1: -6.6850424, 16.6613579, -4.5962033, 11.8650751, -18.5501175, 21.2575607
2: -5.6154485, 18.6152859, -3.8960257, 13.3130264, -18.9284744, 22.5113106
3: -6.7668982, 23.9010849, -4.5771689, 17.1330128, -23.8999100, 28.4782543
4: -5.4928355, 22.0815315, -3.8639328, 15.8723202, -21.3651524, 25.9454632

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5137461, upper bound: 27.5015248
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5137462, upper bound: 27.5351364
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.0129819, 14.2722778, -18.9928875, 20.2453060
1: -6.7455416, 16.6543579, -5.7867179, 14.6021156, -21.3476562, 22.4410763
2: -5.7037611, 18.6685505, -4.8859429, 16.4078541, -22.1116123, 23.5544930
3: -6.8226271, 23.8968391, -5.8579979, 21.0537167, -27.8763428, 29.7548370
4: -5.5644245, 22.1408634, -4.8296361, 19.4827900, -25.0472126, 26.9704990

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5281275, upper bound: 27.5073686
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5310373, upper bound: 27.5102859
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5310375, upper bound: 27.5227264
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.3852854, 15.2826328, -20.0032425, 20.6176109
1: -6.7455416, 16.6543579, -6.2835150, 15.6714468, -22.4169884, 22.9378738
2: -5.7037611, 18.6685505, -5.3162026, 17.5887966, -23.2925568, 23.9847527
3: -6.8226271, 23.8968391, -6.3600073, 22.5200157, -29.3426418, 30.2568436
4: -5.5644245, 22.1408634, -5.2129455, 20.8621902, -26.4266129, 27.3538074

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5285432, upper bound: 27.5163495
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5130149, upper bound: 27.5122413
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.2066560, 14.8591013, -4.8903704, 16.7688160, -20.9754696, 19.7494717
1: -6.0614095, 15.2031517, -6.9869814, 17.1864223, -23.2478313, 22.1901321
2: -5.1123199, 17.0585728, -5.8964348, 19.2150192, -24.3273373, 22.9550056
3: -6.1311889, 21.9030704, -7.0507531, 24.6006584, -30.7318459, 28.9538212
4: -5.0373154, 20.2538452, -5.7372026, 22.7977104, -27.8350239, 25.9910469

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5211652, upper bound: 27.5296025
time: 0.68 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5162857, upper bound: 27.5303202
time: 0.64 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5191368, upper bound: 27.5150119
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5385308, upper bound: 27.5410573
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5385308, upper bound: 27.5418315
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.6036935, 15.9168310, -4.8903704, 16.7688160, -21.3725071, 20.8072014
1: -6.5887237, 16.3248081, -6.9869814, 17.1864223, -23.7751465, 23.3117905
2: -5.5700936, 18.3021870, -5.8964348, 19.2150192, -24.7851124, 24.1986160
3: -6.6645889, 23.4384232, -7.0507531, 24.6006584, -31.2652473, 30.4891720
4: -5.4441853, 21.7062130, -5.7372026, 22.7977104, -28.2418938, 27.4434166

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297880, upper bound: 27.5405543
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297880, upper bound: 27.5368476
time: 0.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.21 seconds
NS_A1_B1_A1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4869831, upper bound: 27.5158242
NS_A1_B1_A1_B1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4847860, upper bound: 27.5130146
NS_A1_B1_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4769126, upper bound: 27.5218345
NS_A1_B1_A1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4847860, upper bound: 27.5124774
NS_A1_B1_A1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4866340, upper bound: 27.5157480
NS_A1_B1_A1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4844369, upper bound: 27.5129384
NS_A1_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5176003, upper bound: 27.5217584
NS_A1_B1_A1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4958082, upper bound: 27.5124013
NS_A1_B1_A2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4706088, upper bound: 27.5024798
NS_A1_B1_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5147899, upper bound: 27.5282608
NS_A1_B1_A2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4768208, upper bound: 27.5041867
NS_A1_B1_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5150622, upper bound: 27.5272430
NS_A1_B1_A2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4706088, upper bound: 27.5025066
NS_A1_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5291952, upper bound: 27.5287901
NS_A1_B1_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4888332, upper bound: 27.5041867
NS_A1_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5271296, upper bound: 27.5269979
NS_A1_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4702597, upper bound: 27.5281846
NS_A1_B1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5147132, upper bound: 27.5271669
NS_A1_B1_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5288461, upper bound: 27.5287139
NS_A1_B1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5267806, upper bound: 27.5269218
NS_A1_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5273737, upper bound: 27.5335334
NS_A1_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4667987, upper bound: 27.5376803
NS_A1_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5132512, upper bound: 27.5212161
NS_A1_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5132512, upper bound: 27.5292835
NS_A1_B2_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5051627, upper bound: 27.5015179
NS_A1_B2_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4640054, upper bound: 27.4594100
NS_A1_B2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5309146, upper bound: 27.5029662
NS_A1_B2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5391862, upper bound: 27.5154066
NS_A1_B2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5216402, upper bound: 27.4887660
NS_A1_B2_B2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5076469, upper bound: 27.4825937
NS_A1_B2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4989281, upper bound: 27.5032751
NS_A1_B2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5408930, upper bound: 27.5384039
NS_A1_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5281908, upper bound: 27.5240909
NS_A1_B2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5091244, upper bound: 27.4864111
NS_A2_B1_A1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5157480, upper bound: 27.4866340
NS_A2_B1_A1_B1_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5129384, upper bound: 27.4844369
NS_A2_B1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5217584, upper bound: 27.5176003
NS_A2_B1_A1_B1_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5124013, upper bound: 27.4958082
NS_A2_B1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5281846, upper bound: 27.5144408
NS_A2_B1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5271669, upper bound: 27.5147132
NS_A2_B1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5287139, upper bound: 27.5288461
NS_A2_B1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5269218, upper bound: 27.5267806
NS_A2_B1_A2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5129343, upper bound: 27.5132512
NS_A2_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5210520, upper bound: 27.5299744
NS_A2_B1_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5015179, upper bound: 27.5051627
NS_A2_B1_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4594100, upper bound: 27.4640054
NS_A2_B1_A2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4802686, upper bound: 27.5196484
NS_A2_B1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5154066, upper bound: 27.5391863
NS_A2_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4887660, upper bound: 27.5216402
NS_A2_B1_A2_B2_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4825937, upper bound: 27.5076469
NS_A2_B1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5136965, upper bound: 27.5002052
NS_A2_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5386653, upper bound: 27.5319472
NS_A2_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5225435, upper bound: 27.5218573
NS_A2_B1_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4826782, upper bound: 27.5011701
NS_A2_B2_B1_A1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4777494, upper bound: 27.5130376
NS_A2_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4971038, upper bound: 27.5223832
NS_A2_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5047887, upper bound: 27.5205857
NS_A2_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5059324, upper bound: 27.5224305
NS_A2_B2_B1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4936334, upper bound: 27.4987556
NS_A2_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4936335, upper bound: 27.5315420
NS_A2_B2_B1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5027442, upper bound: 27.4998684
NS_A2_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5027443, upper bound: 27.5333412
NS_A2_B2_B1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5031611, upper bound: 27.5015248
NS_A2_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.4734031, upper bound: 27.5351364
NS_A2_B2_B1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5137461, upper bound: 27.5015248
NS_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5137462, upper bound: 27.5351364
NS_A2_B2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5310373, upper bound: 27.5102859
NS_A2_B2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5310375, upper bound: 27.5227264
NS_A2_B2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5285432, upper bound: 27.5163495
NS_A2_B2_B2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5130149, upper bound: 27.5122413
NS_A2_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5385308, upper bound: 27.5410573
NS_A2_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5385308, upper bound: 27.5418315
NS_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5297880, upper bound: 27.5405543
NS_A2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.21
Output dim: 3, lower bound: -27.5297880, upper bound: 27.5368476

## BFS NS instance: NS_A1_B1_A1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -3.2330034, 12.0068417, -2.6493955, 10.1003323, -13.3333349, 14.6562366
1: -4.6482325, 12.2217264, -3.8986821, 10.2718744, -14.9201069, 16.1204090
2: -3.8814385, 13.7559328, -3.3160415, 11.5687923, -15.4502296, 17.0719719
3: -4.6916032, 17.7408600, -3.8553152, 14.8965769, -19.5881786, 21.5961761
4: -3.8766093, 16.2748146, -3.3364959, 13.8042555, -17.6808643, 19.6113110

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4985280, upper bound: 27.4868007
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4985281, upper bound: 27.5148312
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -3.2330034, 12.0068417, -3.2230899, 11.6287680, -14.8617706, 15.2299318
1: -4.6482325, 12.2217264, -4.5962033, 11.8650751, -16.5133076, 16.8179283
2: -3.8814385, 13.7559328, -3.8960257, 13.3130264, -17.1944637, 17.6519585
3: -4.6916032, 17.7408600, -4.5771689, 17.1330128, -21.8246117, 22.3180294
4: -3.8766093, 16.2748146, -3.8639328, 15.8723202, -19.7489243, 20.1387482

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4985041, upper bound: 27.4868089
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4675379, upper bound: 27.5148312
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -3.5948901, 13.2204552, -2.6493955, 10.1003323, -13.6952229, 15.8698502
1: -5.1536551, 13.4331245, -3.8986821, 10.2718744, -15.4255285, 17.3318062
2: -4.2864428, 15.0356531, -3.3160415, 11.5687923, -15.8552351, 18.3516941
3: -5.2128172, 19.4023666, -3.8553152, 14.8965769, -20.1093941, 23.2576809
4: -4.2474804, 17.8104210, -3.3364959, 13.8042555, -18.0517330, 21.1469173

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4675619, upper bound: 27.4932271
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4953688, upper bound: 27.5282606
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -4.1081610, 14.6355429, -2.6493955, 10.1003323, -14.2084923, 17.2849388
1: -5.8841381, 14.9051600, -3.8986821, 10.2718744, -16.1560135, 18.8038387
2: -4.9335394, 16.6662235, -3.3160415, 11.5687923, -16.5023308, 19.9822617
3: -5.9328828, 21.4054794, -3.8553152, 14.8965769, -20.8294601, 25.2607937
4: -4.8342190, 19.7068176, -3.3364959, 13.8042555, -18.6384735, 23.0433140

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4956412, upper bound: 27.4922093
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4956412, upper bound: 27.5272431
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -3.5622625, 13.1057920, -2.6493955, 10.1003323, -13.6625938, 15.7551880
1: -5.1223269, 13.3287449, -3.8986821, 10.2718744, -15.3942013, 17.2274265
2: -4.2653337, 14.9308376, -3.3160415, 11.5687923, -15.8341255, 18.2468777
3: -5.1733098, 19.2479973, -3.8553152, 14.8965769, -20.0698853, 23.1033115
4: -4.2304645, 17.6688519, -3.3364959, 13.8042555, -18.0347137, 21.0053444

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5097740, upper bound: 27.4937563
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5097740, upper bound: 27.5287901
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -4.0363760, 14.4265327, -2.6493955, 10.1003323, -14.1367083, 17.0759258
1: -5.7989521, 14.6991920, -3.8986821, 10.2718744, -16.0708275, 18.5978737
2: -4.8629599, 16.4490166, -3.3160415, 11.5687923, -16.4317513, 19.7650547
3: -5.8439355, 21.1054325, -3.8553152, 14.8965769, -20.7405109, 24.9607468
4: -4.7735724, 19.4315796, -3.3364959, 13.8042555, -18.5778275, 22.7680759

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5077085, upper bound: 27.4919642
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5077085, upper bound: 27.5269980
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -3.5987697, 13.2322645, -3.2230899, 11.6287680, -15.2275372, 16.4553547
1: -5.1593447, 13.4451246, -4.5962033, 11.8650751, -17.0244198, 18.0413208
2: -4.2915444, 15.0486994, -3.8960257, 13.3130264, -17.6045685, 18.9447231
3: -5.2183175, 19.4191303, -4.5771689, 17.1330128, -22.3513260, 23.9962997
4: -4.2520895, 17.8261108, -3.8639328, 15.8723202, -20.1244049, 21.6900425

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4953446, upper bound: 27.4932352
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4953449, upper bound: 27.5281847
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -4.1084003, 14.6362095, -3.2230899, 11.6287680, -15.7371674, 17.8592987
1: -5.8844862, 14.9058580, -4.5962033, 11.8650751, -17.7495613, 19.5020618
2: -4.9338779, 16.6669903, -3.8960257, 13.3130264, -18.2469044, 20.5630150
3: -5.9332085, 21.4064541, -4.5771689, 17.1330128, -23.0662193, 25.9836216
4: -4.8345366, 19.7077599, -3.8639328, 15.8723202, -20.7068501, 23.5716915

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4573755, upper bound: 27.4922174
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4956172, upper bound: 27.5271670
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -3.5670896, 13.1203718, -3.2230899, 11.6287680, -15.1958580, 16.3434620
1: -5.1293883, 13.3436155, -4.5962033, 11.8650751, -16.9944630, 17.9398155
2: -4.2717123, 14.9469910, -3.8960257, 13.3130264, -17.5847378, 18.8430176
3: -5.1802359, 19.2688313, -4.5771689, 17.1330128, -22.3132458, 23.8460007
4: -4.2362761, 17.6883545, -3.8639328, 15.8723202, -20.1085949, 21.5522842

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5097500, upper bound: 27.4937645
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5097500, upper bound: 27.5287140
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -4.0363760, 14.4265327, -3.2230899, 11.6287680, -15.6651440, 17.6496201
1: -5.7989521, 14.6991920, -4.5962033, 11.8650751, -17.6640282, 19.2953949
2: -4.8629599, 16.4490166, -3.8960257, 13.3130264, -18.1759853, 20.3450394
3: -5.8439355, 21.1054325, -4.5771689, 17.1330128, -22.9769478, 25.6826000
4: -4.7735724, 19.4315796, -3.8639328, 15.8723202, -20.6458874, 23.2955112

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5076845, upper bound: 27.4919724
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5076846, upper bound: 27.5269219
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2.7557993, 10.6135054, -4.2554588, 14.9586115, -17.7144051, 14.8689632
1: -4.0050797, 10.7228260, -6.1641321, 15.3125620, -19.3176422, 16.8869572
2: -3.3896453, 11.9997158, -5.2145309, 17.1568089, -20.5464535, 17.2142467
3: -3.9829648, 15.5411358, -6.1905026, 21.9307594, -25.9137230, 21.7316341
4: -3.4068902, 14.2654495, -5.0852590, 20.2982121, -23.7051029, 19.3507080

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5077225, upper bound: 27.4960736
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5077227, upper bound: 27.5305676
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -4.1501856, 14.7619257, -4.2554588, 14.9586115, -19.1087971, 19.0173836
1: -6.0045290, 15.0758657, -6.1641321, 15.3125620, -21.3170910, 21.2399960
2: -5.0533237, 16.8712635, -5.2145309, 17.1568089, -22.2101326, 22.0857944
3: -6.0436320, 21.6307240, -6.1905026, 21.9307594, -27.9743881, 27.8212204
4: -4.9472756, 19.9544792, -5.0852590, 20.2982121, -25.2454834, 25.0397377

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4658738, upper bound: 27.4962459
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4658738, upper bound: 27.5242628
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3.5618644, 12.9171572, -4.2013507, 14.8456116, -18.4074745, 17.1185055
1: -5.1606417, 13.1869688, -6.0548878, 15.1893959, -20.3500328, 19.2418537
2: -4.3412910, 14.8576298, -5.1086121, 17.0443916, -21.3856831, 19.9662418
3: -5.1961222, 19.0373878, -6.1252065, 21.8839588, -27.0800800, 25.1625938
4: -4.2972727, 17.5959225, -5.0339437, 20.2378387, -24.5351105, 22.6298656

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4775321, upper bound: 27.5212161
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4802362, upper bound: 27.5212161
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8377326, 13.7042408, -4.2013507, 14.8456116, -18.6833439, 17.9055882
1: -5.5586572, 14.0156584, -6.0548878, 15.1893959, -20.7480488, 20.0705452
2: -4.6897707, 15.7574348, -5.1086121, 17.0443916, -21.7341614, 20.8660469
3: -5.5885458, 20.1776047, -6.1252065, 21.8839588, -27.4725037, 26.3028107
4: -4.6097212, 18.6277122, -5.0339437, 20.2378387, -24.8475609, 23.6616554

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4802362, upper bound: 27.4809777
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4782182, upper bound: 27.4809777
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -3.6487381, 13.3095112, -17.5765839, 18.6416740
1: -6.1816254, 15.3483210, -5.2600460, 13.5622673, -19.7438927, 20.6083641
2: -5.2292953, 17.1967506, -4.4125900, 15.2669535, -20.4962482, 21.6093388
3: -6.2076254, 21.9800053, -5.3424973, 19.6558323, -25.8634567, 27.3225002
4: -5.0987215, 20.3454666, -4.4083128, 18.1566029, -23.2553234, 24.7537785

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5304330, upper bound: 27.4977980
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5305778, upper bound: 27.5029216
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -3.9486008, 14.1665201, -18.4335918, 18.9415359
1: -6.1816254, 15.3483210, -5.6878052, 14.4651852, -20.6468105, 21.0361176
2: -5.2292953, 17.1967506, -4.7900958, 16.2342968, -21.4635925, 21.9868469
3: -6.2076254, 21.9800053, -5.7663260, 20.8878899, -27.0955162, 27.7463303
4: -5.0987215, 20.3454666, -4.7445159, 19.2791748, -24.3778954, 25.0899830

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5390859, upper bound: 27.5154066
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5148316, upper bound: 27.5154065
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.3295169, 15.1413870, -19.4084587, 19.3224525
1: -6.1816254, 15.3483210, -6.2038803, 15.5202951, -21.7019196, 21.5521927
2: -5.2292953, 17.1967506, -5.2456002, 17.4191399, -22.6484356, 22.4423504
3: -6.2076254, 21.9800053, -6.2830019, 22.3127079, -28.5203323, 28.2630081
4: -5.0987215, 20.3454666, -5.1508021, 20.6615257, -25.7602463, 25.4962692

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5158113, upper bound: 27.4711085
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5112267, upper bound: 27.4710085
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5204000, upper bound: 27.4852411
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3.7504961, 13.6328163, -4.8903704, 16.7688160, -20.5193100, 18.5231857
1: -5.4568090, 13.8825703, -6.9869814, 17.1864223, -22.6432304, 20.8695526
2: -4.6023803, 15.5473776, -5.8964348, 19.2150192, -23.8174000, 21.4438095
3: -5.4882956, 19.9745960, -7.0507531, 24.6006584, -30.0889511, 27.0253468
4: -4.5374780, 18.4000874, -5.7372026, 22.7977104, -27.3351879, 24.1372910

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5393655, upper bound: 27.5205857
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5405068, upper bound: 27.5381181
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -4.1517515, 14.6799326, -4.8903704, 16.7688160, -20.9205647, 19.5703030
1: -6.0161939, 15.0170507, -6.9869814, 17.1864223, -23.2026157, 22.0040321
2: -5.0848289, 16.8236141, -5.8964348, 19.2150192, -24.2998486, 22.7200470
3: -6.0437131, 21.5192928, -7.0507531, 24.6006584, -30.6443710, 28.5700417
4: -4.9683080, 19.9032288, -5.7372026, 22.7977104, -27.7660179, 25.6404305

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5143745, upper bound: 27.5053500
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4661808, upper bound: 27.4603956
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -3.2330034, 12.0068417, -15.2299318, 14.8617706
1: -4.5962033, 11.8650751, -4.6482325, 12.2217264, -16.8179283, 16.5133076
2: -3.8960257, 13.3130264, -3.8814385, 13.7559328, -17.6519585, 17.1944637
3: -4.5771689, 17.1330128, -4.6916032, 17.7408600, -22.3180294, 21.8246117
4: -3.8639328, 15.8723202, -3.8766093, 16.2748146, -20.1387482, 19.7489243

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_B2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4789614, upper bound: 27.4985041
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4789614, upper bound: 27.4830819
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -3.5987697, 13.2322645, -16.4553547, 15.2275372
1: -4.5962033, 11.8650751, -5.1593447, 13.4451246, -18.0413246, 17.0244198
2: -3.8960257, 13.3130264, -4.2915444, 15.0486994, -18.9447231, 17.6045685
3: -4.5771689, 17.1330128, -5.2183175, 19.4191303, -23.9962997, 22.3513260
4: -3.8639328, 15.8723202, -4.2520895, 17.8261108, -21.6900425, 20.1244049

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4674543, upper bound: 27.4953444
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4674543, upper bound: 27.5120715
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -4.1084003, 14.6362095, -17.8592987, 15.7371664
1: -4.5962033, 11.8650751, -5.8844862, 14.9058580, -19.5020618, 17.7495613
2: -3.8960257, 13.3130264, -4.9338779, 16.6669903, -20.5630150, 18.2469044
3: -4.5771689, 17.1330128, -5.9332085, 21.4064541, -25.9836216, 23.0662193
4: -3.8639328, 15.8723202, -4.8345366, 19.7077599, -23.5716915, 20.7068501

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4691611, upper bound: 27.4956170
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4674543, upper bound: 27.4573755
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -3.5670896, 13.1203718, -16.3434620, 15.1958580
1: -4.5962033, 11.8650751, -5.1293883, 13.3436155, -17.9398155, 16.9944630
2: -3.8960257, 13.3130264, -4.2717123, 14.9469910, -18.8430176, 17.5847378
3: -4.5771689, 17.1330128, -5.1802359, 19.2688313, -23.8460007, 22.3132458
4: -3.8639328, 15.8723202, -4.2362761, 17.6883545, -21.5522842, 20.1085949

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4674810, upper bound: 27.5097499
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4674810, upper bound: 27.5250973
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -3.2230899, 11.6287680, -4.0363760, 14.4265327, -17.6496201, 15.6651440
1: -4.5962033, 11.8650751, -5.7989521, 14.6991920, -19.2953949, 17.6640282
2: -3.8960257, 13.3130264, -4.8629599, 16.4490166, -20.3450413, 18.1759853
3: -4.5771689, 17.1330128, -5.8439355, 21.1054325, -25.6826019, 22.9769478
4: -3.8639328, 15.8723202, -4.7735724, 19.4315796, -23.2955112, 20.6458874

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4691611, upper bound: 27.5076844
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4674810, upper bound: 27.5238746
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -4.2013507, 14.8456116, -3.8377326, 13.7042408, -17.9055901, 18.6833439
1: -6.0548878, 15.1893959, -5.5586572, 14.0156584, -20.0705452, 20.7480488
2: -5.1086121, 17.0443916, -4.6897707, 15.7574348, -20.8660469, 21.7341614
3: -6.1252065, 21.8839588, -5.5885458, 20.1776047, -26.3028107, 27.4725037
4: -5.0339437, 20.2378387, -4.6097212, 18.6277122, -23.6616554, 24.8475609

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212161, upper bound: 27.5168861
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212161, upper bound: 27.5299744
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -3.9486008, 14.1665201, -4.2670722, 14.9929380, -18.9415379, 18.4335918
1: -5.6878052, 14.4651852, -6.1816254, 15.3483210, -21.0361176, 20.6468105
2: -4.7900958, 16.2342968, -5.2292953, 17.1967506, -21.9868469, 21.4635925
3: -5.7663260, 20.8878899, -6.2076254, 21.9800053, -27.7463303, 27.0955162
4: -4.7445159, 19.2791748, -5.0987215, 20.3454666, -25.0899811, 24.3778954

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4783277, upper bound: 27.5134418
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5109521, upper bound: 27.5279198
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -4.3295169, 15.1413870, -4.2670722, 14.9929380, -19.3224506, 19.4084587
1: -6.2038803, 15.5202951, -6.1816254, 15.3483210, -21.5521965, 21.7019196
2: -5.2456002, 17.4191399, -5.2292953, 17.1967506, -22.4423504, 22.6484356
3: -6.2830019, 22.3127079, -6.2076254, 21.9800053, -28.2630081, 28.5203323
4: -5.1508021, 20.6615257, -5.0987215, 20.3454666, -25.4962692, 25.7602444

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4711085, upper bound: 27.5158113
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4710085, upper bound: 27.5112267
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4852411, upper bound: 27.5204000
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -4.8903704, 16.7688160, -3.7504961, 13.6328163, -18.5231857, 20.5193100
1: -6.9869814, 17.1864223, -5.4568090, 13.8825703, -20.8695526, 22.6432304
2: -5.8964348, 19.2150192, -4.6023803, 15.5473776, -21.4438095, 23.8174000
3: -7.0507531, 24.6006584, -5.4882956, 19.9745960, -27.0253448, 30.0889511
4: -5.7372026, 22.7977104, -4.5374780, 18.4000874, -24.1372910, 27.3351879

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317857, upper bound: 27.5166737
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5317857, upper bound: 27.5319478
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -4.8903704, 16.7688160, -4.1517515, 14.6799326, -19.5703030, 20.9205647
1: -6.9869814, 17.1864223, -6.0161939, 15.0170507, -22.0040321, 23.2026157
2: -5.8964348, 19.2150192, -5.0848289, 16.8236141, -22.7200470, 24.2998486
3: -7.0507531, 24.6006584, -6.0437131, 21.5192928, -28.5700417, 30.6443710
4: -5.7372026, 22.7977104, -4.9683080, 19.9032288, -25.6404305, 27.7660179

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5015179, upper bound: 27.5051627
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4594100, upper bound: 27.4640054
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -4.1917214, 14.8112087, -3.2230899, 11.6287680, -15.8204899, 18.0342960
1: -5.9289794, 15.1069946, -4.5962033, 11.8650751, -17.7940540, 19.7031975
2: -4.9885335, 16.9621601, -3.8960257, 13.3130264, -18.3015575, 20.8581848
3: -6.0232954, 21.7922630, -4.5771689, 17.1330128, -23.1563072, 26.3694324
4: -4.9176126, 20.1535530, -3.8639328, 15.8723202, -20.7899246, 24.0174847

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A1_A1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4780076, upper bound: 27.4874337
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4586532, upper bound: 27.5223832
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -3.4959600, 12.8924370, -3.2230899, 11.6287680, -15.1247263, 16.1155262
1: -4.9927969, 13.1358471, -4.5962033, 11.8650751, -16.8578720, 17.7320499
2: -4.1466570, 14.7810230, -3.8960257, 13.3130264, -17.4596825, 18.6770458
3: -5.1062527, 19.0651340, -4.5771689, 17.1330128, -22.2392654, 23.6423035
4: -4.1699138, 17.5666885, -3.8639328, 15.8723202, -20.0422306, 21.4306221

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4856925, upper bound: 27.4856362
time: 0.50 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4856925, upper bound: 27.5205857
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -4.0789042, 14.4802399, -3.2230899, 11.6287680, -15.7076721, 17.7033272
1: -5.7796154, 14.7697134, -4.5962033, 11.8650751, -17.6446915, 19.3659172
2: -4.8593483, 16.5991631, -3.8960257, 13.3130264, -18.1723747, 20.4951897
3: -5.8634748, 21.3236389, -4.5771689, 17.1330128, -22.9964828, 25.9008083
4: -4.8038688, 19.7043839, -3.8639328, 15.8723202, -20.6761837, 23.5683155

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4868362, upper bound: 27.4874811
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4868362, upper bound: 27.5224305
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -4.4659195, 15.5693312, -3.1206288, 11.4433041, -15.9092236, 18.6899586
1: -6.3684106, 15.9348230, -4.4464903, 11.6433659, -18.0117760, 20.3813076
2: -5.3735919, 17.8539467, -3.7520332, 13.0516863, -18.4252758, 21.6059799
3: -6.4544306, 22.9067936, -4.4443731, 16.8623466, -23.3167763, 27.3511658
4: -5.2639656, 21.1983070, -3.7483118, 15.5597115, -20.8236771, 24.9466190

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4699753, upper bound: 27.5004194
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4915440, upper bound: 27.5291621
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3332319, 15.2043400, -3.1206288, 11.4433041, -15.7765360, 18.3249645
1: -6.1916523, 15.5625191, -4.4464903, 11.6433659, -17.8350182, 20.0090103
2: -5.2286344, 17.4494247, -3.7520332, 13.0516863, -18.2803173, 21.2014580
3: -6.2767315, 22.3976593, -4.4443731, 16.8623466, -23.1390781, 26.8420334
4: -5.1382856, 20.7020950, -3.7483118, 15.5597115, -20.6979980, 24.4504070

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4992238, upper bound: 27.5271746
time: 0.78 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4997350, upper bound: 27.5308498
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -4.8102126, 16.6795921, -3.1206288, 11.4433041, -16.2535133, 19.8002186
1: -6.8670340, 17.0531311, -4.4464903, 11.6433659, -18.5103989, 21.4996185
2: -5.7668905, 19.0408039, -3.7520332, 13.0516863, -18.8185730, 22.7928371
3: -6.9508572, 24.4394665, -4.4443731, 16.8623466, -23.8132038, 28.8838387
4: -5.6242981, 22.6034756, -3.7483118, 15.5597115, -21.1840096, 26.3517876

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4958118, upper bound: 27.5288939
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5012276, upper bound: 27.5328127
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.6728821, 16.2983379, -3.1206288, 11.4433041, -16.1161861, 19.4189625
1: -6.6850424, 16.6613579, -4.4464903, 11.6433659, -18.3284073, 21.1078491
2: -5.6154485, 18.6152859, -3.7520332, 13.0516863, -18.6671333, 22.3673191
3: -6.7668982, 23.9010849, -4.4443731, 16.8623466, -23.6292458, 28.3454590
4: -5.4928355, 22.0815315, -3.7483118, 15.5597115, -21.0525475, 25.8298435

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5099316, upper bound: 27.5302119
time: 0.58 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5107891, upper bound: 27.5328127
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -3.6487381, 13.3095112, -18.0301189, 19.8810616
1: -6.7455416, 16.6543579, -5.2600460, 13.5622673, -20.3078079, 21.9144039
2: -5.7037611, 18.6685505, -4.4125900, 15.2669535, -20.9707127, 23.0811405
3: -6.8226271, 23.8968391, -5.3424973, 19.6558323, -26.4784584, 29.2393360
4: -5.5644245, 22.1408634, -4.4083128, 18.1566029, -23.7210236, 26.5491753

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5213073, upper bound: 27.5090087
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5213073, upper bound: 27.5102859
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -3.9486008, 14.1665201, -18.8871269, 20.1809235
1: -6.7455416, 16.6543579, -5.6878052, 14.4651852, -21.2107277, 22.3421631
2: -5.7037611, 18.6685505, -4.7900958, 16.2342968, -21.9380531, 23.4586468
3: -6.8226271, 23.8968391, -5.7663260, 20.8878899, -27.7105141, 29.6631622
4: -5.5644245, 22.1408634, -4.7445159, 19.2791748, -24.8435974, 26.8853798

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5213075, upper bound: 27.5214493
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5213075, upper bound: 27.5227264
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.3295169, 15.1413870, -19.8619976, 20.5618401
1: -6.7455416, 16.6543579, -6.2038803, 15.5202951, -22.2658367, 22.8582382
2: -5.7037611, 18.6685505, -5.2456002, 17.4191399, -23.1228962, 23.9141502
3: -6.8226271, 23.8968391, -6.2830019, 22.3127079, -29.1353340, 30.1798401
4: -5.5644245, 22.1408634, -5.1508021, 20.6615257, -26.2259483, 27.2916660

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5222332, upper bound: 27.4998690
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5034346, upper bound: 27.4875070
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.2066560, 14.8591013, -4.4093771, 15.4658003, -19.6724567, 19.2684765
1: -6.0614095, 15.2031517, -6.3496609, 15.8067617, -21.8681717, 21.5528126
2: -5.1123199, 17.0585728, -5.3439074, 17.6911068, -22.8034210, 22.4024754
3: -6.1311889, 21.9030704, -6.3984680, 22.6925297, -28.8237152, 28.3015327
4: -5.0373154, 20.2538452, -5.2430549, 21.0035610, -26.0408745, 25.4969006

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5357975, upper bound: 27.5383680
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5367100, upper bound: 27.5388545
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.2066560, 14.8591013, -4.7787714, 16.4864807, -20.6931362, 19.6378727
1: -6.0614095, 15.2031517, -6.8363442, 16.8913631, -22.9527721, 22.0394936
2: -5.1123199, 17.0585728, -5.7713432, 18.8855362, -23.9978542, 22.8299160
3: -6.1311889, 21.9030704, -6.9030724, 24.1919346, -30.3231220, 28.8061390
4: -5.0373154, 20.2538452, -5.6268249, 22.4078331, -27.4451466, 25.8806705

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5234871, upper bound: 27.5226105
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5377892, upper bound: 27.5417534
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -4.3211579, 15.1794024, -4.8903704, 16.7688160, -21.0899734, 20.0697727
1: -6.1862206, 15.5389490, -6.9869814, 17.1864223, -23.3726406, 22.5259304
2: -5.2226639, 17.4232330, -5.8964348, 19.2150192, -24.4376831, 23.3196640
3: -6.2694144, 22.3618946, -7.0507531, 24.6006584, -30.8700733, 29.4126434
4: -5.1318283, 20.6690903, -5.7372026, 22.7977104, -27.9295387, 26.4062920

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5112722, upper bound: 27.5284494
time: 0.58 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5095737, upper bound: 27.5152063
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -4.6624351, 16.2836761, -4.8903704, 16.7688160, -21.4312477, 21.1740456
1: -6.6763396, 16.6494904, -6.9869814, 17.1864223, -23.8627605, 23.6364708
2: -5.6121321, 18.6002197, -5.8964348, 19.2150192, -24.8271503, 24.4966507
3: -6.7603388, 23.8824425, -7.0507531, 24.6006584, -31.3609962, 30.9331875
4: -5.4898567, 22.0649281, -5.7372026, 22.7977104, -28.2875671, 27.8021317

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5241703, upper bound: 27.5197253
time: 0.63 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5095737, upper bound: 27.5158128
time: 0.65 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.49 seconds
NS_A1_B1_A1_B1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4985280, upper bound: 27.4868007
NS_A1_B1_A1_B1_A2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4985281, upper bound: 27.5148312
NS_A1_B1_A1_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4985041, upper bound: 27.4868089
NS_A1_B1_A1_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4675379, upper bound: 27.5148312
NS_A1_B1_A2_B1_A1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4675619, upper bound: 27.4932271
NS_A1_B1_A2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4953688, upper bound: 27.5282606
NS_A1_B1_A2_B1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4956412, upper bound: 27.4922093
NS_A1_B1_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4956412, upper bound: 27.5272431
NS_A1_B1_A2_B1_A2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5097740, upper bound: 27.4937563
NS_A1_B1_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5097740, upper bound: 27.5287901
NS_A1_B1_A2_B1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5077085, upper bound: 27.4919642
NS_A1_B1_A2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5077085, upper bound: 27.5269980
NS_A1_B1_A2_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4953446, upper bound: 27.4932352
NS_A1_B1_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4953449, upper bound: 27.5281847
NS_A1_B1_A2_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4573755, upper bound: 27.4922174
NS_A1_B1_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4956172, upper bound: 27.5271670
NS_A1_B1_A2_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5097500, upper bound: 27.4937645
NS_A1_B1_A2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5097500, upper bound: 27.5287140
NS_A1_B1_A2_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5076845, upper bound: 27.4919724
NS_A1_B1_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5076846, upper bound: 27.5269219
NS_A1_B2_B1_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5077225, upper bound: 27.4960736
NS_A1_B2_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5077227, upper bound: 27.5305676
NS_A1_B2_B1_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4658738, upper bound: 27.4962459
NS_A1_B2_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4658738, upper bound: 27.5242628
NS_A1_B2_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4775321, upper bound: 27.5212161
NS_A1_B2_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4802362, upper bound: 27.5212161
NS_A1_B2_B2_A1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4802362, upper bound: 27.4809777
NS_A1_B2_B2_A1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4782182, upper bound: 27.4809777
NS_A1_B2_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5304330, upper bound: 27.4977980
NS_A1_B2_B2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5305778, upper bound: 27.5029216
NS_A1_B2_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5390859, upper bound: 27.5154066
NS_A1_B2_B2_A2_B1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5148316, upper bound: 27.5154065
NS_A1_B2_B2_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5112267, upper bound: 27.4710085
NS_A1_B2_B2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5204000, upper bound: 27.4852411
NS_A1_B2_B2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5393655, upper bound: 27.5205857
NS_A1_B2_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5405068, upper bound: 27.5381181
NS_A1_B2_B2_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5143745, upper bound: 27.5053500
NS_A1_B2_B2_A2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4661808, upper bound: 27.4603956
NS_A2_B1_A1_B1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4789614, upper bound: 27.4985041
NS_A2_B1_A1_B1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4789614, upper bound: 27.4830819
NS_A2_B1_A1_B2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4674543, upper bound: 27.4953444
NS_A2_B1_A1_B2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4674543, upper bound: 27.5120715
NS_A2_B1_A1_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4691611, upper bound: 27.4956170
NS_A2_B1_A1_B2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4674543, upper bound: 27.4573755
NS_A2_B1_A1_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4674810, upper bound: 27.5097499
NS_A2_B1_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4674810, upper bound: 27.5250973
NS_A2_B1_A1_B2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4691611, upper bound: 27.5076844
NS_A2_B1_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4674810, upper bound: 27.5238746
NS_A2_B1_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5212161, upper bound: 27.5168861
NS_A2_B1_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5212161, upper bound: 27.5299744
NS_A2_B1_A2_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4783277, upper bound: 27.5134418
NS_A2_B1_A2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5109521, upper bound: 27.5279198
NS_A2_B1_A2_B2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4710085, upper bound: 27.5112267
NS_A2_B1_A2_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4852411, upper bound: 27.5204000
NS_A2_B1_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5317857, upper bound: 27.5166737
NS_A2_B1_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5317857, upper bound: 27.5319478
NS_A2_B1_A2_B2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5015179, upper bound: 27.5051627
NS_A2_B1_A2_B2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4594100, upper bound: 27.4640054
NS_A2_B2_B1_A1_A1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4780076, upper bound: 27.4874337
NS_A2_B2_B1_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4586532, upper bound: 27.5223832
NS_A2_B2_B1_A1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4856925, upper bound: 27.4856362
NS_A2_B2_B1_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4856925, upper bound: 27.5205857
NS_A2_B2_B1_A1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4868362, upper bound: 27.4874811
NS_A2_B2_B1_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4868362, upper bound: 27.5224305
NS_A2_B2_B1_A1_A2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4699753, upper bound: 27.5004194
NS_A2_B2_B1_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4915440, upper bound: 27.5291621
NS_A2_B2_B1_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4992238, upper bound: 27.5271746
NS_A2_B2_B1_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4997350, upper bound: 27.5308498
NS_A2_B2_B1_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.4958118, upper bound: 27.5288939
NS_A2_B2_B1_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5012276, upper bound: 27.5328127
NS_A2_B2_B1_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5099316, upper bound: 27.5302119
NS_A2_B2_B1_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5107891, upper bound: 27.5328127
NS_A2_B2_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5213073, upper bound: 27.5090087
NS_A2_B2_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5213073, upper bound: 27.5102859
NS_A2_B2_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5213075, upper bound: 27.5214493
NS_A2_B2_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5213075, upper bound: 27.5227264
NS_A2_B2_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5222332, upper bound: 27.4998690
NS_A2_B2_B2_A2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5034346, upper bound: 27.4875070
NS_A2_B2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5357975, upper bound: 27.5383680
NS_A2_B2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5367100, upper bound: 27.5388545
NS_A2_B2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5234871, upper bound: 27.5226105
NS_A2_B2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5377892, upper bound: 27.5417534
NS_A2_B2_B2_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5112722, upper bound: 27.5284494
NS_A2_B2_B2_A2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5095737, upper bound: 27.5152063
NS_A2_B2_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5241703, upper bound: 27.5197253
NS_A2_B2_B2_A2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 3, lower bound: -27.5095737, upper bound: 27.5158128

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.5948901, 13.2204552, -2.5588205, 9.9024887, -13.4973793, 15.7792721
1: -5.1536551, 13.4331245, -3.7536967, 10.0406675, -15.1943207, 17.1868210
2: -4.2864428, 15.0356531, -3.1842098, 11.2857237, -15.5721655, 18.2198620
3: -5.2128172, 19.4023666, -3.7206895, 14.6038961, -19.8167133, 23.1230564
4: -4.2474804, 17.8104210, -3.2222981, 13.4670906, -17.7145653, 21.0327187

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4943021, upper bound: 27.5279785
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4943021, upper bound: 27.5279785
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.1081610, 14.6355429, -2.5588205, 9.9024887, -14.0106497, 17.1943626
1: -5.8841381, 14.9051600, -3.7536967, 10.0406675, -15.9248047, 18.6588554
2: -4.9335394, 16.6662235, -3.1842098, 11.2857237, -16.2192612, 19.8504295
3: -5.9328828, 21.4054794, -3.7206895, 14.6038961, -20.5367794, 25.1261692
4: -4.8342190, 19.7068176, -3.2222981, 13.4670906, -18.3013058, 22.9291153

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4945744, upper bound: 27.5269607
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4945745, upper bound: 27.5269607
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.5622625, 13.1057920, -2.5588205, 9.9024887, -13.4647512, 15.6646109
1: -5.1223269, 13.3287449, -3.7536967, 10.0406675, -15.1629925, 17.0824413
2: -4.2653337, 14.9308376, -3.1842098, 11.2857237, -15.5510569, 18.1150455
3: -5.1733098, 19.2479973, -3.7206895, 14.6038961, -19.7772064, 22.9686871
4: -4.2304645, 17.6688519, -3.2222981, 13.4670906, -17.6975479, 20.8911476

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5097676, upper bound: 27.5177638
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5093515, upper bound: 27.5252140
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0363760, 14.4265327, -2.5588205, 9.9024887, -13.9388647, 16.9853477
1: -5.7989521, 14.6991920, -3.7536967, 10.0406675, -15.8396187, 18.4528885
2: -4.8629599, 16.4490166, -3.1842098, 11.2857237, -16.1486797, 19.6332245
3: -5.8439355, 21.1054325, -3.7206895, 14.6038961, -20.4478321, 24.8261223
4: -4.7735724, 19.4315796, -3.2222981, 13.4670906, -18.2406616, 22.6538773

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5077049, upper bound: 27.5173041
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5072888, upper bound: 27.5247543
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3.5987697, 13.2322645, -3.1206288, 11.4433041, -15.0420742, 16.3528938
1: -5.1593447, 13.4451246, -4.4464903, 11.6433659, -16.8027115, 17.8916092
2: -4.2915444, 15.0486994, -3.7520332, 13.0516863, -17.3432312, 18.8007317
3: -5.2183175, 19.4191303, -4.4443731, 16.8623466, -22.0806637, 23.8635025
4: -4.2520895, 17.8261108, -3.7483118, 15.5597115, -19.8118019, 21.5744228

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4953449, upper bound: 27.5216665
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4953449, upper bound: 27.5281847
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.1084003, 14.6362095, -3.1206288, 11.4433041, -15.5517044, 17.7568359
1: -5.8844862, 14.9058580, -4.4464903, 11.6433659, -17.5278511, 19.3523483
2: -4.9338779, 16.6669903, -3.7520332, 13.0516863, -17.9855652, 20.4190235
3: -5.9332085, 21.4064541, -4.4443731, 16.8623466, -22.7955551, 25.8508263
4: -4.8345366, 19.7077599, -3.7483118, 15.5597115, -20.3942490, 23.4560719

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4956172, upper bound: 27.5206488
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4956172, upper bound: 27.5271670
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3.5670896, 13.1203718, -3.1206288, 11.4433041, -15.0103931, 16.2410011
1: -5.1293883, 13.3436155, -4.4464903, 11.6433659, -16.7727547, 17.7901058
2: -4.2717123, 14.9469910, -3.7520332, 13.0516863, -17.3233986, 18.6990242
3: -5.1802359, 19.2688313, -4.4443731, 16.8623466, -22.0425835, 23.7132034
4: -4.2362761, 17.6883545, -3.7483118, 15.5597115, -19.7959881, 21.4366665

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5097436, upper bound: 27.5177561
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5096672, upper bound: 27.5251379
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0363760, 14.4265327, -3.1206288, 11.4433041, -15.4796801, 17.5471573
1: -5.7989521, 14.6991920, -4.4464903, 11.6433659, -17.4423180, 19.1456833
2: -4.8629599, 16.4490166, -3.7520332, 13.0516863, -17.9146461, 20.2010498
3: -5.8439355, 21.1054325, -4.4443731, 16.8623466, -22.7062817, 25.5498047
4: -4.7735724, 19.4315796, -3.7483118, 15.5597115, -20.3332844, 23.1798916

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5076810, upper bound: 27.5172964
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5076045, upper bound: 27.5246782
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2.7557993, 10.6135054, -4.1501856, 14.7619257, -17.5177231, 14.7636900
1: -4.0050797, 10.7228260, -6.0045290, 15.0758657, -19.0809441, 16.7273540
2: -3.3896453, 11.9997158, -5.0533237, 16.8712635, -20.2609062, 17.0530396
3: -3.9829648, 15.5411358, -6.0436320, 21.6307240, -25.6136875, 21.5847683
4: -3.4068902, 14.2654495, -4.9472756, 19.9544792, -23.3613701, 19.2127247

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5082586, upper bound: 27.5185168
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5082586, upper bound: 27.5299972
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.1501856, 14.7619257, -4.1501856, 14.7619257, -18.9121113, 18.9121113
1: -6.0045290, 15.0758657, -6.0045290, 15.0758657, -21.0803928, 21.0803928
2: -5.0533237, 16.8712635, -5.0533237, 16.8712635, -21.9245872, 21.9245872
3: -6.0436320, 21.6307240, -6.0436320, 21.6307240, -27.6743546, 27.6743546
4: -4.9472756, 19.9544792, -4.9472756, 19.9544792, -24.9017525, 24.9017506

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4754517, upper bound: 27.5003967
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5150241, upper bound: 27.5373477
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3.1413078, 11.7806015, -4.2013507, 14.8456116, -17.9869194, 15.9819527
1: -4.5967102, 11.9731665, -6.0548878, 15.1893959, -19.7861061, 18.0280533
2: -3.8612831, 13.5137091, -5.1086121, 17.0443916, -20.9056740, 18.6223221
3: -4.6260147, 17.3682518, -6.1252065, 21.8839588, -26.5099735, 23.4934578
4: -3.8649619, 16.0315418, -5.0339437, 20.2378387, -24.1028004, 21.0654850

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4968727, upper bound: 27.5212111
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5018409, upper bound: 27.5209707
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3.4925537, 12.7211142, -4.2013507, 14.8456116, -18.3381634, 16.9224625
1: -5.0625253, 12.9815636, -6.0548878, 15.1893959, -20.2519188, 19.0364513
2: -4.2568483, 14.6295776, -5.1086121, 17.0443916, -21.3012390, 19.7381897
3: -5.0975418, 18.7498779, -6.1252065, 21.8839588, -26.9814987, 24.8750839
4: -4.2205181, 17.3231449, -5.0339437, 20.2378387, -24.4583549, 22.3570862

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4775321, upper bound: 27.5028849
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5038595, upper bound: 27.5028847
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -3.6842446, 13.4150352, -17.6821079, 18.6771793
1: -6.1816254, 15.3483210, -5.2898331, 13.6563931, -19.8380184, 20.6381493
2: -5.2292953, 17.1967506, -4.4382849, 15.3610401, -20.5903358, 21.6350346
3: -6.2076254, 21.9800053, -5.3709159, 19.7881203, -25.9957466, 27.3509216
4: -5.0987215, 20.3454666, -4.4274325, 18.2831478, -23.3818684, 24.7728996

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5278367, upper bound: 27.4960178
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4816115, upper bound: 27.4512753
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5302833, upper bound: 27.4977980
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064154, upper bound: 27.4977980
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -3.5756357, 13.0975380, -17.3646107, 18.5685730
1: -6.1816254, 15.3483210, -5.1526637, 13.3384438, -19.5200691, 20.5009766
2: -5.2292953, 17.1967506, -4.3211308, 15.0190945, -20.2483902, 21.5178776
3: -6.2076254, 21.9800053, -5.2348752, 19.3474312, -25.5550575, 27.2148800
4: -5.0987215, 20.3454666, -4.3262157, 17.8628006, -22.9615211, 24.6716824

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5065602, upper bound: 27.5029216
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5304281, upper bound: 27.5029216
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8483875, 13.8090839, -3.9486008, 14.1665201, -18.0149059, 17.7576828
1: -5.6051860, 14.0970688, -5.6878052, 14.4651852, -20.0703716, 19.7848721
2: -4.7509747, 15.8073130, -4.7900958, 16.2342968, -20.9852715, 20.5974083
3: -5.6255231, 20.2453957, -5.7663260, 20.8878899, -26.5134125, 26.0117207
4: -4.6657948, 18.7160625, -4.7445159, 19.2791748, -23.9449692, 23.4605789

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5373140, upper bound: 27.5132154
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386998, upper bound: 27.5150050
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -4.2670722, 14.9929380, -4.2534475, 15.0084286, -19.2755013, 19.2463856
1: -6.1816254, 15.3483210, -6.0919037, 15.3566914, -21.5383167, 21.4402199
2: -5.2292953, 17.1967506, -5.1396985, 17.2211037, -22.4503994, 22.3364468
3: -6.2076254, 21.9800053, -6.1773243, 22.1120911, -28.3197174, 28.1573296
4: -5.0987215, 20.3454666, -5.0583181, 20.4273663, -25.5260887, 25.4037800

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5178669, upper bound: 27.4724348
time: 0.61 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5202255, upper bound: 27.4850530
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3.7504961, 13.6328163, -4.9399590, 16.9097595, -20.6602554, 18.5727711
1: -5.4568090, 13.8825703, -7.0431247, 17.3266182, -22.7834282, 20.9256954
2: -4.6023803, 15.5473776, -5.9432487, 19.3605976, -23.9629784, 21.4906273
3: -5.4882956, 19.9745960, -7.1096210, 24.7885113, -30.2768040, 27.0842152
4: -4.5374780, 18.4000874, -5.7747784, 22.9828663, -27.5203438, 24.1748657

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5393655, upper bound: 27.5205857
time: 0.86 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5393655, upper bound: 27.5205857
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.7504961, 13.6328163, -4.8025160, 16.5289459, -20.2794380, 18.4353275
1: -5.4568090, 13.8825703, -6.8607712, 16.9337120, -22.3905201, 20.7433376
2: -4.6023803, 15.5473776, -5.7895617, 18.9348183, -23.5371990, 21.3369389
3: -5.4882956, 19.9745960, -6.9263873, 24.2517948, -29.7400875, 26.9009838
4: -4.5374780, 18.4000874, -5.6427827, 22.4642315, -27.0017090, 24.0428677

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5405068, upper bound: 27.5362377
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5405068, upper bound: 27.5381181
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3.1206288, 11.4433041, -3.5670896, 13.1203718, -16.2410011, 15.0103931
1: -4.4464903, 11.6433659, -5.1293883, 13.3436155, -17.7901039, 16.7727547
2: -3.7520332, 13.0516863, -4.2717123, 14.9469910, -18.6990242, 17.3233986
3: -4.4443731, 16.8623466, -5.1802359, 19.2688313, -23.7132034, 22.0425835
4: -3.7483118, 15.5597115, -4.2362761, 17.6883545, -21.4366665, 19.7959881

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4589799, upper bound: 27.5250934
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4674810, upper bound: 27.5248522
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3.1206288, 11.4433041, -4.0363760, 14.4265327, -17.5471573, 15.4796801
1: -4.4464903, 11.6433659, -5.7989521, 14.6991920, -19.1456833, 17.4423180
2: -3.7520332, 13.0516863, -4.8629599, 16.4490166, -20.2010498, 17.9146442
3: -4.4443731, 16.8623466, -5.8439355, 21.1054325, -25.5498047, 22.7062817
4: -3.7483118, 15.5597115, -4.7735724, 19.4315796, -23.1798916, 20.3332844

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4605685, upper bound: 27.5238723
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4605685, upper bound: 27.5236410
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -4.2013507, 14.8456116, -3.4213684, 12.5575457, -16.7588959, 18.2669792
1: -6.0548878, 15.1893959, -4.9989309, 12.7943726, -18.8492584, 20.1883278
2: -5.1086121, 17.0443916, -4.2210941, 14.4054899, -19.5141029, 21.2654858
3: -6.1252065, 21.8839588, -5.0228553, 18.4869843, -24.6121902, 26.9068127
4: -5.0339437, 20.2378387, -4.1848927, 17.0421448, -22.0760880, 24.4227314

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5183336, upper bound: 27.5168859
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5288623, upper bound: 27.5168859
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -4.2013507, 14.8456116, -3.7689612, 13.5107183, -17.7120686, 18.6145725
1: -6.0548878, 15.1893959, -5.4607773, 13.8125982, -19.8674850, 20.6501713
2: -5.1086121, 17.0443916, -4.6060138, 15.5323753, -20.6409874, 21.6504059
3: -6.1252065, 21.8839588, -5.4899335, 19.8952503, -26.0204563, 27.3738899
4: -5.0339437, 20.2378387, -4.5339608, 18.3602276, -23.3941708, 24.7717991

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5109520, upper bound: 27.5279197
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5109520, upper bound: 27.5299742
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.9486008, 14.1665201, -4.2006721, 14.8031902, -18.7517910, 18.3671894
1: -5.6878052, 14.4651852, -6.0869379, 15.1494980, -20.8372974, 20.5521183
2: -4.7900958, 16.2342968, -5.1482048, 16.9752731, -21.7653694, 21.3825016
3: -5.7663260, 20.8878899, -6.1115742, 21.7016144, -27.4679394, 26.9994621
4: -4.7445159, 19.2791748, -5.0243750, 20.0811024, -24.8256168, 24.3035507

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5125247, upper bound: 27.5366036
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5142122, upper bound: 27.5270223
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5140927, upper bound: 27.5325438
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -4.2534475, 15.0084286, -4.2670722, 14.9929380, -19.2463818, 19.2755013
1: -6.0919037, 15.3566914, -6.1816254, 15.3483210, -21.4402199, 21.5383167
2: -5.1396985, 17.2211037, -5.2292953, 17.1967506, -22.3364449, 22.4503994
3: -6.1773243, 22.1120911, -6.2076254, 21.9800053, -28.1573296, 28.3197174
4: -5.0583181, 20.4273663, -5.0987215, 20.3454666, -25.4037819, 25.5260887

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4724348, upper bound: 27.5178667
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2_A1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4850530, upper bound: 27.5202255
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -4.8903704, 16.7688160, -2.2421675, 9.2000980, -14.0904675, 19.0109787
1: -6.9869814, 17.1864223, -3.2305479, 9.2227631, -16.2097416, 20.4169693
2: -5.8964348, 19.2150192, -2.7389321, 10.2997265, -16.1961613, 21.9539509
3: -7.0507531, 24.6006584, -3.2190104, 13.4720726, -20.5228233, 27.8196678
4: -5.7372026, 22.7977104, -2.8346868, 12.2478743, -17.9850769, 25.6323967

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5316989, upper bound: 27.5166737
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5316989, upper bound: 27.5166737
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -4.8903704, 16.7688160, -3.7418022, 13.6122818, -18.5026512, 20.5106144
1: -6.9869814, 17.1864223, -5.4465876, 13.8609514, -20.8479328, 22.6330090
2: -5.8964348, 19.2150192, -4.5936899, 15.5233393, -21.4197731, 23.8087082
3: -7.0507531, 24.6006584, -5.4781079, 19.9450760, -26.9958229, 30.0787659
4: -5.7372026, 22.7977104, -4.5295615, 18.3716660, -24.1088676, 27.3272724

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5006719, upper bound: 27.5264285
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5297853, upper bound: 27.5314148
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -4.1917214, 14.8112087, -3.1206288, 11.4433041, -15.6350250, 17.9318333
1: -5.9289794, 15.1069946, -4.4464903, 11.6433659, -17.5723457, 19.5534859
2: -4.9885335, 16.9621601, -3.7520332, 13.0516863, -18.0402184, 20.7141933
3: -6.0232954, 21.7922630, -4.4443731, 16.8623466, -22.8856430, 26.2366371
4: -4.9176126, 20.1535530, -3.7483118, 15.5597115, -20.4773235, 23.9018650

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4754637, upper bound: 27.5094038
time: 0.68 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4753873, upper bound: 27.5167856
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3.4959600, 12.8924370, -3.1206288, 11.4433041, -14.9392633, 16.0130653
1: -4.9927969, 13.1358471, -4.4464903, 11.6433659, -16.6361618, 17.5823364
2: -4.1466570, 14.7810230, -3.7520332, 13.0516863, -17.1983433, 18.5330563
3: -5.1062527, 19.0651340, -4.4443731, 16.8623466, -21.9685993, 23.5095062
4: -4.1699138, 17.5666885, -3.7483118, 15.5597115, -19.7296257, 21.3150005

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A1_B2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4856810, upper bound: 27.5091446
time: 0.53 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A1_B2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4856045, upper bound: 27.5165264
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0789042, 14.4802399, -3.1206288, 11.4433041, -15.5222082, 17.6008606
1: -5.7796154, 14.7697134, -4.4464903, 11.6433659, -17.4229813, 19.2162037
2: -4.8593483, 16.5991631, -3.7520332, 13.0516863, -17.9110336, 20.3511963
3: -5.8634748, 21.3236389, -4.4443731, 16.8623466, -22.7258205, 25.7680111
4: -4.8038688, 19.7043839, -3.7483118, 15.5597115, -20.3635807, 23.4526958

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4868251, upper bound: 27.5107991
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4867486, upper bound: 27.5181809
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.5097160, 15.7111883, -3.1206288, 11.4433041, -15.9530201, 18.8318119
1: -6.3891387, 16.0570030, -4.4464903, 11.6433659, -18.0325050, 20.5034924
2: -5.3842955, 17.9877872, -3.7520332, 13.0516863, -18.4359818, 21.7398205
3: -6.4823389, 23.0895786, -4.4443731, 16.8623466, -23.3446846, 27.5339508
4: -5.2716980, 21.3440895, -3.7483118, 15.5597115, -20.8314095, 25.0924015

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4966066, upper bound: 27.5041035
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4963921, upper bound: 27.5110854
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.7953649, 13.7514868, -3.1206288, 11.4433041, -15.2386684, 16.8721142
1: -5.4297147, 14.0437889, -4.4464903, 11.6433659, -17.0730801, 18.4902802
2: -4.5389991, 15.7582598, -3.7520332, 13.0516863, -17.5906849, 19.5102921
3: -5.5341959, 20.3090591, -4.4443731, 16.8623466, -22.3965416, 24.7534332
4: -4.5242596, 18.7015858, -3.7483118, 15.5597115, -20.0839710, 22.4498978

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5179661, upper bound: 27.5152584
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5176804, upper bound: 27.5223186
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3747315, 15.3341866, -3.1206288, 11.4433041, -15.8180351, 18.4548149
1: -6.2054200, 15.6689262, -4.4464903, 11.6433659, -17.8487854, 20.1154175
2: -5.2343960, 17.5667667, -3.7520332, 13.0516863, -18.2860775, 21.3188000
3: -6.2924232, 22.5567398, -4.4443731, 16.8623466, -23.1547699, 27.0011139
4: -5.1410632, 20.8291512, -3.7483118, 15.5597115, -20.7007751, 24.5774632

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5188016, upper bound: 27.5190918
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5185215, upper bound: 27.5264327
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.2741466, 15.2362843, -3.1206288, 11.4433041, -15.7174482, 18.3569088
1: -6.1128454, 15.5440960, -4.4464903, 11.6433659, -17.7562103, 19.9905853
2: -5.1103663, 17.3684559, -3.7520332, 13.0516863, -18.1620522, 21.1204891
3: -6.2062607, 22.3792076, -4.4443731, 16.8623466, -23.0686073, 26.8235798
4: -5.0423012, 20.6233673, -3.7483118, 15.5597115, -20.6020126, 24.3716793

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5149077, upper bound: 27.5217602
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5149077, upper bound: 27.5288936
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.8503065, 16.8103180, -3.1206288, 11.4433041, -16.2936096, 19.9309464
1: -6.8873014, 17.1638966, -4.4464903, 11.6433659, -18.5306664, 21.6103859
2: -5.7752533, 19.1638699, -3.7520332, 13.0516863, -18.8269367, 22.9159031
3: -6.9768243, 24.6037540, -4.4443731, 16.8623466, -23.8391705, 29.0481262
4: -5.6286907, 22.7314110, -3.7483118, 15.5597115, -21.1884022, 26.4797230

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5195552, upper bound: 27.5196219
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5192771, upper bound: 27.5267319
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1482620, 14.8815165, -3.1206288, 11.4433041, -15.5915661, 18.0021420
1: -5.9450006, 15.1811714, -4.4464903, 11.6433659, -17.5883675, 19.6276627
2: -4.9566765, 16.9697647, -3.7520332, 13.0516863, -18.0083618, 20.7217979
3: -6.0400605, 21.8677635, -4.4443731, 16.8623466, -22.9024067, 26.3121376
4: -4.9060669, 20.1351357, -3.7483118, 15.5597115, -20.4657784, 23.8834476

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5290195, upper bound: 27.5191531
time: 0.62 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5287414, upper bound: 27.5265349
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7102542, 16.4189987, -3.1206288, 11.4433041, -16.1535568, 19.5396271
1: -6.6995034, 16.7579174, -4.4464903, 11.6433659, -18.3428688, 21.2044067
2: -5.6185560, 18.7233448, -3.7520332, 13.0516863, -18.6702423, 22.4753780
3: -6.7803822, 24.0479298, -4.4443731, 16.8623466, -23.6427288, 28.4923019
4: -5.4927816, 22.1969910, -3.7483118, 15.5597115, -21.0524940, 25.9453030

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5298764, upper bound: 27.5213206
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5295983, upper bound: 27.5287024
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -4.4364772, 15.4927216, -3.6487381, 13.3095112, -17.7459888, 19.1414585
1: -6.3414073, 15.8659678, -5.2600460, 13.5622673, -19.9036751, 21.1260147
2: -5.3552303, 17.7862968, -4.4125900, 15.2669535, -20.6221848, 22.1988831
3: -6.4260044, 22.8168812, -5.3424973, 19.6558323, -26.0818367, 28.1593781
4: -5.2513251, 21.1006165, -4.4083128, 18.1566029, -23.4079247, 25.5089283

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5126474, upper bound: 27.5084783
time: 0.68 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5126475, upper bound: 27.5090087
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -4.7756276, 16.5842285, -3.6487381, 13.3095112, -18.0851364, 20.2329636
1: -6.8313746, 16.9627132, -5.2600460, 13.5622673, -20.3936424, 22.2227592
2: -5.7417583, 18.9490433, -4.4125900, 15.2669535, -21.0087090, 23.3616295
3: -6.9129910, 24.3177433, -5.3424973, 19.6558323, -26.5688171, 29.6602402
4: -5.6055784, 22.4766064, -4.4083128, 18.1566029, -23.7621784, 26.8849182

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5088731, upper bound: 27.5091000
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5210489, upper bound: 27.5089642
time: 0.76 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.35 + 417.72 = 420.07 seconds
