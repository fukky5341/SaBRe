## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 38.7593690755


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.4935045, 16.7457676, -4.4935045, 16.7457676, -21.2392731, 21.2392712)
1: (-12.6764650, 34.5451698, -12.6764650, 34.5451698, -47.2216339, 47.2216339)
2: (-19.5602589, 31.8633156, -19.5602589, 31.8633156, -51.4235764, 51.4235764)
3: (-10.6092863, 40.3084297, -10.6092863, 40.3084297, -50.9177132, 50.9177170)
4: (-17.5911102, 27.9587860, -17.5911102, 27.9587860, -45.5498962, 45.5498962)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.90 + 2.30 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -38.7671225, upper bound: 38.7671225

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7510005, upper bound: 38.7636174
time: 0.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7510005, upper bound: 38.7670314
time: 0.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.65 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 4, lower bound: -38.7510005, upper bound: 38.7636174
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 4, lower bound: -38.7510005, upper bound: 38.7670314

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.3467560, 12.4833908, -4.3217182, 16.1028194, -19.4495754, 16.8051052
1: -9.4122744, 25.7768497, -12.1783247, 33.2198486, -42.6321144, 37.9551697
2: -14.4818916, 23.6561546, -18.7716808, 30.6500530, -45.1319389, 42.4278297
3: -7.8629112, 30.1335220, -10.1911373, 38.7725372, -46.6354485, 40.3246613
4: -13.0019255, 20.7626228, -16.8833294, 26.8960991, -39.8980217, 37.6459503

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7493014, upper bound: 38.7493014
time: 0.60 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7493014, upper bound: 38.7636174
time: 0.62 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.4573598, 16.6108780, -4.4894800, 16.7307510, -21.1881104, 21.1003513
1: -12.5720854, 34.2636490, -12.6648445, 34.5137978, -47.0858841, 46.9284935
2: -19.3943081, 31.6146030, -19.5417767, 31.8356190, -51.2299271, 51.1563797
3: -10.5214348, 39.9831924, -10.5995035, 40.2717934, -50.7932243, 50.5826950
4: -17.4427109, 27.7403603, -17.5745811, 27.9344635, -45.3771629, 45.3149414

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7636174, upper bound: 38.7510005
time: 0.59 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7636174, upper bound: 38.7670314
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.14 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.14
Output dim: 4, lower bound: -38.7493014, upper bound: 38.7493014
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 4, lower bound: -38.7493014, upper bound: 38.7636174
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 4, lower bound: -38.7636174, upper bound: 38.7510005
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 4, lower bound: -38.7636174, upper bound: 38.7670314

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.3467560, 12.4833908, -4.4528513, 16.5930710, -19.9398212, 16.9362392
1: -9.4122744, 25.7768497, -12.5586758, 34.2289352, -43.6412048, 38.3355255
2: -14.4818916, 23.6561546, -19.3733349, 31.5822144, -46.0641022, 43.0294876
3: -7.8629112, 30.1335220, -10.5101929, 39.9434395, -47.8063507, 40.6437149
4: -13.0019255, 20.7626228, -17.4237480, 27.7120590, -40.7139816, 38.1863708

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7427451, upper bound: 38.7605263
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7458052, upper bound: 38.7616730
time: 0.59 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.4573598, 16.6108780, -3.3467560, 12.4833908, -16.9407501, 19.9576302
1: -12.5720854, 34.2636490, -9.4122744, 25.7768497, -38.3489304, 43.6759224
2: -19.3943081, 31.6146030, -14.4818916, 23.6561546, -43.0504608, 46.0964966
3: -10.5214348, 39.9831924, -7.8629112, 30.1335220, -40.6549530, 47.8461037
4: -17.4427109, 27.7403603, -13.0019255, 20.7626228, -38.2053299, 40.7422829

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7499772, upper bound: 38.7463609
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7616730, upper bound: 38.7480869
time: 0.85 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.4573598, 16.6108780, -4.4573598, 16.6108780, -21.0682335, 21.0682335
1: -12.5720854, 34.2636490, -12.5720854, 34.2636490, -46.8357353, 46.8357353
2: -19.3943081, 31.6146030, -19.3943081, 31.6146030, -51.0089111, 51.0089111
3: -10.5214348, 39.9831924, -10.5214348, 39.9831924, -50.5046272, 50.5046272
4: -17.4427109, 27.7403603, -17.4427109, 27.7403603, -45.1830635, 45.1830635

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7499773, upper bound: 38.7633287
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7427451, upper bound: 38.7480869
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.33 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 4, lower bound: -38.7427451, upper bound: 38.7605263
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 4, lower bound: -38.7458052, upper bound: 38.7616730
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 4, lower bound: -38.7499772, upper bound: 38.7463609
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 4, lower bound: -38.7616730, upper bound: 38.7480869
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 4, lower bound: -38.7499773, upper bound: 38.7633287
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 4, lower bound: -38.7427451, upper bound: 38.7480869

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.1007779, 11.5883722, -4.4528513, 16.5930710, -19.6938457, 16.0412216
1: -8.7085648, 23.9029922, -12.5586758, 34.2289352, -42.9374962, 36.4616699
2: -13.3632832, 21.9669895, -19.3733349, 31.5822144, -44.9454956, 41.3403244
3: -7.2686996, 28.0077953, -10.5101929, 39.9434395, -47.2121315, 38.5179901
4: -11.9931393, 19.2846527, -17.4237480, 27.7120590, -39.7051926, 36.7083969

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7433008, upper bound: 38.7489124
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7433008, upper bound: 38.7605263
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.1381559, 11.7177000, -4.4179769, 16.4634686, -19.6016235, 16.1356735
1: -8.8128052, 24.2017918, -12.4602413, 33.9611359, -42.7739334, 36.6620293
2: -13.5281200, 22.3033257, -19.2218075, 31.3375664, -44.8656845, 41.5251312
3: -7.3591671, 28.3426857, -10.4275217, 39.6323051, -46.9914703, 38.7702065
4: -12.1489334, 19.5659904, -17.2869930, 27.4968815, -39.6458130, 36.8529816

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7463609, upper bound: 38.7499772
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7463609, upper bound: 38.7616730
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.2385187, 15.8061705, -3.3162704, 12.3707552, -16.6092701, 19.1224403
1: -11.9449787, 32.6061325, -9.3256121, 25.5436954, -37.4886742, 41.9317436
2: -18.4001579, 30.1615677, -14.3465271, 23.4460793, -41.8462372, 44.5080948
3: -9.9945278, 38.0952148, -7.7900949, 29.8630924, -39.8576202, 45.8853035
4: -16.5515156, 26.4587784, -12.8801613, 20.5776234, -37.1291389, 39.3389359

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7605262, upper bound: 38.7450199
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7605262, upper bound: 38.7480869
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1781120, 15.5949068, -4.4573598, 16.6108780, -20.7889862, 20.0522652
1: -11.7708502, 32.1254539, -12.5720854, 34.2636490, -46.0345001, 44.6975403
2: -18.1361446, 29.6870670, -19.3943081, 31.6146030, -49.7507477, 49.0813713
3: -9.8482132, 37.5558205, -10.5214348, 39.9831924, -49.8314056, 48.0772552
4: -16.3078804, 26.0557480, -17.4427109, 27.7403603, -44.0482368, 43.4984550

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7505329
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7633287
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.36 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.36
Output dim: 4, lower bound: -38.7433008, upper bound: 38.7489124
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -38.7433008, upper bound: 38.7605263
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.36
Output dim: 4, lower bound: -38.7463609, upper bound: 38.7499772
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -38.7463609, upper bound: 38.7616730
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -38.7605262, upper bound: 38.7450199
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -38.7605262, upper bound: 38.7480869
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.36
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7505329
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7633287

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.1007779, 11.5883722, -4.2349863, 15.7925758, -18.8933525, 15.8233557
1: -8.7085648, 23.9029922, -11.9344845, 32.5791855, -41.2877502, 35.8374786
2: -13.3632832, 21.9669895, -18.3838730, 30.1363392, -43.4996223, 40.3508606
3: -7.2686996, 28.0077953, -9.9857378, 38.0639839, -45.3326759, 37.9935341
4: -11.9931393, 19.2846527, -16.5368099, 26.4366493, -38.4297867, 35.8214569

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7432361, upper bound: 38.7481954
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7364640, upper bound: 38.7502033
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7178074, upper bound: 38.7473146
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.1381559, 11.7177000, -4.2349863, 15.7925758, -18.9307327, 15.9526834
1: -8.8128052, 24.2017918, -11.9344845, 32.5791855, -41.3919907, 36.1362762
2: -13.5281200, 22.3033257, -18.3838730, 30.1363392, -43.6644592, 40.6871986
3: -7.3591671, 28.3426857, -9.9857378, 38.0639839, -45.4231453, 38.3284225
4: -12.1489334, 19.5659904, -16.5368099, 26.4366493, -38.5855827, 36.1027985

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7462962, upper bound: 38.7494321
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7413117, upper bound: 38.7503230
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7462279, upper bound: 38.7556254
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7459371, upper bound: 38.7555632
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7458367, upper bound: 38.7506654
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7421201, upper bound: 38.7556254
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7441462, upper bound: 38.7546652
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.2385187, 15.8061705, -3.1007779, 11.5883722, -15.8268881, 18.9069481
1: -11.9449787, 32.6061325, -8.7085648, 23.9029922, -35.8479691, 41.3146896
2: -18.4001579, 30.1615677, -13.3632832, 21.9669895, -40.3671455, 43.5248451
3: -9.9945278, 38.0952148, -7.2686996, 28.0077953, -38.0023232, 45.3639069
4: -16.5515156, 26.4587784, -11.9931393, 19.2846527, -35.8361626, 38.4519157

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7605262, upper bound: 38.7450199
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7603882, upper bound: 38.7450018
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.2385187, 15.8061705, -3.1381559, 11.7177000, -15.9562187, 18.9443264
1: -11.9449787, 32.6061325, -8.8128052, 24.2017918, -36.1467705, 41.4189339
2: -18.4001579, 30.1615677, -13.5281200, 22.3033257, -40.7034760, 43.6896820
3: -9.9945278, 38.0952148, -7.3591671, 28.3426857, -38.3372116, 45.4543762
4: -16.5515156, 26.4587784, -12.1489334, 19.5659904, -36.1175079, 38.6077118

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7605262, upper bound: 38.7458668
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7603882, upper bound: 38.7458492
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.1781120, 15.5949068, -4.2385187, 15.8061705, -19.9842815, 19.8334255
1: -11.7708502, 32.1254539, -11.9449787, 32.6061325, -44.3769760, 44.0704346
2: -18.1361446, 29.6870670, -18.4001579, 30.1615677, -48.2977104, 48.0872192
3: -9.8482132, 37.5558205, -9.9945278, 38.0952148, -47.9434242, 47.5503464
4: -16.3078804, 26.0557480, -16.5515156, 26.4587784, -42.7666550, 42.6072578

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7500778, upper bound: 38.7433008
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7612928
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.65 seconds
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7364640, upper bound: 38.7502033
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7178074, upper bound: 38.7473146
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7421201, upper bound: 38.7556254
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7441462, upper bound: 38.7546652
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7605262, upper bound: 38.7450199
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7603882, upper bound: 38.7450018
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7605262, upper bound: 38.7458668
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7603882, upper bound: 38.7458492
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7500778, upper bound: 38.7433008
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.65
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7612928

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.2179031, 15.7329702, -3.1007779, 11.5883722, -15.8062735, 18.8337440
1: -11.8898153, 32.4463692, -8.7085648, 23.9029922, -35.7928047, 41.1549263
2: -18.3172112, 30.0103245, -13.3632832, 21.9669895, -40.2842026, 43.3736038
3: -9.9484634, 37.9145317, -7.2686996, 28.0077953, -37.9562607, 45.1832237
4: -16.4761658, 26.3230953, -11.9931393, 19.2846527, -35.7608109, 38.3162346

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7588706, upper bound: 38.7436704
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7588706, upper bound: 38.7450017
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.2185087, 15.7317734, -3.1007779, 11.5883722, -15.8068800, 18.8325500
1: -11.8887796, 32.4540100, -8.7085648, 23.9029922, -35.7917709, 41.1625671
2: -18.3148880, 30.0150185, -13.3632832, 21.9669895, -40.2818756, 43.3782997
3: -9.9472942, 37.9196472, -7.2686996, 28.0077953, -37.9550896, 45.1883469
4: -16.4736309, 26.3303986, -11.9931393, 19.2846527, -35.7582817, 38.3235397

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7588706, upper bound: 38.7436704
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7588706, upper bound: 38.7450018
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.2179031, 15.7329702, -3.1381559, 11.7177000, -15.9356031, 18.8711262
1: -11.8898153, 32.4463692, -8.8128052, 24.2017918, -36.0916023, 41.2591705
2: -18.3172112, 30.0103245, -13.5281200, 22.3033257, -40.6205330, 43.5384369
3: -9.9484634, 37.9145317, -7.3591671, 28.3426857, -38.2911491, 45.2736931
4: -16.4761658, 26.3230953, -12.1489334, 19.5659904, -36.0421562, 38.4720306

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7606775, upper bound: 38.7458492
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7606775, upper bound: 38.7458492
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.2185087, 15.7317734, -3.1381559, 11.7177000, -15.9362087, 18.8699303
1: -11.8887796, 32.4540100, -8.8128052, 24.2017918, -36.0905724, 41.2668114
2: -18.3148880, 30.0150185, -13.5281200, 22.3033257, -40.6182137, 43.5431366
3: -9.9472942, 37.9196472, -7.3591671, 28.3426857, -38.2899780, 45.2788162
4: -16.4736309, 26.3303986, -12.1489334, 19.5659904, -36.0396194, 38.4793320

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7606775, upper bound: 38.7458492
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7606775, upper bound: 38.7458492
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.1268620, 15.4032192, -4.2385187, 15.8061705, -19.9330330, 19.6417389
1: -11.6270456, 31.7293167, -11.9449787, 32.6061325, -44.2331772, 43.6742935
2: -17.9145794, 29.3193798, -18.4001579, 30.1615677, -48.0761414, 47.7195320
3: -9.7269125, 37.0954361, -9.9945278, 38.0952148, -47.8221207, 47.0899582
4: -16.1073837, 25.7313728, -16.5515156, 26.4587784, -42.5661621, 42.2828903

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7518760, upper bound: 38.7605794
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7517810, upper bound: 38.7612594
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.59 seconds
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7588706, upper bound: 38.7436704
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7588706, upper bound: 38.7450017
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7588706, upper bound: 38.7436704
NS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7588706, upper bound: 38.7450018
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7606775, upper bound: 38.7458492
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7606775, upper bound: 38.7458492
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7606775, upper bound: 38.7458492
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7606775, upper bound: 38.7458492
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7518760, upper bound: 38.7605794
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 4, lower bound: -38.7517810, upper bound: 38.7612594

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.2179031, 15.7329702, -3.1175520, 11.6491060, -15.8670092, 18.8505230
1: -11.8898153, 32.4463692, -8.7564507, 24.0471725, -35.9369774, 41.2028122
2: -18.3172112, 30.0103245, -13.4434071, 22.1640167, -40.4812279, 43.4537277
3: -9.9484634, 37.9145317, -7.3124356, 28.1674156, -38.1158791, 45.2269592
4: -16.4761658, 26.3230953, -12.0717487, 19.4424267, -35.9185829, 38.3948441

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.2179031, 15.7329702, -3.1215446, 11.6570091, -15.8749123, 18.8545132
1: -11.8898153, 32.4463692, -8.7665911, 24.0765839, -35.9663887, 41.2129517
2: -18.3172112, 30.0103245, -13.4583864, 22.1821671, -40.4993744, 43.4687042
3: -9.9484634, 37.9145317, -7.3203278, 28.1978359, -38.1463013, 45.2348595
4: -16.4761658, 26.3230953, -12.0851660, 19.4590378, -35.9352036, 38.4082603

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.2185087, 15.7317734, -3.1175520, 11.6491060, -15.8676147, 18.8493252
1: -11.8887796, 32.4540100, -8.7564507, 24.0471725, -35.9359512, 41.2104530
2: -18.3148880, 30.0150185, -13.4434071, 22.1640167, -40.4789047, 43.4584274
3: -9.9472942, 37.9196472, -7.3124356, 28.1674156, -38.1147079, 45.2320786
4: -16.4736309, 26.3303986, -12.0717487, 19.4424267, -35.9160538, 38.4021454

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.2185087, 15.7317734, -3.1215446, 11.6570091, -15.8755178, 18.8533173
1: -11.8887796, 32.4540100, -8.7665911, 24.0765839, -35.9653625, 41.2205887
2: -18.3148880, 30.0150185, -13.4583864, 22.1821671, -40.4970551, 43.4734001
3: -9.9472942, 37.9196472, -7.3203278, 28.1978359, -38.1451302, 45.2399750
4: -16.4736309, 26.3303986, -12.0851660, 19.4590378, -35.9326706, 38.4155655

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.0803533, 15.2318897, -3.9038270, 14.5659609, -18.6463146, 19.1357155
1: -11.4969158, 31.3768654, -11.0152178, 30.0431213, -41.5400391, 42.3920746
2: -17.7178612, 28.9725952, -17.0000057, 27.7150631, -45.4329224, 45.9725990
3: -9.6181650, 36.6772881, -9.2172012, 35.1124496, -44.7306137, 45.8944893
4: -15.9280787, 25.4270325, -15.2773285, 24.3134193, -40.2414932, 40.7043610

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7441529
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7605794
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.1268620, 15.4032192, -4.1563940, 15.5011339, -19.6279945, 19.5596085
1: -11.6270456, 31.7293167, -11.7120266, 31.9809704, -43.6080170, 43.4413452
2: -17.9145794, 29.3193798, -18.0555115, 29.5564137, -47.4709930, 47.3748894
3: -9.7269125, 37.0954361, -9.8002186, 37.3707771, -47.0976868, 46.8956451
4: -16.1073837, 25.7313728, -16.2366905, 25.9281597, -42.0355453, 41.9680634

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7441529
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7612594
time: 1.32 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.95 seconds
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7441529
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7605794
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.95
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7441529
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.95
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7612594

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0459719, 15.1032467, -3.9038270, 14.5659609, -18.6119328, 19.0070724
1: -11.3990774, 31.1168232, -11.0152178, 30.0431213, -41.4421997, 42.1320419
2: -17.5763721, 28.7252522, -17.0000057, 27.7150631, -45.2914276, 45.7252579
3: -9.5364075, 36.3867760, -9.2172012, 35.1124496, -44.6488571, 45.6039772
4: -15.7984543, 25.2109222, -15.2773285, 24.3134193, -40.1118698, 40.4882507

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7452229, upper bound: 38.7605794
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7446620, upper bound: 38.7592104
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0459719, 15.1032467, -4.1563940, 15.5011339, -19.5471058, 19.2596397
1: -11.3990774, 31.1168232, -11.7120266, 31.9809704, -43.3800468, 42.8288498
2: -17.5763721, 28.7252522, -18.0555115, 29.5564137, -47.1327858, 46.7807617
3: -9.5364075, 36.3867760, -9.8002186, 37.3707771, -46.9071846, 46.1869926
4: -15.7984543, 25.2109222, -16.2366905, 25.9281597, -41.7266121, 41.4476089

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.19 + 83.15 = 86.34 seconds
