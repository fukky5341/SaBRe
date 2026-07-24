## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 42.134888934


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.6358709, 7.2639017, -8.6358709, 7.2639017, -15.8997726, 15.8997726)
1: (-32.9253387, 26.6185799, -32.9253387, 26.6185799, -59.5439186, 59.5439186)
2: (-17.6399612, 27.3018894, -17.6399612, 27.3018894, -44.9418449, 44.9418449)
3: (-29.9307785, 24.8325806, -29.9307785, 24.8325806, -54.7633591, 54.7633591)
4: (-22.0805244, 27.8851662, -22.0805244, 27.8851662, -49.9656906, 49.9656906)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.02 + 2.35 = 3.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -42.1770660, upper bound: 42.1770660

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1743079, upper bound: 42.1769187
time: 0.59 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1769907, upper bound: 42.1769907
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.26 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -42.1743079, upper bound: 42.1769187
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -42.1769907, upper bound: 42.1769907

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.4973760, 6.3254008, -8.3987980, 7.0755973, -14.5729733, 14.7241993
1: -28.5568676, 23.2108345, -32.0141449, 25.9364986, -54.4933624, 55.2249794
2: -15.4047670, 23.8262024, -17.1840267, 26.6181126, -42.0228767, 41.0102158
3: -25.9883919, 21.6498375, -29.1005707, 24.2021446, -50.1905365, 50.7504082
4: -19.1922970, 24.3685474, -21.4715080, 27.1967010, -46.3889999, 45.8400574

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1742359, upper bound: 42.1742359
time: 0.56 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1742359, upper bound: 42.1769187
time: 0.51 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.3875084, 7.0410771, -8.6358709, 7.2639017, -15.6514101, 15.6769476
1: -31.9755249, 25.7738094, -32.9253387, 26.6185799, -58.5941048, 58.6991501
2: -17.1077137, 26.5254345, -17.6399612, 27.3018894, -44.4096031, 44.1653976
3: -29.0687828, 24.0699139, -29.9307785, 24.8325806, -53.9013634, 54.0006943
4: -21.4405136, 27.0563068, -22.0805244, 27.8851662, -49.3256798, 49.1368256

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1769187, upper bound: 42.1743079
time: 0.64 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1769187, upper bound: 42.1769907
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.29 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 4, lower bound: -42.1742359, upper bound: 42.1742359
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 4, lower bound: -42.1742359, upper bound: 42.1769187
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 4, lower bound: -42.1769187, upper bound: 42.1743079
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 4, lower bound: -42.1769187, upper bound: 42.1769907

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.4973760, 6.3254008, -7.4973760, 6.3254008, -13.8227768, 13.8227768
1: -28.5568676, 23.2108345, -28.5568676, 23.2108345, -51.7676926, 51.7676964
2: -15.4047670, 23.8262024, -15.4047670, 23.8262024, -39.2309685, 39.2309685
3: -25.9883919, 21.6498375, -25.9883919, 21.6498375, -47.6382294, 47.6382294
4: -19.1922970, 24.3685474, -19.1922970, 24.3685474, -43.5608444, 43.5608444

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1260411, upper bound: 42.1691717
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1739502, upper bound: 42.1739502
time: 0.64 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.4973760, 6.3254008, -8.3875084, 7.0410771, -14.5384531, 14.7129078
1: -28.5568676, 23.2108345, -31.9755249, 25.7738094, -54.3306732, 55.1863594
2: -15.4047670, 23.8262024, -17.1077137, 26.5254345, -41.9302025, 40.9339142
3: -25.9883919, 21.6498375, -29.0687828, 24.0699139, -50.0583038, 50.7186203
4: -19.1922970, 24.3685474, -21.4405136, 27.0563068, -46.2486038, 45.8090591

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1260411, upper bound: 42.1691717
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1739502, upper bound: 42.1768725
time: 0.61 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.3875084, 7.0410771, -7.4973760, 6.3254008, -14.7129087, 14.5384531
1: -31.9755249, 25.7738094, -28.5568676, 23.2108345, -55.1863556, 54.3306732
2: -17.1077137, 26.5254345, -15.4047670, 23.8262024, -40.9339142, 41.9302025
3: -29.0687828, 24.0699139, -25.9883919, 21.6498375, -50.7186203, 50.0583038
4: -21.4405136, 27.0563068, -19.1922970, 24.3685474, -45.8090591, 46.2486038

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1637077, upper bound: 42.1740877
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1769161, upper bound: 42.1743014
time: 0.62 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.3875084, 7.0410771, -8.3875084, 7.0410771, -15.4285851, 15.4285831
1: -31.9755249, 25.7738094, -31.9755249, 25.7738094, -57.7493362, 57.7493362
2: -17.1077137, 26.5254345, -17.1077137, 26.5254345, -43.6331482, 43.6331482
3: -29.0687828, 24.0699139, -29.0687828, 24.0699139, -53.1386948, 53.1386948
4: -21.4405136, 27.0563068, -21.4405136, 27.0563068, -48.4968185, 48.4968185

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1637077, upper bound: 42.1767706
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1769161, upper bound: 42.1769687
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.16 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 4, lower bound: -42.1260411, upper bound: 42.1691717
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 4, lower bound: -42.1739502, upper bound: 42.1739502
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 4, lower bound: -42.1260411, upper bound: 42.1691717
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 4, lower bound: -42.1739502, upper bound: 42.1768725
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 4, lower bound: -42.1637077, upper bound: 42.1740877
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 4, lower bound: -42.1769161, upper bound: 42.1743014
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 4, lower bound: -42.1637077, upper bound: 42.1767706
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.16
Output dim: 4, lower bound: -42.1769161, upper bound: 42.1769687

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.6363592, 5.5968761, -7.4121857, 6.2585902, -12.8949461, 13.0090570
1: -25.1734962, 20.5544987, -28.2178783, 22.9726410, -48.1461372, 48.7723770
2: -13.6732416, 21.2471561, -15.2469120, 23.5917091, -37.2649498, 36.4940643
3: -22.9342060, 19.2104855, -25.6803684, 21.4309807, -44.3651886, 44.8908424
4: -16.9686089, 21.7119408, -18.9719505, 24.1325378, -41.1011391, 40.6838875

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1255735, upper bound: 42.1255735
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1255735, upper bound: 42.1697237
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3509603, 6.2078753, -7.4928827, 6.3217840, -13.6727448, 13.7007580
1: -27.9853477, 22.7828808, -28.5393467, 23.1976337, -51.1829758, 51.3222275
2: -15.1225176, 23.3899498, -15.3961191, 23.8128529, -38.9353714, 38.7860680
3: -25.4735489, 21.2494183, -25.9726086, 21.6375179, -47.1110687, 47.2220268
4: -18.8193302, 23.9285812, -19.1808605, 24.3550682, -43.1743889, 43.1094398

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1697237, upper bound: 42.1260686
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1697237, upper bound: 42.1739502
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.6363592, 5.5968761, -8.2991142, 6.9714713, -13.6078300, 13.8959885
1: -25.1734962, 20.5544987, -31.6248512, 25.5233555, -50.6968536, 52.1793518
2: -13.6732416, 21.2471561, -16.9425354, 26.2787991, -39.9520416, 38.1896858
3: -22.9342060, 19.2104855, -28.7504349, 23.8398304, -46.7740326, 47.9609146
4: -16.9686089, 21.7119408, -21.2124043, 26.8072796, -43.7758865, 42.9243431

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1253730, upper bound: 42.1078656
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1253730, upper bound: 42.1691717
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.3509603, 6.2078753, -8.3835859, 7.0379605, -14.3889208, 14.5914612
1: -27.9853477, 22.7828808, -31.9601879, 25.7625008, -53.7478447, 54.7430649
2: -15.1225176, 23.3899498, -17.1001511, 26.5138435, -41.6363525, 40.4900932
3: -25.4735489, 21.2494183, -29.0549507, 24.0593090, -49.5328598, 50.3043633
4: -18.8193302, 23.9285812, -21.4304829, 27.0446568, -45.8639755, 45.3590622

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1698191, upper bound: 42.1083787
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1697237, upper bound: 42.1768725
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.9736938, 5.9385271, -7.3852429, 6.2335124, -13.2072039, 13.3237696
1: -26.4600143, 21.8084393, -28.1275520, 22.8759460, -49.3359566, 49.9359894
2: -14.4261379, 22.4957504, -15.1830044, 23.4909534, -37.9170914, 37.6787567
3: -24.0550556, 20.4266109, -25.5961800, 21.3395977, -45.3946533, 46.0227890
4: -17.7938061, 23.0525379, -18.9022446, 24.0283012, -41.8221054, 41.9547806

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1245666, upper bound: 42.1249692
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1632747, upper bound: 42.1738048
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.1691628, 6.8648739, -7.4973760, 6.3254008, -14.4945621, 14.3622484
1: -31.1458073, 25.1323471, -28.5568676, 23.2108345, -54.3566399, 53.6892166
2: -16.6797314, 25.9045086, -15.4047670, 23.8262024, -40.5059280, 41.3092728
3: -28.3043690, 23.4843750, -25.9883919, 21.6498375, -49.9542046, 49.4727669
4: -20.8562508, 26.4280434, -19.1922970, 24.3685474, -45.2248001, 45.6203346

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1691690, upper bound: 42.1260365
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1768701, upper bound: 42.1260365
time: 2.92 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.9736938, 5.9385271, -8.2833204, 6.9544764, -13.9281693, 14.2218475
1: -26.4600143, 21.8084393, -31.5722313, 25.4567623, -51.9167786, 53.3806686
2: -14.4261379, 22.4957504, -16.9002991, 26.2072392, -40.6333771, 39.3960419
3: -24.0550556, 20.4266109, -28.7045307, 23.7745686, -47.8296242, 49.1311340
4: -17.7938061, 23.0525379, -21.1718330, 26.7325668, -44.5263748, 44.2243652

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1635622, upper bound: 42.1635622
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1635622, upper bound: 42.1767706
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.1691628, 6.8648739, -8.3875084, 7.0410771, -15.2102385, 15.2523785
1: -31.1458073, 25.1323471, -31.9755249, 25.7738094, -56.9196167, 57.1078720
2: -16.6797314, 25.9045086, -17.1077137, 26.5254345, -43.2051659, 43.0122223
3: -28.3043690, 23.4843750, -29.0687828, 24.0699139, -52.3742828, 52.5531578
4: -20.8562508, 26.4280434, -21.4405136, 27.0563068, -47.9125595, 47.8685532

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1635622, upper bound: 42.1635897
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1635622, upper bound: 42.1769687
time: 0.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.39 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1255735, upper bound: 42.1255735
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1255735, upper bound: 42.1697237
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1697237, upper bound: 42.1260686
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1697237, upper bound: 42.1739502
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1253730, upper bound: 42.1078656
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1253730, upper bound: 42.1691717
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1698191, upper bound: 42.1083787
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1697237, upper bound: 42.1768725
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1245666, upper bound: 42.1249692
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1632747, upper bound: 42.1738048
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1691690, upper bound: 42.1260365
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1768701, upper bound: 42.1260365
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1635622, upper bound: 42.1635622
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1635622, upper bound: 42.1767706
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1635622, upper bound: 42.1635897
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 4, lower bound: -42.1635622, upper bound: 42.1769687

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.6363592, 5.5968761, -7.3509603, 6.2078753, -12.8442345, 12.9478350
1: -25.1734962, 20.5544987, -27.9853477, 22.7828808, -47.9563751, 48.5398483
2: -13.6732416, 21.2471561, -15.1225176, 23.3899498, -37.0631905, 36.3696632
3: -22.9342060, 19.2104855, -25.4735489, 21.2494183, -44.1836243, 44.6840248
4: -16.9686089, 21.7119408, -18.8193302, 23.9285812, -40.8971901, 40.5312614

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1255735, upper bound: 42.1436261
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1255735, upper bound: 42.1436369
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.3509603, 6.2078753, -6.6363592, 5.5968761, -12.9478340, 12.8442345
1: -27.9853477, 22.7828808, -25.1734962, 20.5544987, -48.5398445, 47.9563751
2: -15.1225176, 23.3899498, -13.6732416, 21.2471561, -36.3696632, 37.0631905
3: -25.4735489, 21.2494183, -22.9342060, 19.2104855, -44.6840248, 44.1836243
4: -18.8193302, 23.9285812, -16.9686089, 21.7119408, -40.5312653, 40.8971901

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1686871, upper bound: 42.1260686
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1436369, upper bound: 42.1258456
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.3509603, 6.2078753, -7.3509603, 6.2078753, -13.5588360, 13.5588360
1: -27.9853477, 22.7828808, -27.9853477, 22.7828808, -50.7682266, 50.7682266
2: -15.1225176, 23.3899498, -15.1225176, 23.3899498, -38.5124664, 38.5124626
3: -25.4735489, 21.2494183, -25.4735489, 21.2494183, -46.7229652, 46.7229652
4: -18.8193302, 23.9285812, -18.8193302, 23.9285812, -42.7479057, 42.7479057

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1686871, upper bound: 42.1441936
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1436369, upper bound: 42.1439091
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.6363592, 5.5968761, -8.2625847, 6.9419351, -13.5782928, 13.8594589
1: -25.1734962, 20.5544987, -31.4886589, 25.4148350, -50.5883255, 52.0431595
2: -13.6732416, 21.2471561, -16.8655243, 26.1565704, -39.8298073, 38.1126747
3: -22.9342060, 19.2104855, -28.6281490, 23.7340088, -46.6682129, 47.8386192
4: -16.9686089, 21.7119408, -21.1199703, 26.6856899, -43.6542969, 42.8319092

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0948163, upper bound: 42.1673299
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1235567, upper bound: 42.1684481
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.3509603, 6.2078753, -7.5209236, 6.3026705, -13.6536303, 13.7287989
1: -27.9853477, 22.7828808, -28.5851860, 23.0766869, -51.0620346, 51.3680649
2: -15.1225176, 23.3899498, -15.3422337, 23.8997211, -39.0222321, 38.7321854
3: -25.4735489, 21.2494183, -26.0065804, 21.5886993, -47.0622444, 47.2559967
4: -18.8193302, 23.9285812, -19.2102718, 24.3456821, -43.1650009, 43.1388550

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1401044, upper bound: 42.1051714
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1695372, upper bound: 42.1055585
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.3509603, 6.2078753, -8.2625847, 6.9419351, -14.2928944, 14.4704599
1: -27.9853477, 22.7828808, -31.4886589, 25.4148350, -53.4001732, 54.2715378
2: -15.1225176, 23.3899498, -16.8655243, 26.1565704, -41.2790794, 40.2554741
3: -25.4735489, 21.2494183, -28.6281490, 23.7340088, -49.2075577, 49.8775597
4: -18.8193302, 23.9285812, -21.1199703, 26.6856899, -45.5050087, 45.0485535

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1401045, upper bound: 42.1404450
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1695372, upper bound: 42.1742900
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.9701438, 5.9357371, -7.2470341, 6.1227536, -13.0928974, 13.1827698
1: -26.4461613, 21.7982769, -27.5882835, 22.4730587, -48.9192200, 49.3865585
2: -14.4193249, 22.4855137, -14.9165487, 23.0814457, -37.5007668, 37.4020538
3: -24.0424576, 20.4171448, -25.1098156, 20.9627724, -45.0052299, 45.5269623
4: -17.7846031, 23.0422382, -18.5490856, 23.6149101, -41.3995132, 41.5913200

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1608390, upper bound: 42.1737387
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1632582, upper bound: 42.1721778
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0655781, upper bound: 42.1264809
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.0806828, 6.7952332, -6.6363592, 5.5968761, -13.6775560, 13.4315910
1: -30.7951584, 24.8817673, -25.1734962, 20.5544987, -51.3496552, 50.0552635
2: -16.5143604, 25.6582832, -13.6732416, 21.2471561, -37.7615128, 39.3315239
3: -27.9857521, 23.2541580, -22.9342060, 19.2104855, -47.1962318, 46.1883621
4: -20.6278496, 26.1793289, -16.9686089, 21.7119408, -42.3397903, 43.1479301

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1632043, upper bound: 42.1242197
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1632043, upper bound: 42.1242054
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.1652718, 6.8617740, -7.3509603, 6.2078753, -14.3731470, 14.2127342
1: -31.1305790, 25.1210766, -27.9853477, 22.7828808, -53.9134598, 53.1064224
2: -16.6722584, 25.8930435, -15.1225176, 23.3899498, -40.0622101, 41.0155563
3: -28.2906494, 23.4738121, -25.4735489, 21.2494183, -49.5400696, 48.9473534
4: -20.8463230, 26.4164886, -18.8193302, 23.9285812, -44.7749023, 45.2358055

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1083762, upper bound: 42.1698155
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1083762, upper bound: 42.1698155
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.9736938, 5.9385271, -6.9736938, 5.9385271, -12.9122200, 12.9122200
1: -26.4600143, 21.8084393, -26.4600143, 21.8084393, -48.2684555, 48.2684555
2: -14.4261379, 22.4957504, -14.4261379, 22.4957504, -36.9218864, 36.9218903
3: -24.0550556, 20.4266109, -24.0550556, 20.4266109, -44.4816666, 44.4816666
4: -17.7938061, 23.0525379, -17.7938061, 23.0525379, -40.8463440, 40.8463440

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0234853, upper bound: 42.1335979
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1631316, upper bound: 42.1631318
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.9736938, 5.9385271, -8.1691628, 6.8648739, -13.8385639, 14.1076899
1: -26.4600143, 21.8084393, -31.1458073, 25.1323471, -51.5923615, 52.9542465
2: -14.4261379, 22.4957504, -16.6797314, 25.9045086, -40.3306389, 39.1754799
3: -24.0550556, 20.4266109, -28.3043690, 23.4843750, -47.5394287, 48.7309799
4: -17.7938061, 23.0525379, -20.8562508, 26.4280434, -44.2218437, 43.9087906

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0234853, upper bound: 42.1428631
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1631316, upper bound: 42.1767271
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.1691628, 6.8648739, -6.9736938, 5.9385271, -14.1076899, 13.8385639
1: -31.1458073, 25.1323471, -26.4600143, 21.8084393, -52.9542465, 51.5923615
2: -16.6797314, 25.9045086, -14.4261379, 22.4957504, -39.1754837, 40.3306389
3: -28.3043690, 23.4843750, -24.0550556, 20.4266109, -48.7309799, 47.5394287
4: -20.8562508, 26.4280434, -17.7938061, 23.0525379, -43.9087906, 44.2218437

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0234853, upper bound: 42.1325977
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1631316, upper bound: 42.1631595
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.1691628, 6.8648739, -8.1691628, 6.8648739, -15.0340338, 15.0340338
1: -31.1458073, 25.1323471, -31.1458073, 25.1323471, -56.2781525, 56.2781525
2: -16.6797314, 25.9045086, -16.6797314, 25.9045086, -42.5842400, 42.5842361
3: -28.3043690, 23.4843750, -28.3043690, 23.4843750, -51.7887421, 51.7887421
4: -20.8562508, 26.4280434, -20.8562508, 26.4280434, -47.2842941, 47.2842941

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0283289, upper bound: 42.1692642
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1631316, upper bound: 42.1769274
time: 0.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.29 seconds
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1255735, upper bound: 42.1436261
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1255735, upper bound: 42.1436369
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1686871, upper bound: 42.1260686
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1436369, upper bound: 42.1258456
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1686871, upper bound: 42.1441936
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1436369, upper bound: 42.1439091
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.0948163, upper bound: 42.1673299
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1235567, upper bound: 42.1684481
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1401044, upper bound: 42.1051714
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1695372, upper bound: 42.1055585
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1401045, upper bound: 42.1404450
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1695372, upper bound: 42.1742900
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1632582, upper bound: 42.1721778
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.0655781, upper bound: 42.1264809
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1632043, upper bound: 42.1242197
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1632043, upper bound: 42.1242054
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1083762, upper bound: 42.1698155
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1083762, upper bound: 42.1698155
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.0234853, upper bound: 42.1335979
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1631316, upper bound: 42.1631318
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.0234853, upper bound: 42.1428631
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1631316, upper bound: 42.1767271
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.0234853, upper bound: 42.1325977
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1631316, upper bound: 42.1631595
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.0283289, upper bound: 42.1692642
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 4, lower bound: -42.1631316, upper bound: 42.1769274

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.4883018, 5.4759121, -7.3509603, 6.2078753, -12.6961765, 12.8268719
1: -24.5961266, 20.1034985, -27.9853477, 22.7828808, -47.3789978, 48.0888443
2: -13.3884697, 20.8137512, -15.1225176, 23.3899498, -36.7784157, 35.9362602
3: -22.4122162, 18.7938576, -25.4735489, 21.2494183, -43.6616249, 44.2674026
4: -16.5840015, 21.2673397, -18.8193302, 23.9285812, -40.5125809, 40.0866623

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1255627, upper bound: 42.1436261
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1258456, upper bound: 42.1436261
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.6365323, 5.5056200, -7.2586045, 6.1347079, -12.7712402, 12.7642231
1: -25.2478600, 20.1276054, -27.6238670, 22.5156059, -47.7634583, 47.7514725
2: -13.4587021, 20.9614220, -14.9481897, 23.1326809, -36.5913773, 35.9096107
3: -23.0048962, 18.8198586, -25.1452007, 21.0026264, -44.0075226, 43.9650574
4: -16.9144535, 21.2759056, -18.5788727, 23.6678524, -40.5823059, 39.8547783

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1246156, upper bound: 42.1316683
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1246387, upper bound: 42.1435997
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.2030978, 6.0868549, -6.6363592, 5.5968761, -12.7999716, 12.7232141
1: -27.4097786, 22.3412838, -25.1734962, 20.5544987, -47.9642792, 47.5147743
2: -14.8383522, 22.9614334, -13.6732416, 21.2471561, -36.0855064, 36.6346703
3: -24.9513969, 20.8417320, -22.9342060, 19.2104855, -44.1618805, 43.7759399
4: -18.4353981, 23.4920616, -16.9686089, 21.7119408, -40.1473389, 40.4606628

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1436261, upper bound: 42.1258452
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1436261, upper bound: 42.1258456
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.4157219, 6.1576800, -6.5518155, 5.5312324, -12.9469547, 12.7094955
1: -28.3268032, 22.5180664, -24.8413143, 20.3106174, -48.6374130, 47.3593674
2: -14.9386253, 23.2305889, -13.5147610, 21.0167294, -35.9553528, 36.7453499
3: -25.7925682, 20.9962559, -22.6320229, 18.9869423, -44.7795105, 43.6282692
4: -18.9443398, 23.6593857, -16.7477226, 21.4772263, -40.4215660, 40.4070969

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1435766, upper bound: 42.1258157
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1435997, upper bound: 42.1249185
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.2030978, 6.0868549, -7.3509603, 6.2078753, -13.4109726, 13.4378147
1: -27.4097786, 22.3412838, -27.9853477, 22.7828808, -50.1926575, 50.3266220
2: -14.8383522, 22.9614334, -15.1225176, 23.3899498, -38.2283020, 38.0839424
3: -24.9513969, 20.8417320, -25.4735489, 21.2494183, -46.2008133, 46.3152809
4: -18.4353981, 23.4920616, -18.8193302, 23.9285812, -42.3639755, 42.3113785

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1438286, upper bound: 42.1439091
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1438286, upper bound: 42.1439091
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.4157219, 6.1576800, -7.2586045, 6.1347079, -13.5504303, 13.4162846
1: -28.3268032, 22.5180664, -27.6238670, 22.5156059, -50.8423996, 50.1419296
2: -14.9386253, 23.2305889, -14.9481897, 23.1326809, -38.0713005, 38.1787796
3: -25.7925682, 20.9962559, -25.1452007, 21.0026264, -46.7951965, 46.1414452
4: -18.9443398, 23.6593857, -18.5788727, 23.6678524, -42.6121902, 42.2382507

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1436785, upper bound: 42.1319481
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1437978, upper bound: 42.1438796
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.3462434, 5.3757463, -8.2625847, 6.9419351, -13.2881775, 13.6383305
1: -24.0655079, 19.7904472, -31.4886589, 25.4148350, -49.4803314, 51.2791061
2: -13.0577421, 20.3897095, -16.8655243, 26.1565704, -39.2143059, 37.2552299
3: -21.8915710, 18.4972954, -28.6281490, 23.7340088, -45.6255798, 47.1254311
4: -16.2021866, 20.8756046, -21.1199703, 26.6856899, -42.8878784, 41.9955711

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0954658, upper bound: 42.1613148
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0954658, upper bound: 42.1673299
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.4707246, 5.4809747, -8.2625847, 6.9419351, -13.4126587, 13.7435570
1: -24.5124283, 20.1439152, -31.4886589, 25.4148350, -49.9272423, 51.6325760
2: -13.3955460, 20.8377399, -16.8655243, 26.1565704, -39.5521126, 37.7032585
3: -22.3166428, 18.8336010, -28.6281490, 23.7340088, -46.0506516, 47.4617348
4: -16.5297852, 21.2963467, -21.1199703, 26.6856899, -43.2154694, 42.4163094

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1242110, upper bound: 42.1632065
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1242110, upper bound: 42.1684481
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.0730286, 5.9704599, -7.5209236, 6.3026705, -13.3756990, 13.4913816
1: -26.9138279, 21.9011765, -28.5851860, 23.0766869, -49.9905167, 50.4863472
2: -14.5541716, 22.5381660, -15.3422337, 23.8997211, -38.4538879, 37.8804016
3: -24.5055180, 20.4324741, -26.0065804, 21.5886993, -46.0942154, 46.4390488
4: -18.1029053, 23.0376472, -19.2102718, 24.3456821, -42.4485741, 42.2479095

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1386000, upper bound: 42.0889435
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1386000, upper bound: 42.1028062
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.3855829, 6.2298532, -7.5209236, 6.3026705, -13.6882534, 13.7507763
1: -28.1211166, 22.8619728, -28.5851860, 23.0766869, -51.1978035, 51.4471550
2: -15.1779146, 23.4398193, -15.3422337, 23.8997211, -39.0776253, 38.7820511
3: -25.6057873, 21.3124771, -26.0065804, 21.5886993, -47.1944809, 47.3190536
4: -18.9243641, 23.9788113, -19.2102718, 24.3456821, -43.2700424, 43.1890831

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1694793, upper bound: 42.1055407
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1683911, upper bound: 42.0893071
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1689699, upper bound: 42.1031772
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.0730286, 5.9704599, -8.2625847, 6.9419351, -14.0149632, 14.2330418
1: -26.9138279, 21.9011765, -31.4886589, 25.4148350, -52.3286629, 53.3898239
2: -14.5541716, 22.5381660, -16.8655243, 26.1565704, -40.7107391, 39.4036865
3: -24.5055180, 20.4324741, -28.6281490, 23.7340088, -48.2395248, 49.0606079
4: -18.1029053, 23.0376472, -21.1199703, 26.6856899, -44.7885895, 44.1576080

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1389593, upper bound: 42.1274169
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1389947, upper bound: 42.1389944
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.3855829, 6.2298532, -8.2625847, 6.9419351, -14.3275175, 14.4924355
1: -28.1211166, 22.8619728, -31.4886589, 25.4148350, -53.5359497, 54.3506317
2: -15.1779146, 23.4398193, -16.8655243, 26.1565704, -41.3344727, 40.3053436
3: -25.6057873, 21.3124771, -28.6281490, 23.7340088, -49.3397980, 49.9406166
4: -18.9243641, 23.9788113, -21.1199703, 26.6856899, -45.6100502, 45.0987816

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1735056, upper bound: 42.1695619
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1734845, upper bound: 42.1737097
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.5909829, 5.5873561, -7.2074032, 6.0866323, -12.6776133, 12.7947598
1: -24.9450760, 20.4672585, -27.4365196, 22.3355846, -47.2806587, 47.9037781
2: -13.6418095, 21.2233734, -14.8344498, 22.9508533, -36.5926590, 36.0578194
3: -22.7296696, 19.1634750, -24.9747429, 20.8342152, -43.5638847, 44.1382179
4: -16.8233032, 21.7049599, -18.4487820, 23.4765625, -40.2998657, 40.1537399

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1588711, upper bound: 42.1681300
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1588498, upper bound: 42.1717885
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.7911348, 6.5721350, -6.6363592, 5.5968761, -13.3880072, 13.2084932
1: -29.6847992, 24.1055336, -25.1734962, 20.5544987, -50.2392960, 49.2790298
2: -15.8980675, 24.7937336, -13.6732416, 21.2471561, -37.1452179, 38.4669685
3: -26.9438381, 22.5292301, -22.9342060, 19.2104855, -46.1543083, 45.4634323
4: -19.8655491, 25.3290195, -16.9686089, 21.7119408, -41.5774918, 42.2976227

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1631470, upper bound: 42.1241719
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1601177, upper bound: 42.1233116
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.1372395, 6.8510528, -6.6363592, 5.5968761, -13.7341137, 13.4874105
1: -30.9891644, 25.1049747, -25.1734962, 20.5544987, -51.5436630, 50.2784691
2: -16.6546764, 25.8586311, -13.6732416, 21.2471561, -37.9018211, 39.5318565
3: -28.1455193, 23.4581947, -22.9342060, 19.2104855, -47.3560028, 46.3924026
4: -20.7656860, 26.3863869, -16.9686089, 21.7119408, -42.4776268, 43.3549881

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1683858, upper bound: 42.1241575
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1666439, upper bound: 42.1232973
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3067865, 6.1290946, -7.3509603, 6.2078753, -13.5146618, 13.4800520
1: -27.7741547, 22.4439907, -27.9853477, 22.7828808, -50.5570374, 50.4293365
2: -14.9191265, 23.2865257, -15.1225176, 23.3899498, -38.3090744, 38.4090309
3: -25.2660370, 21.0110912, -25.4735489, 21.2494183, -46.5154572, 46.4846344
4: -18.6393318, 23.7244930, -18.8193302, 23.9285812, -42.5679131, 42.5438118

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1048712, upper bound: 42.1401021
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1050495, upper bound: 42.1695348
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.0292177, 6.7537384, -7.3509603, 6.2078753, -14.2370930, 14.1046972
1: -30.5981636, 24.7288456, -27.9853477, 22.7828808, -53.3810425, 52.7141953
2: -16.4103374, 25.4915905, -15.1225176, 23.3899498, -39.8002853, 40.6141090
3: -27.8108501, 23.1059799, -25.4735489, 21.2494183, -49.0602608, 48.5795250
4: -20.4988327, 26.0128632, -18.8193302, 23.9285812, -44.4274139, 44.8321838

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1048712, upper bound: 42.1116906
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1050495, upper bound: 42.1253382
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.8836098, 5.8665242, -6.9701438, 5.9357371, -12.8193474, 12.8366680
1: -26.1062317, 21.5468769, -26.4461613, 21.7982769, -47.9045067, 47.9930382
2: -14.2515154, 22.2294064, -14.4193249, 22.4855137, -36.7370300, 36.6487274
3: -23.7357693, 20.1815529, -24.0424576, 20.4171448, -44.1529160, 44.2240067
4: -17.5615540, 22.7838631, -17.7846031, 23.0422382, -40.6037903, 40.5684662

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0659848, upper bound: 42.1565692
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0652363, upper bound: 42.0652363
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.1073685, 5.1969953, -8.0806828, 6.7952332, -12.9026012, 13.2776775
1: -23.0096493, 19.0621185, -30.7951584, 24.8817673, -47.8914185, 49.8572769
2: -12.6603308, 19.8433228, -16.5143604, 25.6582832, -38.3186150, 36.3576813
3: -20.9727955, 17.9096909, -27.9857521, 23.2541580, -44.2269516, 45.8954430
4: -15.5533142, 20.3062344, -20.6278496, 26.1793289, -41.7326317, 40.9340820

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0230807, upper bound: 42.1006452
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0230807, upper bound: 42.1428631
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.8836098, 5.8665242, -8.1652718, 6.8617740, -13.7453842, 14.0317955
1: -26.1062317, 21.5468769, -31.1305790, 25.1210766, -51.2273102, 52.6774521
2: -14.2515154, 22.2294064, -16.6722584, 25.8930435, -40.1445580, 38.9016647
3: -23.7357693, 20.1815529, -28.2906494, 23.4738121, -47.2095795, 48.4722023
4: -17.5615540, 22.7838631, -20.8463230, 26.4164886, -43.9780350, 43.6301880

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1322629, upper bound: 42.1079591
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1322629, upper bound: 42.1767271
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.0292177, 6.7537384, -6.9701438, 5.9357371, -13.9649544, 13.7238808
1: -30.5981636, 24.7288456, -26.4461613, 21.7982769, -52.3964386, 51.1750069
2: -16.4103374, 25.4915905, -14.4193249, 22.4855137, -38.8958473, 39.9109154
3: -27.8108501, 23.1059799, -24.0424576, 20.4171448, -48.2279892, 47.1484337
4: -20.4988327, 26.0128632, -17.7846031, 23.0422382, -43.5410652, 43.7974625

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1275024, upper bound: 42.0234571
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1275024, upper bound: 42.1631595
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3067865, 6.1290946, -8.0806828, 6.7952332, -14.1020203, 14.2097740
1: -27.7741547, 22.4439907, -30.7951584, 24.8817673, -52.6559219, 53.2391510
2: -14.9191265, 23.2865257, -16.5143604, 25.6582832, -40.5774078, 39.8008766
3: -25.2660370, 21.0110912, -27.9857521, 23.2541580, -48.5201950, 48.9968414
4: -18.6393318, 23.7244930, -20.6278496, 26.1793289, -44.8186569, 44.3523407

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0666958, upper bound: 42.1448053
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1037384, upper bound: 42.1621118
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.0292177, 6.7537384, -8.1652718, 6.8617740, -14.8909912, 14.9190083
1: -30.5981636, 24.7288456, -31.1305790, 25.1210766, -55.7192383, 55.8594246
2: -16.4103374, 25.4915905, -16.6722584, 25.8930435, -42.3033752, 42.1638489
3: -27.8108501, 23.1059799, -28.2906494, 23.4738121, -51.2846489, 51.3966293
4: -20.4988327, 26.0128632, -20.8463230, 26.4164886, -46.9153099, 46.8591805

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1677375
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1723240, upper bound: 42.1723665
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.18 seconds
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1255627, upper bound: 42.1436261
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1258456, upper bound: 42.1436261
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1246156, upper bound: 42.1316683
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1246387, upper bound: 42.1435997
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1436261, upper bound: 42.1258452
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1436261, upper bound: 42.1258456
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1435766, upper bound: 42.1258157
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1435997, upper bound: 42.1249185
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1438286, upper bound: 42.1439091
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1438286, upper bound: 42.1439091
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1436785, upper bound: 42.1319481
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1437978, upper bound: 42.1438796
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.0954658, upper bound: 42.1613148
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.0954658, upper bound: 42.1673299
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1242110, upper bound: 42.1632065
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1242110, upper bound: 42.1684481
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1386000, upper bound: 42.0889435
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1386000, upper bound: 42.1028062
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1683911, upper bound: 42.0893071
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1689699, upper bound: 42.1031772
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1389593, upper bound: 42.1274169
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1389947, upper bound: 42.1389944
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1735056, upper bound: 42.1695619
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1734845, upper bound: 42.1737097
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1588711, upper bound: 42.1681300
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1588498, upper bound: 42.1717885
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1631470, upper bound: 42.1241719
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1601177, upper bound: 42.1233116
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1683858, upper bound: 42.1241575
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1666439, upper bound: 42.1232973
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1048712, upper bound: 42.1401021
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1050495, upper bound: 42.1695348
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1048712, upper bound: 42.1116906
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1050495, upper bound: 42.1253382
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.0659848, upper bound: 42.1565692
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.0652363, upper bound: 42.0652363
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.0230807, upper bound: 42.1006452
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.0230807, upper bound: 42.1428631
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1322629, upper bound: 42.1079591
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1322629, upper bound: 42.1767271
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1275024, upper bound: 42.0234571
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1275024, upper bound: 42.1631595
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.0666958, upper bound: 42.1448053
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1037384, upper bound: 42.1621118
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1677375
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 4, lower bound: -42.1723240, upper bound: 42.1723665

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.4883018, 5.4759121, -7.2030978, 6.0868549, -12.5751572, 12.6790104
1: -24.5961266, 20.1034985, -27.4097786, 22.3412838, -46.9373932, 47.5132751
2: -13.3884697, 20.8137512, -14.8383522, 22.9614334, -36.3498955, 35.6521034
3: -22.4122162, 18.7938576, -24.9513969, 20.8417320, -43.2539444, 43.7452545
4: -16.5840015, 21.2673397, -18.4353981, 23.4920616, -40.0760574, 39.7027359

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1258162, upper bound: 42.1435687
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1121967, upper bound: 42.1434467
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1121967, upper bound: 42.1434467
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.4883018, 5.4759121, -7.4157219, 6.1576800, -12.6459818, 12.8916340
1: -24.5961266, 20.1034985, -28.3268032, 22.5180664, -47.1141853, 48.4303017
2: -13.3884697, 20.8137512, -14.9386253, 23.2305889, -36.6190567, 35.7523727
3: -22.4122162, 18.7938576, -25.7925682, 20.9962559, -43.4084549, 44.5864220
4: -16.5840015, 21.2673397, -18.9443398, 23.6593857, -40.2433853, 40.2116776

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1258162, upper bound: 42.1435687
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1121967, upper bound: 42.1434467
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1258456, upper bound: 42.1436093
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.4051156, 5.3187490, -8.0529547, 6.7291908, -13.1343040, 13.3717041
1: -24.3420372, 19.4453602, -30.6990776, 24.6742020, -49.0162392, 50.1444321
2: -13.0169783, 20.2820110, -16.4115734, 25.2890835, -38.3060608, 36.6935806
3: -22.1849937, 18.1888638, -27.9423332, 22.9853611, -45.1703415, 46.1311951
4: -16.3249531, 20.5800667, -20.6441689, 25.7811012, -42.1060486, 41.2242355

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1249185, upper bound: 42.1435766
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1249185, upper bound: 42.1435997
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.2030978, 6.0868549, -6.4883018, 5.4759121, -12.6790104, 12.5751572
1: -27.4097786, 22.3412838, -24.5961266, 20.1034985, -47.5132751, 46.9373932
2: -14.8383522, 22.9614334, -13.3884697, 20.8137512, -35.6521034, 36.3498955
3: -24.9513969, 20.8417320, -22.4122162, 18.7938576, -43.7452545, 43.2539444
4: -18.4353981, 23.4920616, -16.5840015, 21.2673397, -39.7027359, 40.0760651

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1403084, upper bound: 42.1256824
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1686703, upper bound: 42.1260548
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.2030978, 6.0868549, -6.6365323, 5.5056200, -12.7087173, 12.7233868
1: -27.4097786, 22.3412838, -25.2478600, 20.1276054, -47.5373840, 47.5891342
2: -14.8383522, 22.9614334, -13.4587021, 20.9614220, -35.7997742, 36.4201279
3: -24.9513969, 20.8417320, -23.0048962, 18.8198586, -43.7712555, 43.8466263
4: -18.4353981, 23.4920616, -16.9144535, 21.2759056, -39.7112961, 40.4065094

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1403084, upper bound: 42.1256824
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1686703, upper bound: 42.1260548
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.4157219, 6.1576800, -5.8604727, 4.9897346, -12.4054556, 12.0181522
1: -28.3268032, 22.5180664, -22.1693439, 18.3700180, -46.6968117, 44.6873970
2: -14.9386253, 23.2305889, -12.1784649, 18.9970245, -33.9356499, 35.4090538
3: -25.7925682, 20.9962559, -20.1723709, 17.1807270, -42.9732971, 41.1686134
4: -18.9443398, 23.6593857, -14.9517355, 19.4411831, -38.3855209, 38.6111145

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1316452, upper bound: 42.1247174
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1316452, upper bound: 42.1247171
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.1776657, 5.9657469, -7.1566787, 5.9604783, -13.1381435, 13.1224251
1: -27.4012547, 21.8153973, -27.2332916, 21.8822517, -49.2835083, 49.0486832
2: -14.4790144, 22.5095501, -14.5886269, 22.4808064, -36.9598198, 37.0981750
3: -24.9588032, 20.3407631, -24.8168983, 20.4122467, -45.3710480, 45.1576576
4: -18.3404579, 22.9333878, -18.3466320, 22.9031811, -41.2436371, 41.2800217

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1316683, upper bound: 42.1247174
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1316683, upper bound: 42.1249185
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.2030978, 6.0868549, -7.2030978, 6.0868549, -13.2899532, 13.2899532
1: -27.4097786, 22.3412838, -27.4097786, 22.3412838, -49.7510605, 49.7510529
2: -14.8383522, 22.9614334, -14.8383522, 22.9614334, -37.7997818, 37.7997780
3: -24.9513969, 20.8417320, -24.9513969, 20.8417320, -45.7931290, 45.7931290
4: -18.4353981, 23.4920616, -18.4353981, 23.4920616, -41.9274483, 41.9274483

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1587246, upper bound: 42.1439074
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1736363, upper bound: 42.1441815
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.2030978, 6.0868549, -7.4157219, 6.1576800, -13.3607779, 13.5025768
1: -27.4097786, 22.3412838, -28.3268032, 22.5180664, -49.9278450, 50.6680717
2: -14.8383522, 22.9614334, -14.9386253, 23.2305889, -38.0689392, 37.9000473
3: -24.9513969, 20.8417320, -25.7925682, 20.9962559, -45.9476471, 46.6343002
4: -18.4353981, 23.4920616, -18.9443398, 23.6593857, -42.0947685, 42.4363976

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1587247, upper bound: 42.1439075
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1736363, upper bound: 42.1441815
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.4157219, 6.1576800, -6.5412874, 5.5717869, -12.9875088, 12.6989670
1: -28.3268032, 22.5180664, -24.8600922, 20.5184307, -48.8452301, 47.3781395
2: -14.9386253, 23.2305889, -13.5689917, 21.0450745, -35.9836884, 36.7995796
3: -25.7925682, 20.9962559, -22.5961285, 19.1385174, -44.9310799, 43.5923843
4: -18.9443398, 23.6593857, -16.7168522, 21.5712738, -40.5156136, 40.3762321

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1317055, upper bound: 42.1317470
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1317055, upper bound: 42.1319481
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.1776657, 5.9657469, -8.0529547, 6.7291908, -13.9068556, 14.0187016
1: -27.4012547, 21.8153973, -30.6990776, 24.6742020, -52.0754547, 52.5144730
2: -14.4790144, 22.5095501, -16.4115734, 25.2890835, -39.7680893, 38.9211235
3: -24.9588032, 20.3407631, -27.9423332, 22.9853611, -47.9441566, 48.2830963
4: -18.3404579, 22.9333878, -20.6441689, 25.7811012, -44.1215591, 43.5775566

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1317448, upper bound: 42.1429100
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1317448, upper bound: 42.1438796
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.3462434, 5.3757463, -7.9733176, 6.7187214, -13.0649633, 13.3490639
1: -24.0655079, 19.7904472, -30.3791008, 24.6392021, -48.7047119, 50.1695480
2: -13.0577421, 20.3897095, -16.2489929, 25.3012714, -38.3590126, 36.6387024
3: -21.8915710, 18.4972954, -27.5864220, 23.0095901, -44.9011612, 46.0837173
4: -16.2021866, 20.8756046, -20.3548717, 25.8419018, -42.0440788, 41.2304726

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0896768, upper bound: 42.1579415
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.3462434, 5.3757463, -8.3133841, 6.9936090, -13.3398523, 13.6891308
1: -24.0655079, 19.7904472, -31.6607399, 25.6225071, -49.6880112, 51.4511871
2: -13.0577421, 20.3897095, -16.9971638, 26.3419380, -39.3996811, 37.3868637
3: -21.8915710, 18.4972954, -28.7680817, 23.9240208, -45.8155899, 47.2653770
4: -16.2021866, 20.8756046, -21.2419167, 26.8776150, -43.0797997, 42.1175156

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0896768, upper bound: 42.1650414
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.4707246, 5.4809747, -7.9733176, 6.7187214, -13.1894436, 13.4542894
1: -24.5124283, 20.1439152, -30.3791008, 24.6392021, -49.1516190, 50.5230103
2: -13.3955460, 20.8377399, -16.2489929, 25.3012714, -38.6968155, 37.0867310
3: -22.3166428, 18.8336010, -27.5864220, 23.0095901, -45.3262329, 46.4200172
4: -16.5297852, 21.2963467, -20.3548717, 25.8419018, -42.3716774, 41.6512146

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1113000, upper bound: 42.1631845
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1241845, upper bound: 42.1628095
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.4707246, 5.4809747, -8.3133841, 6.9936090, -13.4643316, 13.7943563
1: -24.5124283, 20.1439152, -31.6607399, 25.6225071, -50.1349258, 51.8046494
2: -13.3955460, 20.8377399, -16.9971638, 26.3419380, -39.7374840, 37.8348923
3: -22.3166428, 18.8336010, -28.7680817, 23.9240208, -46.2406616, 47.6016808
4: -16.5297852, 21.2963467, -21.2419167, 26.8776150, -43.4073982, 42.5382538

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1113001, upper bound: 42.1684255
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1241849, upper bound: 42.1682163
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.0730286, 5.9704599, -7.2342577, 6.0809770, -13.1540051, 13.2047138
1: -26.9138279, 21.9011765, -27.4884567, 22.3091373, -49.2229652, 49.3896294
2: -14.5541716, 22.5381660, -14.7548876, 23.0420036, -37.5961723, 37.2930450
3: -24.5055180, 20.4324741, -24.9738770, 20.8753986, -45.3809166, 45.4063454
4: -18.1029053, 23.0376472, -18.4537086, 23.5064278, -41.6093330, 41.4913483

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.0730286, 5.9704599, -7.5653849, 6.3510666, -13.4240952, 13.5358448
1: -26.9138279, 21.9011765, -28.7326012, 23.2720337, -50.1858597, 50.6337662
2: -14.5541716, 22.5381660, -15.4641953, 24.0739460, -38.6281166, 38.0023537
3: -24.5055180, 20.4324741, -26.1274834, 21.7682762, -46.2737923, 46.5599480
4: -18.1029053, 23.0376472, -19.3142757, 24.5278664, -42.6307678, 42.3519135

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.3855829, 6.2298532, -7.2342577, 6.0809770, -13.4665604, 13.4641094
1: -28.1211166, 22.8619728, -27.4884567, 22.3091373, -50.4302521, 50.3504295
2: -15.1779146, 23.4398193, -14.7548876, 23.0420036, -38.2199097, 38.1947021
3: -25.6057873, 21.3124771, -24.9738770, 20.8753986, -46.4811859, 46.2863503
4: -18.9243641, 23.9788113, -18.4537086, 23.5064278, -42.4307938, 42.4325180

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1625460, upper bound: 42.0893071
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1625460, upper bound: 42.0893071
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.3855829, 6.2298532, -7.5653849, 6.3510666, -13.7366495, 13.7952385
1: -28.1211166, 22.8619728, -28.7326012, 23.2720337, -51.3931465, 51.5945740
2: -15.1779146, 23.4398193, -15.4641953, 24.0739460, -39.2518539, 38.9040108
3: -25.6057873, 21.3124771, -26.1274834, 21.7682762, -47.3740578, 47.4399567
4: -18.9243641, 23.9788113, -19.3142757, 24.5278664, -43.4522324, 43.2930870

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1632216, upper bound: 42.1031773
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1632216, upper bound: 42.1031773
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.0730286, 5.9704599, -7.9733176, 6.7187214, -13.7917490, 13.9437742
1: -26.9138279, 21.9011765, -30.3791008, 24.6392021, -51.5530319, 52.2802620
2: -14.5541716, 22.5381660, -16.2489929, 25.3012714, -39.8554420, 38.7871590
3: -24.5055180, 20.4324741, -27.5864220, 23.0095901, -47.5151062, 48.0188942
4: -18.1029053, 23.0376472, -20.3548717, 25.8419018, -43.9447937, 43.3925133

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1351674, upper bound: 42.1101881
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.0730286, 5.9704599, -8.3133841, 6.9936090, -14.0666370, 14.2838411
1: -26.9138279, 21.9011765, -31.6607399, 25.6225071, -52.5363350, 53.5619049
2: -14.5541716, 22.5381660, -16.9971638, 26.3419380, -40.8961105, 39.5353203
3: -24.5055180, 20.4324741, -28.7680817, 23.9240208, -48.4295387, 49.2005539
4: -18.1029053, 23.0376472, -21.2419167, 26.8776150, -44.9805183, 44.2795486

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1351808, upper bound: 42.1144249
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.3855829, 6.2298532, -7.9733176, 6.7187214, -14.1043034, 14.2031689
1: -28.1211166, 22.8619728, -30.3791008, 24.6392021, -52.7603188, 53.2410698
2: -15.1779146, 23.4398193, -16.2489929, 25.3012714, -40.4791794, 39.6888123
3: -25.6057873, 21.3124771, -27.5864220, 23.0095901, -48.6153755, 48.8988991
4: -18.9243641, 23.9788113, -20.3548717, 25.8419018, -44.7662582, 44.3336830

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1521621, upper bound: 42.1497004
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1734895, upper bound: 42.1695501
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.3855829, 6.2298532, -8.3133841, 6.9936090, -14.3791924, 14.5432358
1: -28.1211166, 22.8619728, -31.6607399, 25.6225071, -53.7436218, 54.5227051
2: -15.1779146, 23.4398193, -16.9971638, 26.3419380, -41.5198479, 40.4369774
3: -25.6057873, 21.3124771, -28.7680817, 23.9240208, -49.5298080, 50.0805588
4: -18.9243641, 23.9788113, -21.2419167, 26.8776150, -45.8019791, 45.2207260

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1573787, upper bound: 42.1588058
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1734707, upper bound: 42.1736979
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.5909829, 5.5873561, -6.9216790, 5.8675747, -12.4585562, 12.5090351
1: -24.9450760, 20.4672585, -26.3487301, 21.5746021, -46.5196724, 46.8159866
2: -13.6418095, 21.2233734, -14.2259674, 22.0851326, -35.7269402, 35.4493370
3: -22.7296696, 19.1634750, -23.9505138, 20.1222572, -42.8519211, 43.1139908
4: -16.8233032, 21.7049599, -17.6975842, 22.6317577, -39.4550629, 39.4025421

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1577913, upper bound: 42.1681279
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1577913, upper bound: 42.1681300
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.5909829, 5.5873561, -7.1050310, 6.0235333, -12.6145144, 12.6923866
1: -24.9450760, 20.4672585, -27.0171547, 22.1244316, -47.0695076, 47.4844131
2: -13.6418095, 21.2233734, -14.6778831, 22.7411022, -36.3829041, 35.9012489
3: -22.7296696, 19.1634750, -24.5708675, 20.6426811, -43.3723526, 43.7343445
4: -16.8233032, 21.7049599, -18.1691284, 23.2647362, -40.0880394, 39.8740883

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1577913, upper bound: 42.1717864
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1577913, upper bound: 42.1717885
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.7911348, 6.5721350, -5.9477453, 5.0575151, -12.8486481, 12.5198803
1: -29.6847992, 24.1055336, -22.5129051, 18.6221199, -48.3069153, 46.6184387
2: -15.8980675, 24.7937336, -12.3415947, 19.2346973, -35.1327629, 37.1353226
3: -26.9438381, 22.5292301, -20.4837761, 17.4104309, -44.3542595, 43.0129967
4: -19.8655491, 25.3290195, -15.1791763, 19.6839581, -39.5495071, 40.5081940

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1554235, upper bound: 42.1232375
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1554235, upper bound: 42.1233116
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.4342051, 6.2850499, -7.2240224, 6.0141058, -13.4483109, 13.5090723
1: -28.3053684, 23.0678425, -27.4968166, 22.0761547, -50.3815155, 50.5646591
2: -15.2026157, 23.7269230, -14.7198486, 22.6710320, -37.8736382, 38.4467697
3: -25.6915169, 21.5610638, -25.0566368, 20.5917702, -46.2832870, 46.6176987
4: -18.9513893, 24.2542629, -18.5228233, 23.0940151, -42.0454025, 42.7770844

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1554235, upper bound: 42.1232375
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1554235, upper bound: 42.1233116
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.1372395, 6.8510528, -5.9477453, 5.0575151, -13.1947536, 12.7987976
1: -30.9891644, 25.1049747, -22.5129051, 18.6221199, -49.6112823, 47.6178741
2: -16.6546764, 25.8586311, -12.3415947, 19.2346973, -35.8893661, 38.2002144
3: -28.1455193, 23.4581947, -20.4837761, 17.4104309, -45.5559502, 43.9419708
4: -20.7656860, 26.3863869, -15.1791763, 19.6839581, -40.4496460, 41.5655632

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1621427, upper bound: 42.1232231
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1621427, upper bound: 42.1232973
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.7894573, 6.5710630, -7.2240224, 6.0141058, -13.8035631, 13.7950859
1: -29.6437645, 24.0924187, -27.4968166, 22.0761547, -51.7199173, 51.5892334
2: -15.9762011, 24.8198662, -14.7198486, 22.6710320, -38.6472321, 39.5397148
3: -26.9238529, 22.5138378, -25.0566368, 20.5917702, -47.5156250, 47.5704727
4: -19.8744984, 25.3381100, -18.5228233, 23.0940151, -42.9685135, 43.8609314

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1637397, upper bound: 42.1231934
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1666302, upper bound: 42.1232729
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.3067865, 6.1290946, -7.0730286, 5.9704599, -13.2772465, 13.2021217
1: -27.7741547, 22.4439907, -26.9138279, 21.9011765, -49.6753273, 49.3578186
2: -14.9191265, 23.2865257, -14.5541716, 22.5381660, -37.4572906, 37.8406868
3: -25.2660370, 21.0110912, -24.5055180, 20.4324741, -45.6985054, 45.5166092
4: -18.6393318, 23.7244930, -18.1029053, 23.0376472, -41.6769714, 41.8273888

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0889435, upper bound: 42.1385958
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1028062, upper bound: 42.1386510
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3067865, 6.1290946, -7.3855829, 6.2298532, -13.5366402, 13.5146761
1: -27.7741547, 22.4439907, -28.1211166, 22.8619728, -50.6361275, 50.5651093
2: -14.9191265, 23.2865257, -15.1779146, 23.4398193, -38.3589478, 38.4644241
3: -25.2660370, 21.0110912, -25.6057873, 21.3124771, -46.5785141, 46.6168785
4: -18.6393318, 23.7244930, -18.9243641, 23.9788113, -42.6181412, 42.6488571

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1055379, upper bound: 42.1694762
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0893040, upper bound: 42.1683867
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1055559, upper bound: 42.1695348
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1055559, upper bound: 42.1695349
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.8452606, 5.8309255, -6.5909829, 5.5873561, -12.4326172, 12.4219055
1: -25.9590454, 21.4107075, -24.9450760, 20.4672585, -46.4263039, 46.3557816
2: -14.1704302, 22.1001835, -13.6418095, 21.2233734, -35.3937988, 35.7419853
3: -23.6052780, 20.0540352, -22.7296696, 19.1634750, -42.7687531, 42.7837067
4: -17.4646206, 22.6468563, -16.8233032, 21.7049599, -39.1695786, 39.4701614

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0473515, upper bound: 42.1492722
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0621269, upper bound: 42.1522022
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.1073685, 5.1969953, -8.0292177, 6.7537384, -12.8611069, 13.2262135
1: -23.0096493, 19.0621185, -30.5981636, 24.7288456, -47.7384949, 49.6602821
2: -12.6603308, 19.8433228, -16.4103374, 25.4915905, -38.1519203, 36.2536621
3: -20.9727955, 17.9096909, -27.8108501, 23.1059799, -44.0787735, 45.7205391
4: -15.5533142, 20.3062344, -20.4988327, 26.0128632, -41.5661736, 40.8050690

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -41.9960482, upper bound: 42.1156664
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0230807, upper bound: 42.1361134
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.8836098, 5.8665242, -8.0292177, 6.7537384, -13.6373472, 13.8957424
1: -26.1062317, 21.5468769, -30.5981636, 24.7288456, -50.8350754, 52.1450424
2: -14.2515154, 22.2294064, -16.4103374, 25.4915905, -39.7431068, 38.6397438
3: -23.7357693, 20.1815529, -27.8108501, 23.1059799, -46.8417511, 47.9923935
4: -17.5615540, 22.7838631, -20.4988327, 26.0128632, -43.5744133, 43.2826920

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1322629, upper bound: 42.1745240
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0654096, upper bound: 42.1325475
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.0292177, 6.7537384, -6.8836098, 5.8665242, -13.8957424, 13.6373472
1: -30.5981636, 24.7288456, -26.1062317, 21.5468769, -52.1450386, 50.8350754
2: -16.4103374, 25.4915905, -14.2515154, 22.2294064, -38.6397438, 39.7431068
3: -27.8108501, 23.1059799, -23.7357693, 20.1815529, -47.9923935, 46.8417511
4: -20.4988327, 26.0128632, -17.5615540, 22.7838631, -43.2826958, 43.5744095

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1275024, upper bound: 42.0659807
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0615170, upper bound: 42.0653010
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.3067865, 6.1290946, -7.3201694, 6.1741719, -13.4809589, 13.4492598
1: -27.7741547, 22.4439907, -27.8716335, 22.6488609, -50.4230156, 50.3156242
2: -14.9191265, 23.2865257, -15.0137396, 23.3993320, -38.3184586, 38.3002472
3: -25.2660370, 21.0110912, -25.2970600, 21.2022343, -46.4682693, 46.3081474
4: -18.6393318, 23.7244930, -18.6500950, 23.8920059, -42.5313377, 42.3745880

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0465783, upper bound: 42.1441683
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0465783, upper bound: 42.1448053
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3058600, 6.1283779, -9.4568195, 7.8734126, -15.1792717, 15.5851974
1: -27.7705765, 22.4414024, -36.2854042, 28.9125271, -56.6831055, 58.7268066
2: -14.9173956, 23.2839222, -18.9898376, 29.2519073, -44.1692963, 42.2737503
3: -25.2627945, 21.0087032, -32.9487076, 27.0702019, -52.3329964, 53.9574127
4: -18.6369171, 23.7218876, -24.1913795, 29.9412003, -48.5781174, 47.9132614

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0465783, upper bound: 42.1512675
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0465783, upper bound: 42.1621118
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.0292177, 6.7537384, -7.4035354, 6.2400146, -14.2692318, 14.1572742
1: -30.5981636, 24.7288456, -28.2022667, 22.8853912, -53.4835548, 52.9311142
2: -16.4103374, 25.4915905, -15.1697969, 23.6350708, -40.0454025, 40.6613884
3: -27.8108501, 23.1059799, -25.5976562, 21.4185829, -49.2294197, 48.7036362
4: -20.4988327, 26.0128632, -18.8654232, 24.1289005, -44.6277313, 44.8782768

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1677375
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1677375
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.0283060, 6.7530332, -9.5443735, 7.9453588, -15.9736652, 16.2974072
1: -30.5946541, 24.7263069, -36.6316490, 29.1718826, -59.7665367, 61.3579445
2: -16.4086361, 25.4890633, -19.1646919, 29.5116196, -45.9202576, 44.6537514
3: -27.8076496, 23.1036415, -33.2612267, 27.3086987, -55.1163483, 56.3648682
4: -20.4964542, 26.0103264, -24.4158955, 30.2059040, -50.7023506, 50.4262123

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1723508
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1723665
time: 0.96 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.06 seconds
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1121967, upper bound: 42.1434467
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1121967, upper bound: 42.1434467
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1121967, upper bound: 42.1434467
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1258456, upper bound: 42.1436093
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1249185, upper bound: 42.1435766
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1249185, upper bound: 42.1435997
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1403084, upper bound: 42.1256824
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1686703, upper bound: 42.1260548
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1403084, upper bound: 42.1256824
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1686703, upper bound: 42.1260548
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1316452, upper bound: 42.1247174
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1316452, upper bound: 42.1247171
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1316683, upper bound: 42.1247174
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1316683, upper bound: 42.1249185
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1587246, upper bound: 42.1439074
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1736363, upper bound: 42.1441815
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1587247, upper bound: 42.1439075
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1736363, upper bound: 42.1441815
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1317055, upper bound: 42.1317470
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1317055, upper bound: 42.1319481
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1317448, upper bound: 42.1429100
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1317448, upper bound: 42.1438796
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1113000, upper bound: 42.1631845
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1241845, upper bound: 42.1628095
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1113001, upper bound: 42.1684255
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1241849, upper bound: 42.1682163
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1625460, upper bound: 42.0893071
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1625460, upper bound: 42.0893071
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1632216, upper bound: 42.1031773
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1632216, upper bound: 42.1031773
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1521621, upper bound: 42.1497004
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1734895, upper bound: 42.1695501
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1573787, upper bound: 42.1588058
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1734707, upper bound: 42.1736979
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1577913, upper bound: 42.1681279
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1577913, upper bound: 42.1681300
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1577913, upper bound: 42.1717864
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1577913, upper bound: 42.1717885
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1554235, upper bound: 42.1232375
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1554235, upper bound: 42.1233116
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1554235, upper bound: 42.1232375
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1554235, upper bound: 42.1233116
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1621427, upper bound: 42.1232231
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1621427, upper bound: 42.1232973
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1637397, upper bound: 42.1231934
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1666302, upper bound: 42.1232729
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0889435, upper bound: 42.1385958
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1028062, upper bound: 42.1386510
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1055559, upper bound: 42.1695348
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1055559, upper bound: 42.1695349
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0473515, upper bound: 42.1492722
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0621269, upper bound: 42.1522022
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -41.9960482, upper bound: 42.1156664
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0230807, upper bound: 42.1361134
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1322629, upper bound: 42.1745240
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0654096, upper bound: 42.1325475
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1275024, upper bound: 42.0659807
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0615170, upper bound: 42.0653010
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0465783, upper bound: 42.1441683
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0465783, upper bound: 42.1448053
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0465783, upper bound: 42.1512675
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.0465783, upper bound: 42.1621118
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1677375
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1677375
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1723508
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 4, lower bound: -42.1677375, upper bound: 42.1723665

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.2106028, 5.2379313, -7.2030978, 6.0868549, -12.2974558, 12.4410286
1: -23.5227795, 19.2090187, -27.4097786, 22.3412838, -45.8640518, 46.6187973
2: -12.8160295, 19.9602623, -14.8383522, 22.9614334, -35.7774544, 34.7986069
3: -21.4436264, 17.9655342, -24.9513969, 20.8417320, -42.2853584, 42.9169312
4: -15.8686504, 20.3688126, -18.4353981, 23.4920616, -39.3607025, 38.8042107

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1120335, upper bound: 42.1401459
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1120335, upper bound: 42.1695447
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.5627375, 5.5260830, -7.2030978, 6.0868549, -12.6495924, 12.7291813
1: -24.8789349, 20.2889233, -27.4097786, 22.3412838, -47.2202110, 47.6987000
2: -13.5151358, 20.9776917, -14.8383522, 22.9614334, -36.4765625, 35.8160362
3: -22.6771431, 18.9547539, -24.9513969, 20.8417320, -43.5188751, 43.9061508
4: -16.7903728, 21.4330730, -18.4353981, 23.4920616, -40.2824249, 39.8684692

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1256824, upper bound: 42.1403084
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1256824, upper bound: 42.1695447
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.2106028, 5.2379313, -7.4157219, 6.1576800, -12.3682823, 12.6536531
1: -23.5227795, 19.2090187, -28.3268032, 22.5180664, -46.0408440, 47.5358162
2: -12.8160295, 19.9602623, -14.9386253, 23.2305889, -36.0466156, 34.8988800
3: -21.4436264, 17.9655342, -25.7925682, 20.9962559, -42.4398766, 43.7581024
4: -15.8686504, 20.3688126, -18.9443398, 23.6593857, -39.5280266, 39.3131523

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0372358, upper bound: 42.1224282
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.5627375, 5.5260830, -7.4157219, 6.1576800, -12.7204170, 12.9418049
1: -24.8789349, 20.2889233, -28.3268032, 22.5180664, -47.3969955, 48.6157112
2: -13.5151358, 20.9776917, -14.9386253, 23.2305889, -36.7457237, 35.9163055
3: -22.6771431, 18.9547539, -25.7925682, 20.9962559, -43.6733894, 44.7473183
4: -16.7903728, 21.4330730, -18.9443398, 23.6593857, -40.4497490, 40.3774109

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0971990, upper bound: 42.1431432
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0969689, upper bound: 42.1280263
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.9460249, 4.9541650, -8.0529547, 6.7291908, -12.6752148, 13.0071201
1: -22.5798683, 18.1879120, -30.6990776, 24.6742020, -47.2540627, 48.8869896
2: -12.0998077, 18.8939648, -16.4115734, 25.2890835, -37.3888931, 35.3055305
3: -20.5487156, 16.9958210, -27.9423332, 22.9853611, -43.5340767, 44.9381561
4: -15.1276922, 19.2179623, -20.6441689, 25.7811012, -40.9087906, 39.8621292

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1247174, upper bound: 42.1435766
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1247174, upper bound: 42.1435766
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.0321383, 5.8174853, -8.0529547, 6.7291908, -13.7613297, 13.8704395
1: -26.7627754, 21.2591610, -30.6990776, 24.6742020, -51.4369621, 51.9582367
2: -14.3160305, 22.0805664, -16.4115734, 25.2890835, -39.6051025, 38.4921379
3: -24.4016151, 19.8794899, -27.9423332, 22.9853611, -47.3869743, 47.8218231
4: -17.9825306, 22.3854542, -20.6441689, 25.7811012, -43.7636337, 43.0296249

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1247174, upper bound: 42.1393212
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1247174, upper bound: 42.1393213
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.9252691, 5.8492455, -6.4883018, 5.4759121, -12.4011812, 12.3375473
1: -26.3390808, 21.4586048, -24.5961266, 20.1034985, -46.4425812, 46.0547180
2: -14.2692490, 22.1088886, -13.3884697, 20.8137512, -35.0830002, 35.4973564
3: -23.9838524, 20.0242386, -22.4122162, 18.7938576, -42.7777100, 42.4364471
4: -17.7192307, 22.5997753, -16.5840015, 21.2673397, -38.9865723, 39.1837769

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1401459, upper bound: 42.1120335
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1401459, upper bound: 42.1256824
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.2396202, 6.1100483, -6.4883018, 5.4759121, -12.7155323, 12.5983505
1: -27.5517712, 22.4243202, -24.5961266, 20.1034985, -47.6552620, 47.0204315
2: -14.8967161, 23.0171394, -13.3884697, 20.8137512, -35.7104645, 36.4056053
3: -25.0887413, 20.9088268, -22.4122162, 18.7938576, -43.8825989, 43.3210373
4: -18.5446339, 23.5476208, -16.5840015, 21.2673397, -39.8119698, 40.1316223

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1695447, upper bound: 42.1124202
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1695447, upper bound: 42.1260548
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.9252691, 5.8492455, -6.6365323, 5.5056200, -12.4308891, 12.4857779
1: -26.3390808, 21.4586048, -25.2478600, 20.1276054, -46.4666862, 46.7064590
2: -14.2692490, 22.1088886, -13.4587021, 20.9614220, -35.2306709, 35.5675888
3: -23.9838524, 20.0242386, -23.0048962, 18.8198586, -42.8037109, 43.0291367
4: -17.7192307, 22.5997753, -16.9144535, 21.2759056, -38.9951363, 39.5142288

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.2396202, 6.1100483, -6.6365323, 5.5056200, -12.7452402, 12.7465801
1: -27.5517712, 22.4243202, -25.2478600, 20.1276054, -47.6793747, 47.6721764
2: -14.8967161, 23.0171394, -13.4587021, 20.9614220, -35.8581390, 36.4758415
3: -25.0887413, 20.9088268, -23.0048962, 18.8198586, -43.9085999, 43.9137230
4: -18.5446339, 23.5476208, -16.9144535, 21.2759056, -39.8205338, 40.4620705

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1665114, upper bound: 42.1198208
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.4837389, 5.5221415, -7.2030978, 6.0868549, -12.5705938, 12.7252388
1: -24.6376781, 20.3366394, -27.4097786, 22.3412838, -46.9789581, 47.7464142
2: -13.4542866, 20.8644276, -14.8383522, 22.9614334, -36.4157181, 35.7027702
3: -22.3943233, 18.9702606, -24.9513969, 20.8417320, -43.2360535, 43.9216576
4: -16.5676537, 21.3854122, -18.4353981, 23.4920616, -40.0597076, 39.8208046

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1587436, upper bound: 42.1587525
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1587436, upper bound: 42.1635101
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.9566426, 6.6496143, -7.0165229, 5.9301338, -13.8867760, 13.6661377
1: -30.3258419, 24.3846645, -26.6872406, 21.7670574, -52.0928917, 51.0719070
2: -16.2261829, 25.0084019, -14.4579067, 22.3721981, -38.5983772, 39.4663048
3: -27.6025467, 22.7183170, -24.3006458, 20.3028984, -47.9054451, 47.0189629
4: -20.3935318, 25.4939117, -17.9586849, 22.8903046, -43.2838326, 43.4525833

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1735139, upper bound: 42.1694029
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1734984, upper bound: 42.1735018
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.4837389, 5.5221415, -7.4157219, 6.1576800, -12.6414185, 12.9378633
1: -24.6376781, 20.3366394, -28.3268032, 22.5180664, -47.1557465, 48.6634293
2: -13.4542866, 20.8644276, -14.9386253, 23.2305889, -36.6848755, 35.8030434
3: -22.3943233, 18.9702606, -25.7925682, 20.9962559, -43.3905716, 44.7628250
4: -16.5676537, 21.3854122, -18.9443398, 23.6593857, -40.2270317, 40.3297501

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1585236, upper bound: 42.1319760
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1585236, upper bound: 42.1439075
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.9566426, 6.6496143, -7.1776657, 5.9657469, -13.9223900, 13.8272800
1: -30.3258419, 24.3846645, -27.4012547, 21.8153973, -52.1412315, 51.7859192
2: -16.2261829, 25.0084019, -14.4790144, 22.5095501, -38.7357330, 39.4874039
3: -27.6025467, 22.7183170, -24.9588032, 20.3407631, -47.9433022, 47.6771202
4: -20.3935318, 25.4939117, -18.3404579, 22.9333878, -43.3269196, 43.8343658

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1727034, upper bound: 42.1322101
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1727034, upper bound: 42.1441815
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.6782227, 5.5773668, -8.0529547, 6.7291908, -13.4074135, 13.6303215
1: -25.4791546, 20.4650421, -30.6990776, 24.6742020, -50.1533508, 51.1641159
2: -13.5373583, 21.0587997, -16.4115734, 25.2890835, -38.8264389, 37.4703712
3: -23.1762333, 19.0780811, -27.9423332, 22.9853611, -46.1615868, 47.0204163
4: -17.0387611, 21.4968071, -20.6441689, 25.7811012, -42.8198586, 42.1409760

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1317055, upper bound: 42.1429100
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1317055, upper bound: 42.1429100
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.9360323, 6.5718722, -8.0529547, 6.7291908, -14.6652231, 14.6248264
1: -30.2841492, 24.0246029, -30.6990776, 24.6742020, -54.9583511, 54.7236710
2: -16.0474625, 24.7655296, -16.4115734, 25.2890835, -41.3365440, 41.1770935
3: -27.5764961, 22.3966236, -27.9423332, 22.9853611, -50.5618477, 50.3389587
4: -20.3110561, 25.1365280, -20.6441689, 25.7811012, -46.0921516, 45.7806969

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1317055, upper bound: 42.1438796
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1317055, upper bound: 42.1438796
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.2007623, 5.2489672, -7.9733176, 6.7187214, -12.9194822, 13.2222834
1: -23.4704590, 19.2740822, -30.3791008, 24.6392021, -48.1096611, 49.6531754
2: -12.8386374, 19.9988804, -16.2489929, 25.3012714, -38.1399078, 36.2478714
3: -21.3751621, 18.0285072, -27.5864220, 23.0095901, -44.3847504, 45.6149292
4: -15.8355827, 20.4178715, -20.3548717, 25.8419018, -41.6774788, 40.7727394

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1112003, upper bound: 42.1520408
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1112003, upper bound: 42.1628094
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.5536051, 5.5378981, -7.9733176, 6.7187214, -13.2723255, 13.5112152
1: -24.8307343, 20.3561420, -30.3791008, 24.6392021, -49.4699364, 50.7352333
2: -13.5419569, 21.0191994, -16.2489929, 25.3012714, -38.8432274, 37.2681923
3: -22.6109772, 19.0189724, -27.5864220, 23.0095901, -45.6205673, 46.6053925
4: -16.7582664, 21.4810982, -20.3548717, 25.8419018, -42.6001663, 41.8359680

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1240840, upper bound: 42.1520414
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1240840, upper bound: 42.1628096
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.2007623, 5.2489672, -8.3133841, 6.9936090, -13.1943712, 13.5623503
1: -23.4704590, 19.2740822, -31.6607399, 25.6225071, -49.0929642, 50.9348221
2: -12.8386374, 19.9988804, -16.9971638, 26.3419380, -39.1805763, 36.9960365
3: -21.3751621, 18.0285072, -28.7680817, 23.9240208, -45.2991829, 46.7965889
4: -15.8355827, 20.4178715, -21.2419167, 26.8776150, -42.7131958, 41.6597824

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1112443, upper bound: 42.1659142
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1112443, upper bound: 42.1682162
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.5536051, 5.5378981, -8.3133841, 6.9936090, -13.5472145, 13.8512821
1: -24.8307343, 20.3561420, -31.6607399, 25.6225071, -50.4532394, 52.0168800
2: -13.5419569, 21.0191994, -16.9971638, 26.3419380, -39.8838959, 38.0163574
3: -22.6109772, 19.0189724, -28.7680817, 23.9240208, -46.5349960, 47.7870560
4: -16.7582664, 21.4810982, -21.2419167, 26.8776150, -43.6358795, 42.7230148

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1241328, upper bound: 42.1659142
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1241328, upper bound: 42.1682162
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.1112185, 6.0194793, -7.2342577, 6.0809770, -13.1921940, 13.2537365
1: -27.0774384, 22.1315079, -27.4884567, 22.3091373, -49.3865738, 49.6199646
2: -14.5902710, 22.6110744, -14.7548876, 23.0420036, -37.6322746, 37.3659515
3: -24.6194763, 20.6295319, -24.9738770, 20.8753986, -45.4948730, 45.6034050
4: -18.2001686, 23.1697617, -18.4537086, 23.5064278, -41.7065964, 41.6234703

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1562105, upper bound: 42.0726123
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1562105, upper bound: 42.0893071
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.2587996, 6.1461306, -7.2342577, 6.0809770, -13.3397770, 13.3803864
1: -27.6138000, 22.5771637, -27.4884567, 22.3091373, -49.9229355, 50.0656204
2: -14.9757395, 23.1457253, -14.7548876, 23.0420036, -38.0177383, 37.9006081
3: -25.1222191, 21.0497055, -24.9738770, 20.8753986, -45.9976196, 46.0235786
4: -18.5857792, 23.6831627, -18.4537086, 23.5064278, -42.0922089, 42.1368713

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1562105, upper bound: 42.0726123
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1562105, upper bound: 42.0893071
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.1112185, 6.0194793, -7.5653849, 6.3510666, -13.4622841, 13.5848637
1: -27.0774384, 22.1315079, -28.7326012, 23.2720337, -50.3494720, 50.8641090
2: -14.5902710, 22.6110744, -15.4641953, 24.0739460, -38.6642151, 38.0752602
3: -24.6194763, 20.6295319, -26.1274834, 21.7682762, -46.3877449, 46.7570076
4: -18.2001686, 23.1697617, -19.3142757, 24.5278664, -42.7280350, 42.4840317

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1562105, upper bound: 42.1031772
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1562105, upper bound: 42.1031772
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.2587996, 6.1461306, -7.5653849, 6.3510666, -13.6098661, 13.7115154
1: -27.6138000, 22.5771637, -28.7326012, 23.2720337, -50.8858337, 51.3097649
2: -14.9757395, 23.1457253, -15.4641953, 24.0739460, -39.0496826, 38.6099167
3: -25.1222191, 21.0497055, -26.1274834, 21.7682762, -46.8904953, 47.1771851
4: -18.5857792, 23.6831627, -19.3142757, 24.5278664, -43.1136475, 42.9974365

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1562105, upper bound: 42.1031378
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1562105, upper bound: 42.1031378
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.6966934, 5.6871929, -7.9733176, 6.7187214, -13.4154148, 13.6605072
1: -25.4673100, 20.9392891, -30.3791008, 24.6392021, -50.1065140, 51.3183861
2: -13.8498993, 21.4245853, -16.2489929, 25.3012714, -39.1511688, 37.6735764
3: -23.1564350, 19.5174408, -27.5864220, 23.0095901, -46.1660233, 47.1038628
4: -17.1341248, 21.9544525, -20.3548717, 25.8419018, -42.9760132, 42.3093262

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1521645, upper bound: 42.1496978
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1521645, upper bound: 42.1497008
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.0154285, 6.6853204, -7.6322041, 6.4438367, -14.4592648, 14.3175240
1: -30.6237144, 24.5273876, -29.0600376, 23.6450005, -54.2687149, 53.5874214
2: -16.3013573, 24.9961567, -15.5824680, 24.2792168, -40.5805740, 40.5786247
3: -27.8959141, 22.8106441, -26.3880196, 22.0815125, -49.9774246, 49.1986618
4: -20.5990524, 25.4991074, -19.4814396, 24.8097057, -45.4087563, 44.9805450

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1717554, upper bound: 42.1664932
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1718955, upper bound: 42.1686873
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.6966934, 5.6871929, -8.3133841, 6.9936090, -13.6903019, 14.0005760
1: -25.4673100, 20.9392891, -31.6607399, 25.6225071, -51.0898170, 52.6000290
2: -13.8498993, 21.4245853, -16.9971638, 26.3419380, -40.1918373, 38.4217377
3: -23.1564350, 19.5174408, -28.7680817, 23.9240208, -47.0804558, 48.2855225
4: -17.1341248, 21.9544525, -21.2419167, 26.8776150, -44.0117378, 43.1963654

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1573787, upper bound: 42.1581900
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1573787, upper bound: 42.1588058
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.0154285, 6.6853204, -7.9737144, 6.7199702, -14.7353992, 14.6590347
1: -30.6237144, 24.5273876, -30.3461514, 24.6332550, -55.2569695, 54.8735390
2: -16.3013573, 24.9961567, -16.3335152, 25.3271294, -41.6284866, 41.3296738
3: -27.8959141, 22.8106441, -27.5730724, 23.0010910, -50.8970032, 50.3837166
4: -20.5990524, 25.4991074, -20.3716583, 25.8524323, -46.4514771, 45.8707657

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1712167, upper bound: 42.1666873
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1718764, upper bound: 42.1720508
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.2897935, 5.3589706, -6.9216790, 5.8675747, -12.1573658, 12.2806492
1: -23.8088932, 19.6656475, -26.3487301, 21.5746021, -45.3834877, 46.0143776
2: -13.0074282, 20.3296776, -14.2259674, 22.0851326, -35.0925598, 34.5556450
3: -21.6639709, 18.4175892, -23.9505138, 20.1222572, -41.7862129, 42.3681030
4: -16.0387383, 20.8317394, -17.6975842, 22.6317577, -38.6704941, 38.5293236

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1565230, upper bound: 42.1626870
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1565302, upper bound: 42.1654961
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.6328993, 5.6249795, -6.9216790, 5.8675747, -12.5004740, 12.5466585
1: -25.0435638, 20.6230869, -26.3487301, 21.5746021, -46.6181641, 46.9718132
2: -13.7383623, 21.3461494, -14.2259674, 22.0851326, -35.8234940, 35.5721130
3: -22.8101349, 19.3019238, -23.9505138, 20.1222572, -42.9323883, 43.2524376
4: -16.9062328, 21.8327427, -17.6975842, 22.6317577, -39.5379906, 39.5303268

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1565230, upper bound: 42.1626891
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1565302, upper bound: 42.1654983
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.2897935, 5.3589706, -7.1050310, 6.0235333, -12.3133249, 12.4640017
1: -23.8088932, 19.6656475, -27.0171547, 22.1244316, -45.9333229, 46.6828003
2: -13.0074282, 20.3296776, -14.6778831, 22.7411022, -35.7485313, 35.0075569
3: -21.6639709, 18.4175892, -24.5708675, 20.6426811, -42.3066521, 42.9884567
4: -16.0387383, 20.8317394, -18.1691284, 23.2647362, -39.3034744, 39.0008659

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1564900, upper bound: 42.1667734
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.6328993, 5.6249795, -7.1050310, 6.0235333, -12.6564331, 12.7300100
1: -25.0435638, 20.6230869, -27.0171547, 22.1244316, -47.1679955, 47.6402359
2: -13.7383623, 21.3461494, -14.6778831, 22.7411022, -36.4794617, 36.0240211
3: -22.8101349, 19.3019238, -24.5708675, 20.6426811, -43.4528160, 43.8727875
4: -16.9062328, 21.8327427, -18.1691284, 23.2647362, -40.1709671, 40.0018692

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1564900, upper bound: 42.1633115
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1565078, upper bound: 42.1664034
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.1453376, 6.0601797, -5.9477453, 5.0575151, -12.2028522, 12.0079250
1: -27.1921711, 22.2897320, -22.5129051, 18.6221199, -45.8142891, 44.8026314
2: -14.6487837, 22.8665867, -12.3415947, 19.2346973, -33.8834801, 35.2081757
3: -24.6488914, 20.8503342, -20.4837761, 17.4104309, -42.0593224, 41.3341103
4: -18.1937027, 23.3949337, -15.1791763, 19.6839581, -37.8776627, 38.5741119

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1571538, upper bound: 42.1111767
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1569190, upper bound: 42.1240722
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.0751810, 6.7690587, -5.9477453, 5.0575151, -13.1326952, 12.7168007
1: -30.7853260, 24.8724556, -22.5129051, 18.6221199, -49.4074478, 47.3853569
2: -16.3763580, 25.3717346, -12.3415947, 19.2346973, -35.6110535, 37.7133255
3: -27.9724960, 23.2528610, -20.4837761, 17.4104309, -45.3829231, 43.7366257
4: -20.6466351, 25.8741665, -15.1791763, 19.6839581, -40.3305931, 41.0533409

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1571538, upper bound: 42.1112761
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1569190, upper bound: 42.1241517
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.1453376, 6.0601797, -7.2240224, 6.0141058, -13.1594429, 13.2842026
1: -27.1921711, 22.2897320, -27.4968166, 22.0761547, -49.2683258, 49.7865448
2: -14.6487837, 22.8665867, -14.7198486, 22.6710320, -37.3198090, 37.5864334
3: -24.6488914, 20.8503342, -25.0566368, 20.5917702, -45.2406616, 45.9069710
4: -18.1937027, 23.3949337, -18.5228233, 23.0940151, -41.2877159, 41.9177551

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1529418, upper bound: 42.1169578
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.0750904, 6.7689824, -7.2240224, 6.0141058, -14.0891962, 13.9930029
1: -30.7849941, 24.8721695, -27.4968166, 22.0761547, -52.8611412, 52.3689880
2: -16.3761330, 25.3714638, -14.7198486, 22.6710320, -39.0471535, 40.0913086
3: -27.9721870, 23.2525883, -25.0566368, 20.5917702, -48.5639572, 48.3092270
4: -20.6464005, 25.8738213, -18.5228233, 23.0940151, -43.7404175, 44.3966446

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1529418, upper bound: 42.1170044
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.4772868, 6.3270350, -5.9477453, 5.0575151, -12.5348015, 12.2747803
1: -28.4407120, 23.2446709, -22.5129051, 18.6221199, -47.0628319, 45.7575684
2: -15.3762913, 23.9009266, -12.3415947, 19.2346973, -34.6109810, 36.2425117
3: -25.8018951, 21.7394295, -20.4837761, 17.4104309, -43.2123260, 42.2232018
4: -19.0567093, 24.4139843, -15.1791763, 19.6839581, -38.7406693, 39.5931587

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1612574, upper bound: 42.1111500
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1612333, upper bound: 42.1240480
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.3343267, 6.9779820, -5.9477453, 5.0575151, -13.3918409, 12.9257278
1: -31.7650261, 25.6134739, -22.5129051, 18.6221199, -50.3871422, 48.1263809
2: -16.9676437, 26.1935253, -12.3415947, 19.2346973, -36.2023392, 38.5351181
3: -28.8814087, 23.9410992, -20.4837761, 17.4104309, -46.2918396, 44.4248734
4: -21.3183060, 26.6702442, -15.1791763, 19.6839581, -41.0022621, 41.8494186

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1612574, upper bound: 42.1112544
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1569190, upper bound: 42.1241329
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.5450058, 6.3570495, -7.2240224, 6.0141058, -13.5591106, 13.5810719
1: -28.7052097, 23.2960320, -27.4968166, 22.0761547, -50.7813644, 50.7928467
2: -15.4655991, 24.0539455, -14.7198486, 22.6710320, -38.1366234, 38.7737961
3: -26.0761700, 21.7754650, -25.0566368, 20.5917702, -46.6679382, 46.8320999
4: -19.2466011, 24.5302830, -18.5228233, 23.0940151, -42.3406143, 43.0531082

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1609358, upper bound: 42.1169191
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.6731606, 6.4737806, -7.2240224, 6.0141058, -13.6872663, 13.6978016
1: -29.1907215, 23.7511711, -27.4968166, 22.0761547, -51.2668762, 51.2479782
2: -15.7569389, 24.4393921, -14.7198486, 22.6710320, -38.4279594, 39.1592407
3: -26.5130310, 22.1860085, -25.0566368, 20.5917702, -47.1048012, 47.2426453
4: -19.5902481, 24.9587936, -18.5228233, 23.0940151, -42.6842651, 43.4816170

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1638960, upper bound: 42.1169844
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.0244884, 5.9108953, -7.0730286, 5.9704599, -12.9949455, 12.9839239
1: -26.6950073, 21.6887264, -26.9138279, 21.9011765, -48.5961761, 48.6025543
2: -14.3295946, 22.4424667, -14.5541716, 22.5381660, -36.8677597, 36.9966354
3: -24.2483215, 20.3099461, -24.5055180, 20.4324741, -44.6807938, 44.8154640
4: -17.8963337, 22.8979092, -18.1029053, 23.0376472, -40.9339714, 41.0008087

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3605728, 6.1836419, -7.0730286, 5.9704599, -13.3310289, 13.2566700
1: -27.9530258, 22.6619167, -26.9138279, 21.9011765, -49.8541946, 49.5757446
2: -15.0581074, 23.4829617, -14.5541716, 22.5381660, -37.5962753, 38.0371323
3: -25.4180851, 21.2104111, -24.5055180, 20.4324741, -45.8505592, 45.7159271
4: -18.7712193, 23.9285259, -18.1029053, 23.0376472, -41.8088570, 42.0314255

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.0732355, 5.9205236, -7.3855829, 6.2298532, -13.3030891, 13.3061066
1: -26.8804932, 21.6700325, -28.1211166, 22.8619728, -49.7424622, 49.7911491
2: -14.4238520, 22.5470924, -15.1779146, 23.4398193, -37.8636703, 37.7249985
3: -24.4538231, 20.2966766, -25.6057873, 21.3124771, -45.7663002, 45.9024544
4: -18.0388699, 22.9404297, -18.9243641, 23.9788113, -42.0176811, 41.8647919

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.2365646, 6.0652299, -7.3855829, 6.2298532, -13.4664164, 13.4508123
1: -27.4992027, 22.2268772, -28.1211166, 22.8619728, -50.3611755, 50.3479919
2: -14.7735577, 23.0190907, -15.1779146, 23.4398193, -38.2133751, 38.1970062
3: -25.0172806, 20.7940769, -25.6057873, 21.3124771, -46.3297577, 46.3998642
4: -18.4763470, 23.4594421, -18.9243641, 23.9788113, -42.4551582, 42.3838043

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.5643559, 5.6167583, -6.5909829, 5.5873561, -12.1517105, 12.2077408
1: -24.8861656, 20.6645107, -24.9450760, 20.4672585, -45.3534241, 45.6095886
2: -13.5742502, 21.2625046, -13.6418095, 21.2233734, -34.7976227, 34.9043045
3: -22.5973835, 19.3577690, -22.7296696, 19.1634750, -41.7608566, 42.0874405
4: -16.7249069, 21.8263645, -16.8233032, 21.7049599, -38.4298668, 38.6496658

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0473455, upper bound: 42.1475327
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0473455, upper bound: 42.1492722
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.8613687, 5.8554611, -6.5909829, 5.5873561, -12.4487247, 12.4464436
1: -25.9989223, 21.5145721, -24.9450760, 20.4672585, -46.4661789, 46.4596443
2: -14.2332478, 22.1746540, -13.6418095, 21.2233734, -35.4566154, 35.8164597
3: -23.6293678, 20.1458511, -22.7296696, 19.1634750, -42.7928429, 42.8755188
4: -17.4983139, 22.7249947, -16.8233032, 21.7049599, -39.2032738, 39.5482979

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0621233, upper bound: 42.1506873
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0621233, upper bound: 42.1522022
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.9989395, 5.1131968, -8.0292177, 6.7537384, -12.7526760, 13.1424141
1: -22.5830040, 18.7582970, -30.5981636, 24.7288456, -47.3118515, 49.3564606
2: -12.4536572, 19.5419579, -16.4103374, 25.4915905, -37.9452477, 35.9522934
3: -20.5826111, 17.6327438, -27.8108501, 23.1059799, -43.6885910, 45.4435883
4: -15.2726288, 20.0068398, -20.4988327, 26.0128632, -41.2854919, 40.5056686

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0235007, upper bound: 42.1361134
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.0235007, upper bound: 42.1361133
time: 0.66 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.37 + 416.71 = 420.08 seconds
