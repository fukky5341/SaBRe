## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2954094305


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5784860, 0.5784855)
1: (-16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8586702, 0.8586698)
2: (-4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6296978, 0.6296978)
3: (-12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6798048, 0.6798053)
4: (-10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5727429, 0.5727427)
5: (-7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6351314, 0.6351314)
6: (-5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9446011, 0.9446011)
7: (-11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8562999, 0.8562999)
8: (-2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5692954, 0.5692954)
9: (-2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6717248, 0.6717248)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.55 + 34.50 = 59.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.2968933, upper bound: 0.2968939

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 6163

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5826

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968932, upper bound: 0.2967286
time: 8.09 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2967285, upper bound: 0.2968931
time: 7.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.29 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.29
Output dim: 0, lower bound: -0.2968932, upper bound: 0.2967286
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.29
Output dim: 0, lower bound: -0.2967285, upper bound: 0.2968931

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5783739, 0.5782778
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8600659, 0.8595424
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6310425, 0.6306381
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6787291, 0.6785765
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5674424, 0.5665605
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6355343, 0.6357658
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9437275, 0.9436026
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8545642, 0.8547812
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5685453, 0.5686393
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6722920, 0.6725760

Time for backsubstitution: 22.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2964473, upper bound: 0.2966447
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968088, upper bound: 0.2962832
time: 4.00 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5782776, 0.5783741
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8595424, 0.8600659
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6306381, 0.6310425
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6785760, 0.6787291
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5665607, 0.5674424
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6357660, 0.6355338
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9436016, 0.9437265
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8547812, 0.8545642
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5686393, 0.5685453
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6725757, 0.6722918

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 6163

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 149

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2928545, upper bound: 0.2968915
time: 9.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2967268, upper bound: 0.2930198
time: 4.40 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 36.53 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.53
Output dim: 0, lower bound: -0.2964473, upper bound: 0.2966447
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.53
Output dim: 0, lower bound: -0.2968088, upper bound: 0.2962832
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.53
Output dim: 0, lower bound: -0.2928545, upper bound: 0.2968915
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.53
Output dim: 0, lower bound: -0.2967268, upper bound: 0.2930198

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5783739, 0.5782776
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8600669, 0.8595448
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6310420, 0.6306376
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6787243, 0.6785736
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5674424, 0.5665600
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6355329, 0.6357651
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9437275, 0.9436026
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8545632, 0.8547802
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5685449, 0.5686402
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6722910, 0.6725750

Time for backsubstitution: 23.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2935307, upper bound: 0.2966423
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2964455, upper bound: 0.2937275
time: 5.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5783739, 0.5782776
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8600678, 0.8595428
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6310420, 0.6306381
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6787267, 0.6785712
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5674419, 0.5665603
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6355329, 0.6357653
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9437275, 0.9436026
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8545632, 0.8547802
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5685463, 0.5686388
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6722910, 0.6725750

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 6163

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 149

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2929347, upper bound: 0.2962808
time: 6.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968070, upper bound: 0.2924084
time: 6.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5780272, 0.5782766
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8593879, 0.8600049
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6306009, 0.6310263
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6785755, 0.6787271
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5663633, 0.5669465
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6357594, 0.6355312
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9434376, 0.9436626
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8545895, 0.8540516
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5685334, 0.5685058
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6724696, 0.6720166

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2924083, upper bound: 0.2968072
time: 6.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2927700, upper bound: 0.2964455
time: 5.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5781808, 0.5781231
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8594813, 0.8599119
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6306219, 0.6310048
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6785741, 0.6787286
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5660648, 0.5672452
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6357632, 0.6355276
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9435377, 0.9435625
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8542686, 0.8543720
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5685992, 0.5684395
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6723003, 0.6721859

Time for backsubstitution: 23.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 6143

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2966085, upper bound: 0.2917688
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2954760, upper bound: 0.2929014
time: 6.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.30 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 0, lower bound: -0.2935307, upper bound: 0.2966423
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 0, lower bound: -0.2964455, upper bound: 0.2937275
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 0, lower bound: -0.2929347, upper bound: 0.2962808
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 0, lower bound: -0.2968070, upper bound: 0.2924084
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 0, lower bound: -0.2924083, upper bound: 0.2968072
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 0, lower bound: -0.2927700, upper bound: 0.2964455
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 0, lower bound: -0.2966085, upper bound: 0.2917688
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 0, lower bound: -0.2954760, upper bound: 0.2929014

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5633430, 0.5651922
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8616414, 0.8615642
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6181598, 0.6159177
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6655664, 0.6633959
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5595832, 0.5599966
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6016982, 0.6061652
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9206896, 0.9234419
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8494964, 0.8489928
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5657320, 0.5654268
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6687453, 0.6683660

Time for backsubstitution: 23.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 917

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 912

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2934345, upper bound: 0.2965490
time: 7.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2932728, upper bound: 0.2965489
time: 6.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5652885, 0.5632467
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8620858, 0.8611197
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6163225, 0.6177554
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6635466, 0.6654158
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5608792, 0.5587008
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6059334, 0.6019299
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9235668, 0.9205656
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8487759, 0.8497138
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5653319, 0.5658269
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6680825, 0.6690288

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 5831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2963272, upper bound: 0.2924774
time: 5.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2951948, upper bound: 0.2936092
time: 5.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5781231, 0.5781808
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8599133, 0.8594813
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6310043, 0.6306219
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6787262, 0.6785688
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5672445, 0.5660644
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6355271, 0.6357627
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9435625, 0.9435377
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8543715, 0.8542676
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5684409, 0.5685992
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6721849, 0.6722994

Time for backsubstitution: 23.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 5831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2928164, upper bound: 0.2950303
time: 6.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2916837, upper bound: 0.2961625
time: 5.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5782771, 0.5780268
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8600059, 0.8593884
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6310258, 0.6306005
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6787243, 0.6785707
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5669460, 0.5663631
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6355300, 0.6357591
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9436626, 0.9434376
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8540506, 0.8545885
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5685072, 0.5685329
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6720157, 0.6724687

Time for backsubstitution: 23.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 554

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2966888, upper bound: 0.2911579
time: 7.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2955564, upper bound: 0.2922897
time: 6.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5780272, 0.5782771
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8593879, 0.8600063
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6306005, 0.6310258
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6785707, 0.6787243
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5663633, 0.5669460
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6357589, 0.6355302
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9434376, 0.9436626
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8545885, 0.8540506
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5685334, 0.5685072
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6724687, 0.6720157

Time for backsubstitution: 23.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 6183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2910005, upper bound: 0.2968049
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2911643, upper bound: 0.2915298
time: 6.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5780272, 0.5782766
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8593898, 0.8600049
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6306000, 0.6310258
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6785731, 0.6787219
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5663629, 0.5669465
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6357589, 0.6355307
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9434376, 0.9436626
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8545885, 0.8540506
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5685344, 0.5685058
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6724687, 0.6720157

Time for backsubstitution: 22.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 5831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 528

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2921976, upper bound: 0.2962472
time: 7.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2927691, upper bound: 0.2923626
time: 7.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5814204, 0.5808830
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8505778, 0.8521242
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6257215, 0.6267176
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6756487, 0.6766725
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5652645, 0.5665340
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6354184, 0.6353366
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9426756, 0.9420042
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8566861, 0.8564320
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5663791, 0.5664973
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6706395, 0.6702881

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 6143

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2913306, upper bound: 0.2905249
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2966063, upper bound: 0.2903610
time: 3.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5809407, 0.5813627
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8516936, 0.8510094
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6263347, 0.6261044
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6765175, 0.6758032
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5653536, 0.5664454
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6355720, 0.6351826
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9419794, 0.9426994
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8563280, 0.8567901
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5666571, 0.5662189
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6704030, 0.6705246

Time for backsubstitution: 21.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 6163

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2950302, upper bound: 0.2916836
time: 7.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2953917, upper bound: 0.2924544
time: 4.80 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 33.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2934345, upper bound: 0.2965490
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2932728, upper bound: 0.2965489
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2963272, upper bound: 0.2924774
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2951948, upper bound: 0.2936092
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2928164, upper bound: 0.2950303
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2916837, upper bound: 0.2961625
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2966888, upper bound: 0.2911579
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2955564, upper bound: 0.2922897
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2910005, upper bound: 0.2968049
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2911643, upper bound: 0.2915298
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2921976, upper bound: 0.2962472
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2927691, upper bound: 0.2923626
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2913306, upper bound: 0.2905249
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2966063, upper bound: 0.2903610
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2950302, upper bound: 0.2916836
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.69
Output dim: 0, lower bound: -0.2953917, upper bound: 0.2924544

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5633430, 0.5651946
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8618083, 0.8615637
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6182990, 0.6159172
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6655746, 0.6633263
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5598650, 0.5599961
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6016982, 0.6062303
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9206905, 0.9234428
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8494058, 0.8490038
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5656571, 0.5654364
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6687453, 0.6684513

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 871

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6163

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2934344, upper bound: 0.2965493
time: 5.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2934344, upper bound: 0.2965493
time: 5.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5633430, 0.5651922
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8616414, 0.8615642
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6181593, 0.6159177
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6654973, 0.6633959
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5595827, 0.5599966
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6016982, 0.6061652
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9206896, 0.9234428
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8494964, 0.8489027
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5657320, 0.5653520
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6687453, 0.6683655

Time for backsubstitution: 22.03 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.05 + 555.24 = 614.29 seconds
