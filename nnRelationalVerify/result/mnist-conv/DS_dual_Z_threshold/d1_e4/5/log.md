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
execution time: IAR + RelationalAnalysis = 23.18 + 34.75 = 57.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.2968933, upper bound: 0.2968939

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6183
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6183

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2939767, upper bound: 0.2968914
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968915, upper bound: 0.2939766
time: 4.43 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.23 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.23
Output dim: 0, lower bound: -0.2939767, upper bound: 0.2968914
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.23
Output dim: 0, lower bound: -0.2968915, upper bound: 0.2939766

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5634542, 0.5653996
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8602457, 0.8606906
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6168156, 0.6149783
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6666479, 0.6646276
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5648825, 0.5661786
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6012964, 0.6055317
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9215612, 0.9244375
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8512335, 0.8505130
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5664821, 0.5660815
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6681788, 0.6675158

Time for backsubstitution: 20.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6163

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2939766, upper bound: 0.2968915
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2939766, upper bound: 0.2968914
time: 4.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5653996, 0.5634546
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8606901, 0.8602462
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6149783, 0.6168156
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6646276, 0.6666479
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5661786, 0.5648825
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6055317, 0.6012964
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9244375, 0.9215612
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8505130, 0.8512335
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5660815, 0.5664821
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6675155, 0.6681786

Time for backsubstitution: 20.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6163

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968914, upper bound: 0.2939772
time: 6.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968914, upper bound: 0.2939766
time: 5.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 32.53 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.53
Output dim: 0, lower bound: -0.2939766, upper bound: 0.2968915
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.53
Output dim: 0, lower bound: -0.2939766, upper bound: 0.2968914
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.53
Output dim: 0, lower bound: -0.2968914, upper bound: 0.2939772
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.53
Output dim: 0, lower bound: -0.2968914, upper bound: 0.2939766

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5654640, 0.5669618
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8451109, 0.8474503
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6031318, 0.6030068
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6583695, 0.6585164
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5544343, 0.5570357
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.5994396, 0.6034100
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9130163, 0.9139624
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8336487, 0.8302999
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5587454, 0.5596204
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6658411, 0.6648455

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2887388, upper bound: 0.2968897
time: 6.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2939748, upper bound: 0.2916536
time: 4.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5650163, 0.5674090
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8470058, 0.8455548
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6048446, 0.6012940
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6605368, 0.6563492
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5557394, 0.5557303
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.5991745, 0.6036749
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9110861, 0.9158926
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8310204, 0.8329282
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5600209, 0.5583448
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6655087, 0.6651783

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2887388, upper bound: 0.2968894
time: 6.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2939748, upper bound: 0.2916536
time: 4.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5674090, 0.5650163
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8455553, 0.8470058
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6012940, 0.6048446
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6563492, 0.6605368
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5557303, 0.5557396
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6036749, 0.5991747
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9158926, 0.9110861
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8329282, 0.8310204
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5583448, 0.5600209
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6651783, 0.6655083

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2916536, upper bound: 0.2939747
time: 6.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968896, upper bound: 0.2887394
time: 3.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5669618, 0.5654640
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8474503, 0.8451104
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.6030068, 0.6031313
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6585164, 0.6583695
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5570354, 0.5544345
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6034098, 0.5994396
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.9139624, 0.9130163
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8302999, 0.8336487
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5596204, 0.5587454
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6648455, 0.6658411

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 554

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.2916536, upper bound: 0.2939747
time: 6.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2968896, upper bound: 0.2887394
time: 4.04 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.73 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.73
Output dim: 0, lower bound: -0.2887388, upper bound: 0.2968897
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.73
Output dim: 0, lower bound: -0.2939748, upper bound: 0.2916536
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.73
Output dim: 0, lower bound: -0.2887388, upper bound: 0.2968894
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.73
Output dim: 0, lower bound: -0.2939748, upper bound: 0.2916536
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.73
Output dim: 0, lower bound: -0.2916536, upper bound: 0.2939747
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.73
Output dim: 0, lower bound: -0.2968896, upper bound: 0.2887394
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.73
Output dim: 0, lower bound: -0.2916536, upper bound: 0.2939747
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.73
Output dim: 0, lower bound: -0.2968896, upper bound: 0.2887394

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5283728, 0.5346897
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8478775, 0.8522072
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5971546, 0.5991831
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6490164, 0.6475339
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5359523, 0.5354166
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.5973015, 0.6015389
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8886681, 0.8926582
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8260202, 0.8215818
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5603924, 0.5615478
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6544013, 0.6511936

Time for backsubstitution: 21.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2886206, upper bound: 0.2956375
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2874881, upper bound: 0.2967712
time: 4.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5279255, 0.5351372
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8497734, 0.8503118
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5988674, 0.5974703
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6511836, 0.6453662
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5372579, 0.5341113
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.5970364, 0.6018038
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8867378, 0.8945885
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8233919, 0.8242102
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5616679, 0.5602722
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6540689, 0.6515265

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2886206, upper bound: 0.2956395
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2874862, upper bound: 0.2967714
time: 6.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5351372, 0.5279253
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8503113, 0.8497734
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5974703, 0.5988674
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6453662, 0.6511836
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5341113, 0.5372577
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6018038, 0.5970364
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8945885, 0.8867378
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8242102, 0.8233919
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5602722, 0.5616679
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6515260, 0.6540689

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2967714, upper bound: 0.2874862
time: 8.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2956389, upper bound: 0.2886205
time: 6.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5346899, 0.5283730
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8522072, 0.8478780
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5991831, 0.5971546
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6475339, 0.6490164
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5354164, 0.5359526
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6015387, 0.5973015
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8926582, 0.8886681
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8215818, 0.8260202
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5615478, 0.5603924
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6511936, 0.6544018

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 871

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2967714, upper bound: 0.2874882
time: 11.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2956370, upper bound: 0.2886205
time: 4.09 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 37.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.13
Output dim: 0, lower bound: -0.2886206, upper bound: 0.2956375
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.13
Output dim: 0, lower bound: -0.2874881, upper bound: 0.2967712
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.13
Output dim: 0, lower bound: -0.2886206, upper bound: 0.2956395
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.13
Output dim: 0, lower bound: -0.2874862, upper bound: 0.2967714
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.13
Output dim: 0, lower bound: -0.2967714, upper bound: 0.2874862
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.13
Output dim: 0, lower bound: -0.2956389, upper bound: 0.2886205
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.13
Output dim: 0, lower bound: -0.2967714, upper bound: 0.2874882
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.13
Output dim: 0, lower bound: -0.2956370, upper bound: 0.2886205

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5316143, 0.5374513
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8389773, 0.8444209
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5922513, 0.5948930
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6460977, 0.6454844
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5351522, 0.5347052
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.5969558, 0.6013472
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8878069, 0.8911009
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8284388, 0.8236423
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5581732, 0.5596070
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6527398, 0.6492953

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 5826

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2886205, upper bound: 0.2954724
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2884558, upper bound: 0.2956375
time: 4.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5311346, 0.5379310
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8400922, 0.8433065
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5928645, 0.5942793
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6469669, 0.6446152
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5352409, 0.5346162
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.5971098, 0.6011934
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8871107, 0.8917971
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8280807, 0.8240004
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5584512, 0.5593290
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6525033, 0.6495318

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 5826

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2874880, upper bound: 0.2966066
time: 6.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2873233, upper bound: 0.2967713
time: 5.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5311670, 0.5378990
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8408723, 0.8425255
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5939641, 0.5931802
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6482654, 0.6433167
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5364573, 0.5334001
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.5966911, 0.6016121
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8858767, 0.8930311
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8258104, 0.8262706
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5594487, 0.5583315
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6524069, 0.6496277

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5826

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2886205, upper bound: 0.2954747
time: 4.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2884558, upper bound: 0.2956388
time: 5.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5306873, 0.5383790
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8419881, 0.8414106
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5945778, 0.5925665
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6491342, 0.6424479
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5365465, 0.5333111
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.5968447, 0.6014583
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8851805, 0.8937273
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8254523, 0.8266287
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5597267, 0.5580535
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6521709, 0.6498642

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 5826

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2874860, upper bound: 0.2966072
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2873213, upper bound: 0.2967711
time: 24.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5383787, 0.5306869
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8414111, 0.8419876
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5925665, 0.5945778
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6424479, 0.6491342
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5333111, 0.5365462
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6014581, 0.5968449
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8937273, 0.8851805
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8266287, 0.8254523
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5580535, 0.5597267
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6498644, 0.6521707

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5826
type: DSZ, layer: 1, pos: 6143
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 5831
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 5826

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2967713, upper bound: 0.2873219
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.2966066, upper bound: 0.2874866
time: 11.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 10.2391586, 11.2704544, 10.2391586, 11.2704544, -0.5378990, 0.5311668
1: -16.7365818, -15.2644901, -16.7365818, -15.2644901, -0.8425260, 0.8408728
2: -4.6894851, -3.6523623, -4.6894851, -3.6523623, -0.5931802, 0.5939641
3: -12.7181892, -11.6473284, -12.7181892, -11.6473284, -0.6433167, 0.6482654
4: -10.3790941, -9.2026939, -10.3790941, -9.2026939, -0.5333998, 0.5364575
5: -7.7704892, -6.6585083, -7.7704892, -6.6585083, -0.6016121, 0.5966909
6: -5.4215307, -4.3090539, -5.4215307, -4.3090539, -0.8930311, 0.8858767
7: -11.3050489, -9.8765717, -11.3050489, -9.8765717, -0.8262706, 0.8258104
8: -2.8618202, -1.9497161, -2.8618202, -1.9497161, -0.5583315, 0.5594487
9: -2.4918106, -1.2521186, -2.4918106, -1.2521186, -0.6496279, 0.6524072

Time for backsubstitution: 21.88 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.94 + 547.75 = 605.69 seconds
