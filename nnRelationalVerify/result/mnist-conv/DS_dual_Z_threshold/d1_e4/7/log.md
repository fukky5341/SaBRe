## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.11948726150000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2823706, 0.2823703)
1: (-12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3489695, 0.3489695)
2: (-9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2716053, 0.2716053)
3: (-0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3234239, 0.3234239)
4: (-11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3780911, 0.3780909)
5: (7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2537158, 0.2537158)
6: (-6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2523494, 0.2523494)
7: (-15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4796824, 0.4796824)
8: (-3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417318, 0.2417318)
9: (-3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3122460, 0.3122458)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.95 + 34.68 = 57.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1200877, upper bound: 0.1200876

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4657

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200858, upper bound: 0.1193784
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1193783, upper bound: 0.1200859
time: 3.85 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.62 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.62
Output dim: 5, lower bound: -0.1200858, upper bound: 0.1193784
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.62
Output dim: 5, lower bound: -0.1193783, upper bound: 0.1200859

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2811632, 0.2820522
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3487532, 0.3481472
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2699523, 0.2711673
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3231140, 0.3233422
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3775542, 0.3779488
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2534165, 0.2525756
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2518638, 0.2505107
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4773383, 0.4790647
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2411778, 0.2396219
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3117379, 0.3121122

Time for backsubstitution: 20.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6137

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200856, upper bound: 0.1193489
time: 6.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200564, upper bound: 0.1193781
time: 6.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2820520, 0.2811631
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3481472, 0.3487532
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2711673, 0.2699523
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3233422, 0.3231140
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3779490, 0.3775544
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2525756, 0.2534165
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2505107, 0.2518638
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4790647, 0.4773381
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2396219, 0.2411778
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3121122, 0.3117378

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6137

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1193781, upper bound: 0.1200564
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1193491, upper bound: 0.1200856
time: 4.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.84 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.84
Output dim: 5, lower bound: -0.1200856, upper bound: 0.1193489
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.84
Output dim: 5, lower bound: -0.1200564, upper bound: 0.1193781
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.84
Output dim: 5, lower bound: -0.1193781, upper bound: 0.1200564
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.84
Output dim: 5, lower bound: -0.1193491, upper bound: 0.1200856

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2813005, 0.2814552
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3488843, 0.3475740
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2701383, 0.2703915
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3235756, 0.3213742
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3776467, 0.3775456
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2534263, 0.2525465
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2519213, 0.2502571
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4776115, 0.4778941
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2408125, 0.2397081
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3103544, 0.3124386

Time for backsubstitution: 21.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200842, upper bound: 0.1176137
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183503, upper bound: 0.1193476
time: 4.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2805662, 0.2820522
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3481798, 0.3481472
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2691765, 0.2711673
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3211460, 0.3233422
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3771513, 0.3779488
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2533875, 0.2525756
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2516102, 0.2505107
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4761674, 0.4790647
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2411778, 0.2392565
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3117379, 0.3107285

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200552, upper bound: 0.1176429
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183213, upper bound: 0.1193768
time: 5.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2821898, 0.2805661
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3482783, 0.3481798
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2713532, 0.2691765
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3238038, 0.3211460
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3780410, 0.3771510
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2525856, 0.2533875
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2505682, 0.2516102
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4793379, 0.4761674
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2392565, 0.2412641
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3107287, 0.3120642

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1193767, upper bound: 0.1183212
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1176428, upper bound: 0.1200551
time: 4.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2814550, 0.2811631
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3475740, 0.3487532
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2703915, 0.2699523
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3213742, 0.3231140
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3775456, 0.3775544
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2525465, 0.2534165
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2502571, 0.2518638
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4778941, 0.4773381
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2396219, 0.2408125
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3121122, 0.3103542

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1193477, upper bound: 0.1183504
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1176138, upper bound: 0.1200841
time: 4.23 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.51 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 5, lower bound: -0.1200842, upper bound: 0.1176137
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.51
Output dim: 5, lower bound: -0.1183503, upper bound: 0.1193476
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 5, lower bound: -0.1200552, upper bound: 0.1176429
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.51
Output dim: 5, lower bound: -0.1183213, upper bound: 0.1193768
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.51
Output dim: 5, lower bound: -0.1193767, upper bound: 0.1183212
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 5, lower bound: -0.1176428, upper bound: 0.1200551
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.51
Output dim: 5, lower bound: -0.1193477, upper bound: 0.1183504
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 5, lower bound: -0.1176138, upper bound: 0.1200841

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2754092, 0.2776928
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3458669, 0.3456461
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2698061, 0.2701781
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3110493, 0.3133799
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3732530, 0.3706584
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2499948, 0.2471721
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2488838, 0.2483168
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4677463, 0.4715941
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2375097, 0.2345368
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3011671, 0.2980483

Time for backsubstitution: 21.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200839, upper bound: 0.1174790
time: 4.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199494, upper bound: 0.1176134
time: 3.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2746749, 0.2782898
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3451624, 0.3462193
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2688444, 0.2709541
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3086196, 0.3153479
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3727573, 0.3710618
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2499559, 0.2472017
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2485727, 0.2485703
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4663024, 0.4727650
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2378750, 0.2340852
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3025507, 0.2963383

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200549, upper bound: 0.1175082
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199205, upper bound: 0.1176426
time: 4.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2784271, 0.2746747
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3463504, 0.3451624
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2711399, 0.2688444
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3158085, 0.3086196
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3711542, 0.3727572
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2472112, 0.2499559
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2486278, 0.2485728
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4730380, 0.4663024
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2340852, 0.2379612
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2963382, 0.3028767

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1176425, upper bound: 0.1199204
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1175081, upper bound: 0.1200546
time: 4.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2776928, 0.2752717
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3456461, 0.3457358
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2701781, 0.2696204
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3133799, 0.3105878
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3706585, 0.3731608
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2471721, 0.2499852
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2483168, 0.2488264
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4715943, 0.4674730
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2344505, 0.2375097
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2977220, 0.3011670

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1176135, upper bound: 0.1199496
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1174791, upper bound: 0.1200840
time: 3.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 5, lower bound: -0.1200839, upper bound: 0.1174790
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 5, lower bound: -0.1199494, upper bound: 0.1176134
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 5, lower bound: -0.1200549, upper bound: 0.1175082
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 5, lower bound: -0.1199205, upper bound: 0.1176426
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 5, lower bound: -0.1176425, upper bound: 0.1199204
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 5, lower bound: -0.1175081, upper bound: 0.1200546
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 5, lower bound: -0.1176135, upper bound: 0.1199496
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.09
Output dim: 5, lower bound: -0.1174791, upper bound: 0.1200840

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2670462, 0.2681221
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3319759, 0.3337059
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2698061, 0.2701950
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3100882, 0.3125386
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3581738, 0.3574626
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2493267, 0.2464089
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2425177, 0.2410141
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4579122, 0.4595799
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2360723, 0.2333550
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2827604, 0.2819442

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1189626, upper bound: 0.1174783
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200835, upper bound: 0.1163573
time: 5.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2658386, 0.2693297
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3339267, 0.3317552
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2698233, 0.2701778
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3102081, 0.3124187
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3600574, 0.3555794
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2492316, 0.2465043
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2415812, 0.2419506
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4557321, 0.4617603
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2363280, 0.2330993
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2850628, 0.2796415

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1188281, upper bound: 0.1176131
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199491, upper bound: 0.1164918
time: 4.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2663116, 0.2687193
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3312716, 0.3342793
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2688444, 0.2709713
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3076584, 0.3145068
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3576782, 0.3578662
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2492878, 0.2464381
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2422067, 0.2412678
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4564683, 0.4607511
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2364376, 0.2329035
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2841437, 0.2802340

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1189336, upper bound: 0.1175078
time: 5.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200544, upper bound: 0.1163865
time: 4.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2651041, 0.2699269
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3332224, 0.3323286
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2688615, 0.2709541
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3077784, 0.3143868
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3595614, 0.3559828
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2491927, 0.2465335
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2412702, 0.2422043
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4542880, 0.4629312
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2366933, 0.2326478
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2864463, 0.2779315

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1187992, upper bound: 0.1176422
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199200, upper bound: 0.1165210
time: 4.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2700639, 0.2651042
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3324594, 0.3332224
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2711399, 0.2688618
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3148474, 0.3077785
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3560750, 0.3595614
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2465432, 0.2491927
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2422618, 0.2412702
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4632037, 0.4542882
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2326478, 0.2367795
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2779315, 0.2867725

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165212, upper bound: 0.1199200
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1176422, upper bound: 0.1187990
time: 5.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2688563, 0.2663118
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3344102, 0.3312716
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2711570, 0.2688446
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3149673, 0.3076584
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3579583, 0.3576782
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2464480, 0.2492878
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2413253, 0.2422067
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4610236, 0.4564683
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2329035, 0.2365237
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2802341, 0.2844698

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1163867, upper bound: 0.1200544
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1175077, upper bound: 0.1189335
time: 4.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2693298, 0.2657014
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3317552, 0.3337958
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2701781, 0.2696381
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3124187, 0.3097466
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3555794, 0.3599648
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2465043, 0.2492219
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2419506, 0.2415239
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4617603, 0.4554591
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2330132, 0.2363280
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2793148, 0.2850629

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1164922, upper bound: 0.1199492
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1176129, upper bound: 0.1188283
time: 3.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2681222, 0.2669090
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3337059, 0.3318450
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2701952, 0.2696209
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3125386, 0.3096266
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3574629, 0.3580816
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2464089, 0.2493170
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2410141, 0.2424603
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4595799, 0.4576392
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2332689, 0.2360723
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2816174, 0.2827603

Time for backsubstitution: 21.90 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.63 + 549.09 = 606.71 seconds
