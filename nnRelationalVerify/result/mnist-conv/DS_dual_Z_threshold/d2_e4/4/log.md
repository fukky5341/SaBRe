## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.8511987492


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4265327, 2.4265337)
1: (-14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4702597, 2.4702601)
2: (-8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2232647, 2.2232647)
3: (-6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3200216, 2.3200216)
4: (-11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.8002758, 2.8002748)
5: (-5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9139080, 1.9139075)
6: (-13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9840040, 1.9840040)
7: (-9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5707560, 2.5707560)
8: (8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4787288, 1.4787283)
9: (-6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7844839, 1.7844839)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.80 + 38.98 = 61.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.8520508, upper bound: 0.8520506

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520500, upper bound: 0.8514853
time: 7.78 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514852, upper bound: 0.8520497
time: 10.61 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 18.66 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 18.66
Output dim: 8, lower bound: -0.8520500, upper bound: 0.8514853
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 18.66
Output dim: 8, lower bound: -0.8514852, upper bound: 0.8520497

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4317436, 2.4309053
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4691048, 2.4678965
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2232962, 2.2233019
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2961426, 2.2927294
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7984304, 2.7986612
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9120779, 1.9115515
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9519367, 1.9457698
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5458269, 2.5500669
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4736576, 1.4726439
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7842798, 1.7840981

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520495, upper bound: 0.8511810
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517458, upper bound: 0.8514842
time: 8.78 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4309053, 2.4317436
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4678965, 2.4691048
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2233019, 2.2232962
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2927294, 2.2961426
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7986612, 2.7984304
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9115524, 1.9120779
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9457693, 1.9519367
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5500669, 2.5458269
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4726439, 1.4736576
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7840977, 1.7842798

Time for backsubstitution: 23.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514847, upper bound: 0.8517453
time: 14.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8511810, upper bound: 0.8520496
time: 9.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 47.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 47.20
Output dim: 8, lower bound: -0.8520495, upper bound: 0.8511810
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 47.20
Output dim: 8, lower bound: -0.8517458, upper bound: 0.8514842
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 47.20
Output dim: 8, lower bound: -0.8514847, upper bound: 0.8517453
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 47.20
Output dim: 8, lower bound: -0.8511810, upper bound: 0.8520496

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4304962, 2.4298124
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4755583, 2.4729471
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2054157, 2.2076540
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2947731, 2.2915316
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7843494, 2.7802982
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9039011, 1.9043951
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9487867, 1.9430113
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5453892, 2.5480661
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4711375, 1.4691014
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7684355, 1.7659941

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4555

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8500733, upper bound: 0.8511769
time: 9.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520455, upper bound: 0.8492055
time: 8.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4306507, 2.4296579
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4741554, 2.4743500
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2076483, 2.2054205
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2949438, 2.2913599
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7800674, 2.7845793
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9049206, 1.9033756
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9491777, 1.9426198
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5438261, 2.5496292
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4701152, 1.4701238
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7661753, 1.7682538

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4555

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8497697, upper bound: 0.8514807
time: 7.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517418, upper bound: 0.8495084
time: 8.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4296579, 2.4306507
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4743500, 2.4741554
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2054205, 2.2076483
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2913599, 2.2949438
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7845793, 2.7800674
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9033756, 1.9049206
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9426193, 1.9491782
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5496292, 2.5438261
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4701238, 1.4701152
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7682533, 1.7661757

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4555

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8495087, upper bound: 0.8517416
time: 8.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514807, upper bound: 0.8497696
time: 9.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4298124, 2.4304962
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4729471, 2.4755583
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2076540, 2.2054148
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2915316, 2.2947731
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7802973, 2.7843494
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9043951, 1.9039011
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9430113, 1.9487867
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5480661, 2.5453892
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4691014, 1.4711375
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7659941, 1.7684355

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4555

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8492049, upper bound: 0.8520451
time: 7.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8511770, upper bound: 0.8500730
time: 9.62 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 39.58 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 39.58
Output dim: 8, lower bound: -0.8500733, upper bound: 0.8511769
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.58
Output dim: 8, lower bound: -0.8520455, upper bound: 0.8492055
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.58
Output dim: 8, lower bound: -0.8497697, upper bound: 0.8514807
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.58
Output dim: 8, lower bound: -0.8517418, upper bound: 0.8495084
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.58
Output dim: 8, lower bound: -0.8495087, upper bound: 0.8517416
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.58
Output dim: 8, lower bound: -0.8514807, upper bound: 0.8497696
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.58
Output dim: 8, lower bound: -0.8492049, upper bound: 0.8520451
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 39.58
Output dim: 8, lower bound: -0.8511770, upper bound: 0.8500730

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4295301, 2.4287071
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4779692, 2.4757581
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1990995, 2.2021265
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2952824, 2.2919683
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7717066, 2.7692375
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9108734, 1.9103794
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9483480, 1.9425130
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5429411, 2.5459213
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4616771, 1.4582872
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7745976, 1.7712808

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520410, upper bound: 0.8489530
time: 7.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517940, upper bound: 0.8492003
time: 12.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4295464, 2.4286919
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4769659, 2.4767604
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2021217, 2.1991043
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2953806, 2.2918692
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7690077, 2.7719364
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9109049, 1.9103479
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9486799, 1.9421802
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5416813, 2.5471802
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4593015, 1.4606638
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7714629, 1.7744155

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8497653, upper bound: 0.8512292
time: 9.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8495183, upper bound: 0.8514765
time: 6.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4296846, 2.4285536
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4765654, 2.4771609
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2013321, 2.1998930
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2954531, 2.2917967
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7674246, 2.7735195
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9118929, 1.9093599
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9487391, 1.9421215
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5413780, 2.5474844
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4606547, 1.4593096
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7723374, 1.7735410

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517373, upper bound: 0.8492587
time: 6.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514903, upper bound: 0.8495040
time: 8.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4285536, 2.4296846
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4771605, 2.4765658
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1998940, 2.2013321
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2917967, 2.2954531
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7735195, 2.7674246
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9093599, 1.9118929
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9421206, 1.9487386
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5474844, 2.5413771
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4593101, 1.4606552
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7735410, 1.7723374

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8495041, upper bound: 0.8514903
time: 8.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8492572, upper bound: 0.8517375
time: 12.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4286919, 2.4295464
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4767599, 2.4769654
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1991043, 2.2021208
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2918692, 2.2953806
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7719364, 2.7690077
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9103479, 1.9109049
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9421797, 1.9486799
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5471802, 2.5416813
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4606633, 1.4593010
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7744155, 1.7714629

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514762, upper bound: 0.8495197
time: 8.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8512292, upper bound: 0.8497650
time: 8.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4287071, 2.4295301
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4757566, 2.4779687
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2021265, 2.1990986
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2919683, 2.2952814
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7692375, 2.7717066
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9103794, 1.9108734
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9425135, 1.9483471
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5459213, 2.5429411
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4582877, 1.4616776
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7712808, 1.7745976

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8492004, upper bound: 0.8517937
time: 10.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8489535, upper bound: 0.8520409
time: 8.49 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 41.26 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8520410, upper bound: 0.8489530
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8517940, upper bound: 0.8492003
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8497653, upper bound: 0.8512292
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8495183, upper bound: 0.8514765
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8517373, upper bound: 0.8492587
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8514903, upper bound: 0.8495040
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8495041, upper bound: 0.8514903
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8492572, upper bound: 0.8517375
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8514762, upper bound: 0.8495197
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8512292, upper bound: 0.8497650
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8492004, upper bound: 0.8517937
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.26
Output dim: 8, lower bound: -0.8489535, upper bound: 0.8520409

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4299402, 2.4283218
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4784193, 2.4753242
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1997194, 2.2015352
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2961426, 2.2911463
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7706518, 2.7703457
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9102421, 1.9110446
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9482851, 1.9425778
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5426302, 2.5462475
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4620256, 1.4579554
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7747393, 1.7711506

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 5761

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520402, upper bound: 0.8489526
time: 8.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520402, upper bound: 0.8489527
time: 10.96 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 42.14 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 42.14
Output dim: 8, lower bound: -0.8520402, upper bound: 0.8489526
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 42.14
Output dim: 8, lower bound: -0.8520402, upper bound: 0.8489527
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8517940, upper bound: 0.8492003
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8497653, upper bound: 0.8512292
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8495183, upper bound: 0.8514765
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8517373, upper bound: 0.8492587
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8514903, upper bound: 0.8495040
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8495041, upper bound: 0.8514903
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8492572, upper bound: 0.8517375
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8514762, upper bound: 0.8495197
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8512292, upper bound: 0.8497650
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8492004, upper bound: 0.8517937
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 42.14
Output dim: 8, lower bound: -0.8489535, upper bound: 0.8520409

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 61.78 + 547.96 = 609.74 seconds
