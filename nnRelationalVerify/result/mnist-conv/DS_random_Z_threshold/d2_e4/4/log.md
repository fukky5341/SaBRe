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
execution time: IAR + RelationalAnalysis = 24.55 + 39.81 = 64.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.8520508, upper bound: 0.8520506

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520503, upper bound: 0.8517457
time: 7.10 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517466, upper bound: 0.8520499
time: 8.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.14 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.14
Output dim: 8, lower bound: -0.8520503, upper bound: 0.8517457
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.14
Output dim: 8, lower bound: -0.8517466, upper bound: 0.8520499

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4252882, 2.4254427
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4767132, 2.4753103
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2053843, 2.2076178
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3186550, 2.3188267
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7861958, 2.7819138
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9057283, 1.9067488
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9808521, 1.9812441
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5703163, 2.5687532
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4762087, 1.4751863
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7686415, 1.7663813

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8520458, upper bound: 0.8514949
time: 6.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517989, upper bound: 0.8517416
time: 7.01 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4254417, 2.4252882
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4753103, 2.4767132
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2076178, 2.2053843
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3188257, 2.3186550
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7819138, 2.7861958
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9067488, 1.9057283
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9812431, 1.9808526
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5687532, 2.5703163
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4751863, 1.4762087
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7663813, 1.7686410

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4625

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514955, upper bound: 0.8520492
time: 8.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517461, upper bound: 0.8517995
time: 7.48 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 38.62 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 38.62
Output dim: 8, lower bound: -0.8520458, upper bound: 0.8514949
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 38.62
Output dim: 8, lower bound: -0.8517989, upper bound: 0.8517416
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 38.62
Output dim: 8, lower bound: -0.8514955, upper bound: 0.8520492
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 38.62
Output dim: 8, lower bound: -0.8517461, upper bound: 0.8517995

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4256954, 2.4250546
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4771652, 2.4748783
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2060032, 2.2070255
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3195152, 2.3180046
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7851410, 2.7830210
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9050980, 1.9074159
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9807892, 1.9813085
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5700045, 2.5690784
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4765568, 1.4748540
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7687840, 1.7662520

Time for backsubstitution: 23.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8511420, upper bound: 0.8503183
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8508691, upper bound: 0.8505909
time: 17.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4249001, 2.4254427
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4762812, 2.4753103
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.2047920, 2.2076178
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3178329, 2.3188267
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7861958, 2.7808590
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9057283, 1.9061189
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9808521, 1.9811816
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5703163, 2.5684414
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4758763, 1.4751863
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7685122, 1.7663813

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517951, upper bound: 0.8462206
time: 8.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8462794, upper bound: 0.8517392
time: 8.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4109802, 2.4081593
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4499512, 2.4545193
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1978483, 2.1968346
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3124676, 2.3133736
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7797861, 2.7843323
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9061928, 1.9046569
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9419947, 1.9360108
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5453396, 2.5498209
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4760547, 1.4777436
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7669935, 1.7693529

Time for backsubstitution: 23.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514955, upper bound: 0.8517358
time: 9.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514946, upper bound: 0.8520497
time: 8.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4083138, 2.4108257
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4531155, 2.4513531
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1990681, 2.1956158
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3135443, 2.3122969
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7800503, 2.7840681
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9056759, 1.9051728
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9364023, 1.9416037
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5482578, 2.5469027
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4767213, 1.4770775
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7670937, 1.7692533

Time for backsubstitution: 23.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 581

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517451, upper bound: 0.8509107
time: 8.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8508559, upper bound: 0.8517980
time: 8.99 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 40.58 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 40.58
Output dim: 8, lower bound: -0.8511420, upper bound: 0.8503183
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 40.58
Output dim: 8, lower bound: -0.8508691, upper bound: 0.8505909
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 40.58
Output dim: 8, lower bound: -0.8517951, upper bound: 0.8462206
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 40.58
Output dim: 8, lower bound: -0.8462794, upper bound: 0.8517392
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 40.58
Output dim: 8, lower bound: -0.8514955, upper bound: 0.8517358
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 40.58
Output dim: 8, lower bound: -0.8514946, upper bound: 0.8520497
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 40.58
Output dim: 8, lower bound: -0.8517451, upper bound: 0.8509107
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 40.58
Output dim: 8, lower bound: -0.8508559, upper bound: 0.8517980

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4377136, 2.4340963
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4900351, 2.4857254
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1051245, 2.1204052
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2923126, 2.2964859
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7492990, 2.7372284
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.8802071, 1.8769498
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9553757, 1.9520659
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5732813, 2.5669985
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4415565, 1.4359527
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7680149, 1.7658157

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517951, upper bound: 0.8462207
time: 10.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514798, upper bound: 0.8462216
time: 5.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4335566, 2.4382524
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4866953, 2.4890642
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1175795, 2.1079493
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2954941, 2.2933044
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7425642, 2.7439613
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.8765602, 1.8805957
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9517365, 1.9557037
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5688744, 2.5714045
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4366422, 1.4408665
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7679462, 1.7658834

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4555

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8443037, upper bound: 0.8517332
time: 7.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8462755, upper bound: 0.8497615
time: 9.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4109793, 2.4081583
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4499512, 2.4545193
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1978483, 2.1968346
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3124666, 2.3133726
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7797852, 2.7843304
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9061928, 1.9046569
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9419956, 1.9360113
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5453386, 2.5498199
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4760542, 1.4777427
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7669926, 1.7693529

Time for backsubstitution: 23.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8513276, upper bound: 0.8511265
time: 7.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8508863, upper bound: 0.8515682
time: 6.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4109802, 2.4081593
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4499502, 2.4545193
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1978493, 2.1968346
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3124676, 2.3133717
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7797842, 2.7843304
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9061928, 1.9046569
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9419956, 1.9360113
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5453386, 2.5498199
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4760537, 1.4777427
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7669926, 1.7693529

Time for backsubstitution: 23.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5761

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514937, upper bound: 0.8520488
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8514937, upper bound: 0.8520488
time: 5.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4060583, 2.4082489
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4457235, 2.4450455
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1985846, 2.1957293
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3121700, 2.3121347
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7792301, 2.7835054
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9044180, 1.9040728
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9214430, 1.9285140
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5411110, 2.5387354
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4731760, 1.4725375
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7683673, 1.7702537

Time for backsubstitution: 23.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 4555
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8517443, upper bound: 0.8503435
time: 7.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8511795, upper bound: 0.8509099
time: 8.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4057369, 2.4085703
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4468079, 2.4439611
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1991816, 2.1951323
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.3133821, 2.3109226
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7794886, 2.7832479
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.9045763, 1.9039154
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9233131, 1.9266443
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5400906, 2.5397558
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4721813, 1.4735317
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7680936, 1.7705264

Time for backsubstitution: 23.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 6195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8508559, upper bound: 0.8514843
time: 7.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8508549, upper bound: 0.8517981
time: 6.49 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 37.65 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8517951, upper bound: 0.8462207
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8514798, upper bound: 0.8462216
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8443037, upper bound: 0.8517332
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8462755, upper bound: 0.8497615
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8513276, upper bound: 0.8511265
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8508863, upper bound: 0.8515682
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8514937, upper bound: 0.8520488
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8514937, upper bound: 0.8520488
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8517443, upper bound: 0.8503435
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8511795, upper bound: 0.8509099
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8508559, upper bound: 0.8514843
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.65
Output dim: 8, lower bound: -0.8508549, upper bound: 0.8517981

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4377127, 2.4340954
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4900360, 2.4857264
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1051245, 2.1204047
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2923126, 2.2964869
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7492981, 2.7372265
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.8802071, 1.8769498
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9553757, 1.9520655
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5732822, 2.5669994
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4415550, 1.4359512
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7680140, 1.7658153

Time for backsubstitution: 23.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 5761
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 4555

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8508923, upper bound: 0.8450488
time: 5.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8506193, upper bound: 0.8453201
time: 12.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.4006472, -2.5856199, -5.4006472, -2.5856199, -2.4377117, 2.4340944
1: -14.5097742, -11.6001644, -14.5097742, -11.6001644, -2.4900351, 2.4857264
2: -8.4967813, -5.5774755, -8.4967813, -5.5774755, -2.1051245, 2.1204042
3: -6.8287039, -4.2764769, -6.8287039, -4.2764769, -2.2923136, 2.2964859
4: -11.1889839, -8.1334801, -11.1889839, -8.1334801, -2.7492971, 2.7372265
5: -5.3435760, -2.9016693, -5.3435760, -2.9016693, -1.8802071, 1.8769498
6: -13.0153589, -10.1304665, -13.0153589, -10.1304665, -1.9553738, 1.9520655
7: -9.4142399, -6.6494508, -9.4142399, -6.6494508, -2.5732813, 2.5669994
8: 8.5683594, 10.6499119, 8.5683594, 10.6499119, -1.4415550, 1.4359512
9: -6.3254209, -3.9187176, -6.3254209, -3.9187176, -1.7680140, 1.7658153

Time for backsubstitution: 23.46 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 64.35 + 550.38 = 614.73 seconds
