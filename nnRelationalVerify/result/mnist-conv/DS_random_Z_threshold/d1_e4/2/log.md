## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13914137999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4895549, 0.4895554)
1: (-10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3319845, 0.3319844)
2: (-2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4164302, 0.4164302)
3: (5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3241732, 0.3241735)
4: (-11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3510623, 0.3510623)
5: (-6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3565919, 0.3565919)
6: (-12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4059968, 0.4059967)
7: (-6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3246870, 0.3246870)
8: (2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6173553, 0.6173553)
9: (-6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5370440, 0.5370440)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.06 + 32.58 = 56.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1419808, upper bound: 0.1419806

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 455

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 525

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1406763, upper bound: 0.1419773
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419774, upper bound: 0.1406761
time: 2.86 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.82 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.82
Output dim: 3, lower bound: -0.1406763, upper bound: 0.1419773
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.82
Output dim: 3, lower bound: -0.1419774, upper bound: 0.1406761

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4894509, 0.4896212
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3332338, 0.3300493
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4175332, 0.4147279
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3230791, 0.3248792
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3508184, 0.3512192
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3565519, 0.3566172
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4052875, 0.4064554
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3253939, 0.3235921
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6181684, 0.6160951
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5375938, 0.5361948

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 455

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 455

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1406750, upper bound: 0.1402274
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389265, upper bound: 0.1419759
time: 2.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4895549, 0.4894509
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3300490, 0.3319844
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4147279, 0.4164302
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3241732, 0.3230791
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3510623, 0.3508184
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3565919, 0.3565521
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4059968, 0.4052873
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3235919, 0.3246870
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6160951, 0.6173553
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5361948, 0.5370440

Time for backsubstitution: 22.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 455

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 455

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419761, upper bound: 0.1389265
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406750
time: 2.76 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.57 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.57
Output dim: 3, lower bound: -0.1406750, upper bound: 0.1402274
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.57
Output dim: 3, lower bound: -0.1389265, upper bound: 0.1419759
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.57
Output dim: 3, lower bound: -0.1419761, upper bound: 0.1389265
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.57
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406750

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4882026, 0.4875879
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3300095, 0.3248019
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4158447, 0.4119761
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3201506, 0.3201058
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3502052, 0.3502216
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3503783, 0.3528287
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3999386, 0.4031752
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3233478, 0.3223363
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6179414, 0.6159558
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5369444, 0.5351367

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 717

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1465

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1399891, upper bound: 0.1402242
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1406724, upper bound: 0.1395407
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4874177, 0.4883728
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3279867, 0.3268247
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4147811, 0.4130394
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3183057, 0.3219507
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3498209, 0.3506060
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3527637, 0.3504431
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4020071, 0.4011067
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3241382, 0.3215457
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6180286, 0.6158686
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5365362, 0.5355453

Time for backsubstitution: 22.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1998

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1829

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1362630, upper bound: 0.1394901
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1364402, upper bound: 0.1393124
time: 3.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4883065, 0.4874175
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3268247, 0.3267369
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4130394, 0.4136784
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3212447, 0.3183057
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3504493, 0.3498209
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3504183, 0.3527634
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4006486, 0.4020071
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3215458, 0.3234313
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6158686, 0.6172156
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5355453, 0.5359869

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 1382

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2620

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1401038, upper bound: 0.1385325
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1415812, upper bound: 0.1370550
time: 3.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4875216, 0.4882023
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3248019, 0.3287597
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4119761, 0.4147420
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3193998, 0.3201506
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3500648, 0.3502052
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3528037, 0.3503780
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4027171, 0.3999386
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3223362, 0.3226408
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6159558, 0.6171284
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5351367, 0.5363955

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1465

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2565

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1391110, upper bound: 0.1395579
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1391110, upper bound: 0.1395579
time: 2.90 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.67 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.67
Output dim: 3, lower bound: -0.1399891, upper bound: 0.1402242
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.67
Output dim: 3, lower bound: -0.1406724, upper bound: 0.1395407
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.67
Output dim: 3, lower bound: -0.1362630, upper bound: 0.1394901
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.67
Output dim: 3, lower bound: -0.1364402, upper bound: 0.1393124
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.67
Output dim: 3, lower bound: -0.1401038, upper bound: 0.1385325
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.67
Output dim: 3, lower bound: -0.1415812, upper bound: 0.1370550
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.67
Output dim: 3, lower bound: -0.1391110, upper bound: 0.1395579
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.67
Output dim: 3, lower bound: -0.1391110, upper bound: 0.1395579

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4842896, 0.4842169
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3303742, 0.3253020
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4157505, 0.4118721
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3201771, 0.3201365
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3456967, 0.3452311
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3514822, 0.3541861
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3961675, 0.3990982
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3191938, 0.3176874
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6178288, 0.6158309
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5357304, 0.5341959

Time for backsubstitution: 21.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 205

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1395207, upper bound: 0.1379172
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1376821, upper bound: 0.1397560
time: 3.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4848313, 0.4836755
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3305094, 0.3251668
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4157405, 0.4118822
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3201814, 0.3201323
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3452148, 0.3457129
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3517354, 0.3539329
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3958616, 0.3994042
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3186986, 0.3181825
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6178169, 0.6158433
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5360036, 0.5339227

Time for backsubstitution: 21.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2228

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1829

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1380088, upper bound: 0.1370540
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1381860, upper bound: 0.1368773
time: 3.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4854507, 0.4867067
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3267627, 0.3255817
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4141562, 0.4125822
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3138478, 0.3168724
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3475997, 0.3480396
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3522086, 0.3495867
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3997965, 0.3989725
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3220384, 0.3196517
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6168814, 0.6148963
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5327201, 0.5314078

Time for backsubstitution: 21.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1998

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2620

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1295155, upper bound: 0.1325616
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1295155, upper bound: 0.1325616
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4857516, 0.4864058
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3267436, 0.3256007
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4143240, 0.4124146
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3132277, 0.3174927
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3472548, 0.3483849
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3519075, 0.3498883
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3998728, 0.3988960
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3222442, 0.3194458
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6170564, 0.6147213
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5323982, 0.5317297

Time for backsubstitution: 21.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1382

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1350439, upper bound: 0.1373345
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1356176, upper bound: 0.1367695
time: 3.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4862969, 0.4867473
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3260972, 0.3264246
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4130244, 0.4136868
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3209217, 0.3181825
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3503311, 0.3499951
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3494740, 0.3510964
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4000115, 0.4017392
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3213158, 0.3227290
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6158967, 0.6169882
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5353417, 0.5350590

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 717

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2565

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1390252, upper bound: 0.1374098
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1390572, upper bound: 0.1374096
time: 2.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4876363, 0.4854076
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3265121, 0.3260099
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4130478, 0.4136639
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3211215, 0.3179827
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3506231, 0.3497031
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3487511, 0.3518193
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4003806, 0.4013700
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3208437, 0.3232012
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6156411, 0.6172438
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5346174, 0.5357838

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1400202, upper bound: 0.1343963
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1389209, upper bound: 0.1354953
time: 3.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4852490, 0.4864478
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3066132, 0.3129460
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4074039, 0.4116755
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3134854, 0.3142934
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3474915, 0.3463082
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3402364, 0.3347876
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3821516, 0.3810800
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3020530, 0.3055063
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6083255, 0.6047015
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5210600, 0.5208125

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 2228

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2458

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1381619, upper bound: 0.1391606
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1387136, upper bound: 0.1386086
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4857674, 0.4859293
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3089874, 0.3105708
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4089096, 0.4101698
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3135431, 0.3142357
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3461676, 0.3476317
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3372133, 0.3378112
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3838582, 0.3793731
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3052001, 0.3023573
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6035290, 0.6094975
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5195532, 0.5223188

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2228

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 674

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1370406, upper bound: 0.1384998
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1380516, upper bound: 0.1374887
time: 2.79 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1395207, upper bound: 0.1379172
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1376821, upper bound: 0.1397560
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1380088, upper bound: 0.1370540
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1381860, upper bound: 0.1368773
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1295155, upper bound: 0.1325616
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1295155, upper bound: 0.1325616
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1350439, upper bound: 0.1373345
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1356176, upper bound: 0.1367695
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1390252, upper bound: 0.1374098
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1390572, upper bound: 0.1374096
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1400202, upper bound: 0.1343963
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1389209, upper bound: 0.1354953
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1381619, upper bound: 0.1391606
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1387136, upper bound: 0.1386086
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1370406, upper bound: 0.1384998
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.11
Output dim: 3, lower bound: -0.1380516, upper bound: 0.1374887

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4852486, 0.4849627
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3287675, 0.3239799
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4148171, 0.4106104
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3200274, 0.3198323
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3497624, 0.3499355
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3460541, 0.3500125
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3982210, 0.4018354
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3215847, 0.3199847
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6179008, 0.6159334
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5364985, 0.5349898

Time for backsubstitution: 21.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2579

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1341124, upper bound: 0.1330233
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1346265, upper bound: 0.1325089
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4855771, 0.4846342
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3291872, 0.3235602
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4144790, 0.4109485
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3198771, 0.3199828
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3499191, 0.3497787
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3475621, 0.3485053
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3985987, 0.4014578
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3209960, 0.3205733
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6179194, 0.6159153
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5367975, 0.5346909

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 2229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1382

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1357845, upper bound: 0.1389689
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1362394, upper bound: 0.1384908
time: 3.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4860108, 0.4843426
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3073733, 0.3073589
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4035356, 0.4077086
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3203278, 0.3162649
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3503122, 0.3496382
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3425908, 0.3418725
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3977988, 0.3992832
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.2732420, 0.2759507
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6041088, 0.6065536
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5321422, 0.5264797

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 429

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1346119, upper bound: 0.1295021
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1351261, upper bound: 0.1289877
time: 3.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4871373, 0.4878101
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3239150, 0.3277949
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4115105, 0.4142821
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3144848, 0.3153591
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3471034, 0.3466833
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3527215, 0.3502898
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4023674, 0.3995243
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3109901, 0.3116591
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6159358, 0.6171083
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5359674, 0.5373034

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 1729
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1998

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1374101, upper bound: 0.1389547
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1379574, upper bound: 0.1383987
time: 2.96 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.65
Output dim: 3, lower bound: -0.1341124, upper bound: 0.1330233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.65
Output dim: 3, lower bound: -0.1346265, upper bound: 0.1325089
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.65
Output dim: 3, lower bound: -0.1357845, upper bound: 0.1389689
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.65
Output dim: 3, lower bound: -0.1362394, upper bound: 0.1384908
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.65
Output dim: 3, lower bound: -0.1346119, upper bound: 0.1295021
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.65
Output dim: 3, lower bound: -0.1351261, upper bound: 0.1289877
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.65
Output dim: 3, lower bound: -0.1374101, upper bound: 0.1389547
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.65
Output dim: 3, lower bound: -0.1379574, upper bound: 0.1383987

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.64 + 520.57 = 577.21 seconds
