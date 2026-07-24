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
execution time: IAR + RelationalAnalysis = 21.63 + 33.17 = 54.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1419808, upper bound: 0.1419806

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 455
type: DSZ, layer: 1, pos: 525

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 455

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1402312
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1419794
time: 2.91 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.15 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.15
Output dim: 3, lower bound: -0.1419795, upper bound: 0.1402312
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.15
Output dim: 3, lower bound: -0.1402310, upper bound: 0.1419794

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4883065, 0.4875219
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3287597, 0.3267369
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4147420, 0.4136784
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3212447, 0.3193998
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3504493, 0.3500650
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3504183, 0.3528037
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4006486, 0.4027170
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3226409, 0.3234313
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6171288, 0.6172156
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5363955, 0.5359869

Time for backsubstitution: 20.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 525

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 525

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1406750, upper bound: 0.1402274
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1419761, upper bound: 0.1389265
time: 2.92 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4875216, 0.4883065
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3267369, 0.3287597
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.4136784, 0.4147420
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.3193998, 0.3212447
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3500648, 0.3504493
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3528037, 0.3504183
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.4027171, 0.4006485
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3234310, 0.3226408
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.6172161, 0.6171284
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5359869, 0.5363955

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 525

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 525

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1389265, upper bound: 0.1419759
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1402276, upper bound: 0.1406750
time: 3.10 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.60 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.60
Output dim: 3, lower bound: -0.1406750, upper bound: 0.1402274
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.60
Output dim: 3, lower bound: -0.1419761, upper bound: 0.1389265
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.60
Output dim: 3, lower bound: -0.1389265, upper bound: 0.1419759
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.60
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

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 1729

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 2376

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1383308, upper bound: 0.1313694
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1318147, upper bound: 0.1378818
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 1729

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 2376

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1396305, upper bound: 0.1300659
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1331181, upper bound: 0.1365819
time: 2.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 21.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 1729

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 2376

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1365821, upper bound: 0.1331183
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1300662, upper bound: 0.1396305
time: 2.94 seconds

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

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2376
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 1729

Time for candidate selection: 0.42 seconds

### Candidate
type: DSZ, layer: 3, pos: 2376

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1378818, upper bound: 0.1318145
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1313696, upper bound: 0.1383306
time: 2.92 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.12 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 28.12
Output dim: 3, lower bound: -0.1383308, upper bound: 0.1313694
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.12
Output dim: 3, lower bound: -0.1318147, upper bound: 0.1378818
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.12
Output dim: 3, lower bound: -0.1396305, upper bound: 0.1300659
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.12
Output dim: 3, lower bound: -0.1331181, upper bound: 0.1365819
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 28.12
Output dim: 3, lower bound: -0.1365821, upper bound: 0.1331183
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.12
Output dim: 3, lower bound: -0.1300662, upper bound: 0.1396305
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 28.12
Output dim: 3, lower bound: -0.1378818, upper bound: 0.1318145
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.12
Output dim: 3, lower bound: -0.1313696, upper bound: 0.1383306

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4268281, 0.4291418
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3250184, 0.3294483
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.3907781, 0.3897722
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.2744344, 0.2668216
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3856604, 0.3819811
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3504114, 0.3518097
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3853390, 0.3849949
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3240299, 0.3253167
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.5888367, 0.5921936
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5772934, 0.5644121

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 1729

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 2565

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1385445, upper bound: 0.1288911
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1385728, upper bound: 0.1289070
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1458006, -11.1196728, -12.1458006, -11.1196728, -0.4291418, 0.4268937
1: -10.2953033, -9.5193138, -10.2953033, -9.5193138, -0.3306992, 0.3250184
2: -2.5454104, -1.7512214, -2.5454104, -1.7512214, -0.3908751, 0.3907779
3: 5.9724727, 6.7451792, 5.9724727, 6.7451792, -0.2668215, 0.2751400
4: -11.1797190, -10.2502203, -11.1797190, -10.2502203, -0.3819814, 0.3858173
5: -6.6089749, -5.8434906, -6.6089749, -5.8434906, -0.3518095, 0.3504367
6: -12.3693848, -11.4272785, -12.3693848, -11.4272785, -0.3849947, 0.3857975
7: -6.4395571, -5.4970260, -6.4395571, -5.4970260, -0.3260236, 0.3240298
8: 2.1057334, 3.0144646, 2.1057334, 3.0144646, -0.5930061, 0.5888367
9: -6.2699022, -5.3168850, -6.2699022, -5.3168850, -0.5649614, 0.5772934

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1829
type: DSZ, layer: 3, pos: 674
type: DSZ, layer: 3, pos: 2229
type: DSZ, layer: 3, pos: 2147
type: DSZ, layer: 3, pos: 1382
type: DSZ, layer: 3, pos: 1998
type: DSZ, layer: 3, pos: 746
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2579
type: DSZ, layer: 3, pos: 717
type: DSZ, layer: 3, pos: 1465
type: DSZ, layer: 3, pos: 2536
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 429
type: DSZ, layer: 3, pos: 1501
type: DSZ, layer: 3, pos: 205
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 915
type: DSZ, layer: 3, pos: 2458
type: DSZ, layer: 3, pos: 1729

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 2565

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1289072, upper bound: 0.1385729
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1288909, upper bound: 0.1385446
time: 3.23 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.64 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.64
Output dim: 3, lower bound: -0.1385445, upper bound: 0.1288911
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.64
Output dim: 3, lower bound: -0.1385728, upper bound: 0.1289070
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.64
Output dim: 3, lower bound: -0.1289072, upper bound: 0.1385729
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.64
Output dim: 3, lower bound: -0.1288909, upper bound: 0.1385446

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 54.80 + 232.20 = 287.00 seconds
