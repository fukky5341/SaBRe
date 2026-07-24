## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5245661088


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0760951, 1.0760951)
1: (-16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4490080, 1.4490080)
2: (-12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0568600, 1.0568600)
3: (-11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3407326, 1.3407326)
4: (-2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9267054, 0.9267054)
5: (-8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8810911, 0.8810914)
6: (-16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9718437, 0.9718435)
7: (-6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2129564, 1.2129569)
8: (-3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2451773, 1.2451777)
9: (5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9138064, 0.9138067)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.36 + 34.16 = 58.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.5250911, upper bound: 0.5250914

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 6236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250419, upper bound: 0.5250906
time: 4.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250902, upper bound: 0.5250423
time: 3.96 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.22 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.22
Output dim: 9, lower bound: -0.5250419, upper bound: 0.5250906
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.22
Output dim: 9, lower bound: -0.5250902, upper bound: 0.5250423

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0779099, 1.0753870
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4546342, 1.4467921
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0572305, 1.0567141
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3405132, 1.3412914
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9277024, 0.9263120
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8807802, 0.8818784
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9774084, 0.9696510
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2121267, 1.2150865
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2526321, 1.2422404
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9136291, 0.9142566

Time for backsubstitution: 21.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250307, upper bound: 0.5250767
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250271, upper bound: 0.5250803
time: 3.81 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0753870, 1.0760951
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4467912, 1.4490080
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0567141, 1.0568600
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3407326, 1.3405137
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9263120, 0.9267054
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8810911, 0.8807802
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9696512, 0.9718435
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2129564, 1.2121272
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2422409, 1.2451777
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9138064, 0.9136291

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6236

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246691, upper bound: 0.5250420
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250902, upper bound: 0.5246219
time: 3.53 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.39 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.39
Output dim: 9, lower bound: -0.5250307, upper bound: 0.5250767
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.39
Output dim: 9, lower bound: -0.5250271, upper bound: 0.5250803
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.39
Output dim: 9, lower bound: -0.5246691, upper bound: 0.5250420
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.39
Output dim: 9, lower bound: -0.5250902, upper bound: 0.5246219

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0779018, 1.0753865
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4546285, 1.4467940
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0572257, 1.0567141
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3405132, 1.3412910
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9277024, 0.9263120
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8807807, 0.8818765
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9773941, 0.9696517
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2121267, 1.2150865
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2526226, 1.2422409
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9136295, 0.9142559

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250263, upper bound: 0.5250764
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250305, upper bound: 0.5249715
time: 3.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0779099, 1.0753789
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4546342, 1.4467869
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0572305, 1.0567093
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3405132, 1.3412905
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9277024, 0.9263120
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8807783, 0.8818784
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9774084, 0.9696364
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2121267, 1.2150865
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2526321, 1.2422304
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9136286, 0.9142566

Time for backsubstitution: 21.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248508, upper bound: 0.5249458
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248923, upper bound: 0.5249031
time: 3.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0614519, 1.0593863
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4413061, 1.4449706
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0511508, 1.0477018
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3311844, 1.3278942
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9155955, 0.9177742
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8626499, 0.8586559
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9646792, 0.9658840
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2084131, 1.2090392
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2473555, 1.2511568
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9135504, 0.9134152

Time for backsubstitution: 21.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 5759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5245636, upper bound: 0.5250419
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246688, upper bound: 0.5250375
time: 4.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0586782, 1.0621605
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4427547, 1.4435220
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0475550, 1.0512977
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3281145, 1.3309641
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9173808, 0.9159889
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8589668, 0.8623390
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9636908, 0.9668725
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2098703, 1.2075830
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2482195, 1.2502928
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9135923, 0.9133732

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247626, upper bound: 0.5242065
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5242958
time: 3.91 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.80 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.80
Output dim: 9, lower bound: -0.5250263, upper bound: 0.5250764
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.80
Output dim: 9, lower bound: -0.5250305, upper bound: 0.5249715
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.80
Output dim: 9, lower bound: -0.5248508, upper bound: 0.5249458
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.80
Output dim: 9, lower bound: -0.5248923, upper bound: 0.5249031
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.80
Output dim: 9, lower bound: -0.5245636, upper bound: 0.5250419
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.80
Output dim: 9, lower bound: -0.5246688, upper bound: 0.5250375
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.80
Output dim: 9, lower bound: -0.5247626, upper bound: 0.5242065
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.80
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5242958

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0779018, 1.0753851
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4546285, 1.4467931
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0572262, 1.0567155
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3405132, 1.3412910
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9277024, 0.9263110
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8807797, 0.8818774
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9773932, 0.9696505
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2121263, 1.2150855
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2526226, 1.2422409
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9136291, 0.9142559

Time for backsubstitution: 21.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6236

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246054, upper bound: 0.5250764
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250263, upper bound: 0.5246556
time: 3.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0779018, 1.0753865
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4546285, 1.4467940
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0572257, 1.0567145
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3405132, 1.3412905
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9277015, 0.9263120
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8807807, 0.8818755
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9773941, 0.9696510
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2121263, 1.2150865
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2526226, 1.2422409
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9136295, 0.9142556

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247016, upper bound: 0.5245530
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246123, upper bound: 0.5246423
time: 3.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0762124, 1.0732436
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4319816, 1.4279075
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0545564, 1.0546842
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3444567, 1.3489146
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9270954, 0.9255834
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8553247, 0.8513384
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9719214, 0.9659836
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.1926651, 1.1917353
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2498951, 1.2385654
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9133625, 0.9140348

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248469, upper bound: 0.5249388
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248472, upper bound: 0.5249458
time: 3.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0757747, 1.0736809
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4357553, 1.4241343
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0552049, 1.0540352
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3481369, 1.3452339
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9269738, 0.9257050
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8502378, 0.8564253
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9737563, 0.9641488
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.1887760, 1.1956239
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2489681, 1.2394938
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9134064, 0.9139915

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 6236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5244780, upper bound: 0.5245297
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5244777, upper bound: 0.5246198
time: 3.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0614514, 1.0593848
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4413071, 1.4449706
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0511508, 1.0477023
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3311844, 1.3278942
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9155960, 0.9177732
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8626485, 0.8586550
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9646788, 0.9658830
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2084131, 1.2090392
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2473555, 1.2511568
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9135513, 0.9134152

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5245636, upper bound: 0.5250347
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5245564, upper bound: 0.5250416
time: 4.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0614519, 1.0593863
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4413061, 1.4449716
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0511508, 1.0477014
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3311844, 1.3278947
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9155955, 0.9177742
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8626499, 0.8586545
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9646792, 0.9658835
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2084131, 1.2090392
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2473555, 1.2511568
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9135504, 0.9134154

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5244898, upper bound: 0.5249023
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5245310, upper bound: 0.5248601
time: 3.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0564480, 1.0582252
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4379950, 1.4408321
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0456862, 1.0502996
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3281126, 1.3313642
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9173765, 0.9170032
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8558478, 0.8568273
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9622507, 0.9660568
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2077599, 1.2038646
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2477274, 1.2494226
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9144535, 0.9133685

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246312, upper bound: 0.5240748
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5245419, upper bound: 0.5240740
time: 3.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0547428, 1.0599303
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4400644, 1.4387622
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0465569, 1.0494289
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3285122, 1.3309636
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9183950, 0.9159846
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8534555, 0.8592200
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9628754, 0.9654319
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2061520, 1.2054734
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2473507, 1.2497993
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9135876, 0.9142344

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5245408, upper bound: 0.5240751
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5245416, upper bound: 0.5241657
time: 3.64 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.39 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5246054, upper bound: 0.5250764
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5250263, upper bound: 0.5246556
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5247016, upper bound: 0.5245530
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5246123, upper bound: 0.5246423
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5248469, upper bound: 0.5249388
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5248472, upper bound: 0.5249458
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5244780, upper bound: 0.5245297
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5244777, upper bound: 0.5246198
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5245636, upper bound: 0.5250347
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5245564, upper bound: 0.5250416
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5244898, upper bound: 0.5249023
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5245310, upper bound: 0.5248601
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5246312, upper bound: 0.5240748
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5245419, upper bound: 0.5240740
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5245408, upper bound: 0.5240751
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.39
Output dim: 9, lower bound: -0.5245416, upper bound: 0.5241657

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0639672, 1.0586762
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4491415, 1.4427547
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0516629, 1.0475564
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3309641, 1.3286724
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9169865, 0.9173808
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8623371, 0.8597522
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9724221, 0.9636903
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2075834, 1.2119985
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2577372, 1.2482195
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9133735, 0.9140420

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246054, upper bound: 0.5250733
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5245984, upper bound: 0.5250729
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0611935, 1.0614505
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4505901, 1.4413061
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0480671, 1.0511522
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3278942, 1.3317423
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9187717, 0.9155955
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8586540, 0.8634353
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9714336, 0.9646788
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2090387, 1.2105417
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2586012, 1.2473550
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9134154, 0.9140000

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248503, upper bound: 0.5245189
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5248917, upper bound: 0.5244777
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0756721, 1.0714512
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4498658, 1.4441013
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0553584, 1.0557175
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3405113, 1.3416896
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9276977, 0.9273262
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8776627, 0.8763652
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9759531, 0.9688354
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2100186, 1.2113690
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2521305, 1.2413716
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9144897, 0.9142504

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 6236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5245702, upper bound: 0.5244214
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5244812, upper bound: 0.5244208
time: 3.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0739670, 1.0731564
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4519372, 1.4420314
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0562286, 1.0548472
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3409119, 1.3412895
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9287162, 0.9263077
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8752699, 0.8787580
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9765778, 0.9682105
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2084088, 1.2129779
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2517529, 1.2417488
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9136238, 0.9151163

Time for backsubstitution: 21.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6236

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5241937, upper bound: 0.5246423
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246124, upper bound: 0.5242234
time: 3.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0757866, 1.0725956
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4273314, 1.4248796
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0516968, 1.0502911
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3440380, 1.3486423
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9260445, 0.9239707
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8525324, 0.8470483
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9687934, 0.9639468
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.1871014, 1.1881123
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2484608, 1.2363648
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9127798, 0.9136536

Time for backsubstitution: 22.34 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.52 + 557.55 = 616.07 seconds
