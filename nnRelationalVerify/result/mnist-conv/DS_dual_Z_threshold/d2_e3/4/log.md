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
execution time: IAR + RelationalAnalysis = 22.36 + 34.43 = 56.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.5250911, upper bound: 0.5250914

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6236
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6236

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246699, upper bound: 0.5250911
time: 5.14 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250911, upper bound: 0.5246712
time: 3.90 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.26 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.26
Output dim: 9, lower bound: -0.5246699, upper bound: 0.5250911
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.26
Output dim: 9, lower bound: -0.5250911, upper bound: 0.5246712

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0621605, 1.0593863
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4435220, 1.4449706
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0512977, 1.0477018
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3311844, 1.3281150
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9159889, 0.9177742
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8626499, 0.8589668
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9668727, 0.9658840
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2084131, 1.2098699
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2502928, 1.2511568
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9135504, 0.9135926

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246205, upper bound: 0.5250902
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246691, upper bound: 0.5250420
time: 4.91 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0593863, 1.0621605
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4449697, 1.4435220
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0477018, 1.0512977
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3281145, 1.3311844
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9177742, 0.9159889
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8589668, 0.8626499
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9658842, 0.9668725
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2098703, 1.2084136
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2511568, 1.2502928
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9135923, 0.9135506

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 524
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 524

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250420, upper bound: 0.5246704
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250902, upper bound: 0.5246219
time: 3.73 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.76 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.76
Output dim: 9, lower bound: -0.5246205, upper bound: 0.5250902
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.76
Output dim: 9, lower bound: -0.5246691, upper bound: 0.5250420
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.76
Output dim: 9, lower bound: -0.5250420, upper bound: 0.5246704
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.76
Output dim: 9, lower bound: -0.5250902, upper bound: 0.5246219

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0639749, 1.0586782
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4491482, 1.4427547
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0516677, 1.0475550
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3309631, 1.3286719
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9169860, 0.9173808
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8623390, 0.8597536
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9724369, 0.9636908
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2075834, 1.2119985
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2577477, 1.2482195
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9133735, 0.9140427

Time for backsubstitution: 20.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246205, upper bound: 0.5250845
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246130, upper bound: 0.5250916
time: 3.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246690, upper bound: 0.5250363
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246618, upper bound: 0.5250433
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0612011, 1.0614514
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4505968, 1.4413061
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0480714, 1.0511513
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3278942, 1.3317413
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9187713, 0.9155955
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8586559, 0.8634367
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9714484, 0.9646792
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2090387, 1.2105417
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2586117, 1.2473555
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9134154, 0.9140007

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250419, upper bound: 0.5246619
time: 6.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250349, upper bound: 0.5246704
time: 3.98 seconds

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

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250902, upper bound: 0.5246133
time: 9.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5250832, upper bound: 0.5246208
time: 4.99 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 36.61 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.61
Output dim: 9, lower bound: -0.5246205, upper bound: 0.5250845
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.61
Output dim: 9, lower bound: -0.5246130, upper bound: 0.5250916
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.61
Output dim: 9, lower bound: -0.5246690, upper bound: 0.5250363
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.61
Output dim: 9, lower bound: -0.5246618, upper bound: 0.5250433
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.61
Output dim: 9, lower bound: -0.5250419, upper bound: 0.5246619
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.61
Output dim: 9, lower bound: -0.5250349, upper bound: 0.5246704
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 36.61
Output dim: 9, lower bound: -0.5250902, upper bound: 0.5246133
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 36.61
Output dim: 9, lower bound: -0.5250832, upper bound: 0.5246208

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0635505, 1.0580311
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4444962, 1.4397268
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0488081, 1.0431623
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3305445, 1.3283987
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9159346, 0.9157677
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8595457, 0.8554635
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9693089, 0.9616542
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2020197, 1.2083764
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2563105, 1.2460179
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9127908, 0.9136620

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5242957, upper bound: 0.5246667
time: 7.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5242052, upper bound: 0.5247563
time: 4.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0633283, 1.0582538
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4461203, 1.4381027
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0472746, 1.0446954
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3306904, 1.3282528
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9153728, 0.9163294
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8580484, 0.8569603
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9704003, 0.9605629
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2039614, 1.2064347
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2555456, 1.2467833
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9129925, 0.9134603

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5242893, upper bound: 0.5246732
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5241973, upper bound: 0.5247639
time: 3.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0610275, 1.0587392
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4366541, 1.4419427
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0482912, 1.0433092
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3307657, 1.3276210
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9145441, 0.9161606
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8598566, 0.8543653
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9615512, 0.9638472
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2028484, 1.2054172
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2459192, 1.2489552
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9129686, 0.9130344

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5243437, upper bound: 0.5246197
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5242541, upper bound: 0.5247094
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0608053, 1.0589619
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4382782, 1.4403186
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0467582, 1.0448427
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3309116, 1.3274751
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9139824, 0.9167228
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8583598, 0.8558626
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9626427, 0.9627562
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2047901, 1.2034755
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2451534, 1.2497206
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9131703, 0.9128327

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5243373, upper bound: 0.5246263
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5242462, upper bound: 0.5247159
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0607767, 1.0608048
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4459448, 1.4382782
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0452118, 1.0467577
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3274746, 1.3314686
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9177198, 0.9139824
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8558626, 0.8591466
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9683204, 0.9626427
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2034750, 1.2069197
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2571754, 1.2451539
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9128327, 0.9136200

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247145, upper bound: 0.5242462
time: 7.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5243387
time: 4.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0605545, 1.0610275
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4475689, 1.4366541
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0436788, 1.0482912
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3276215, 1.3313222
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9171581, 0.9145441
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8543653, 0.8606434
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9694118, 0.9615512
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2054167, 1.2049780
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2564096, 1.2459192
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9130344, 0.9134183

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247081, upper bound: 0.5242542
time: 7.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246184, upper bound: 0.5243451
time: 3.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0582538, 1.0615134
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4381027, 1.4404941
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0446954, 1.0469055
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3276958, 1.3306909
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9163294, 0.9143758
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8561735, 0.8580484
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9605632, 0.9648356
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2043056, 1.2039609
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2467833, 1.2480907
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9130106, 0.9129922

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247626, upper bound: 0.5241986
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5242898
time: 4.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0580311, 1.0617356
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4397268, 1.4388700
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0431623, 1.0484385
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3278418, 1.3305445
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9157677, 0.9149375
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8546767, 0.8595457
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9616542, 0.9637444
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2062473, 1.2020192
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2460184, 1.2488561
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9132123, 0.9127908

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 845

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5247562, upper bound: 0.5242065
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5246665, upper bound: 0.5242971
time: 4.30 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5242957, upper bound: 0.5246667
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5242052, upper bound: 0.5247563
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5242893, upper bound: 0.5246732
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5241973, upper bound: 0.5247639
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5243437, upper bound: 0.5246197
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5242541, upper bound: 0.5247094
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5243373, upper bound: 0.5246263
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5242462, upper bound: 0.5247159
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5247145, upper bound: 0.5242462
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5246249, upper bound: 0.5243387
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5247081, upper bound: 0.5242542
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5246184, upper bound: 0.5243451
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5247626, upper bound: 0.5241986
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5246730, upper bound: 0.5242898
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5247562, upper bound: 0.5242065
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.22
Output dim: 9, lower bound: -0.5246665, upper bound: 0.5242971

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0613208, 1.0540967
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4397354, 1.4370356
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0469389, 1.0421638
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3305435, 1.3287988
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9159303, 0.9167819
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8564272, 0.8499527
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9678679, 0.9608381
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.1999092, 1.2046580
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2558193, 1.2451496
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9136519, 0.9136569

Time for backsubstitution: 20.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5241643, upper bound: 0.5245364
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5240738, upper bound: 0.5245355
time: 3.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0596161, 1.0558019
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4418058, 1.4349661
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0478096, 1.0412931
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3309441, 1.3283982
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9169488, 0.9157629
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8540344, 0.8523450
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9684925, 0.9602132
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.1983013, 1.2062664
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2554426, 1.2455263
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9127860, 0.9145231

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5240726, upper bound: 0.5245367
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5240735, upper bound: 0.5246250
time: 6.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0610986, 1.0543189
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4413595, 1.4354119
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0454059, 1.0436974
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3306904, 1.3286529
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9153686, 0.9173436
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8549299, 0.8514495
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9689593, 0.9597468
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2018509, 1.2027164
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2550545, 1.2459149
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9138532, 0.9134552

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5759
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 5759

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5241579, upper bound: 0.5245429
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5240659, upper bound: 0.5245421
time: 3.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.8120441, -14.7792912, -16.8120441, -14.7792912, -1.0593934, 1.0560241
1: -16.2553310, -14.1722717, -16.2553310, -14.1722717, -1.4434299, 1.4333420
2: -12.0810032, -10.6060753, -12.0810032, -10.6060753, -1.0462761, 1.0428267
3: -11.9948082, -10.3719950, -11.9948082, -10.3719950, -1.3310909, 1.3282523
4: -2.2575240, -1.1358814, -2.2575240, -1.1358814, -0.9163871, 0.9163251
5: -8.1275635, -6.6195669, -8.1275635, -6.6195669, -0.8525376, 0.8538423
6: -16.8353367, -15.0711985, -16.8353367, -15.0711985, -0.9695840, 0.9591222
7: -6.8141041, -5.2119651, -6.8141041, -5.2119651, -1.2002430, 1.2043247
8: -3.6343689, -2.3530221, -3.6343689, -2.3530221, -1.2546778, 1.2462916
9: 5.4975553, 6.7105274, 5.4975553, 6.7105274, -0.9129872, 0.9143214

Time for backsubstitution: 21.59 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.79 + 561.39 = 618.19 seconds
