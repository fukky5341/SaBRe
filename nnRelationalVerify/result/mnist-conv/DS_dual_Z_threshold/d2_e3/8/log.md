## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2940516


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5389104, 0.5389103)
1: (-11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4993663, 0.4993664)
2: (6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4197077, 0.4197077)
3: (-4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3537687, 0.3537687)
4: (-12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3765505, 0.3765505)
5: (-13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3536179, 0.3536178)
6: (-10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5343599, 0.5343599)
7: (-1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3434693, 0.3434693)
8: (-0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3664252, 0.3664252)
9: (-10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5006785, 0.5006785)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.51 + 34.50 = 57.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3267234, upper bound: 0.3267232

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 410

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3204588, upper bound: 0.3207861
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3207863, upper bound: 0.3204594
time: 2.91 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.37
Output dim: 2, lower bound: -0.3204588, upper bound: 0.3207861
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.37
Output dim: 2, lower bound: -0.3207863, upper bound: 0.3204594

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5000683, 0.5010936
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4428773, 0.4408654
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3857149, 0.3855838
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3337449, 0.3314050
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3744203, 0.3746868
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3324766, 0.3308541
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5387452, 0.5376310
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3248196, 0.3234736
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3656606, 0.3655997
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4823708, 0.4832615

Time for backsubstitution: 8.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 1698

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2987812, upper bound: 0.2991938
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2987812, upper bound: 0.2991938
time: 2.87 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5010937, 0.5000683
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4408654, 0.4428773
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3855838, 0.3857149
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3314050, 0.3337449
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3746868, 0.3744203
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3308541, 0.3324766
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5376310, 0.5387452
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3234736, 0.3248196
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3655996, 0.3656607
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4832615, 0.4823709

Time for backsubstitution: 8.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 1698

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2991933, upper bound: 0.2987817
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2991933, upper bound: 0.2987817
time: 2.90 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.73 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 2, lower bound: -0.2987812, upper bound: 0.2991938
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 2, lower bound: -0.2987812, upper bound: 0.2991938
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 2, lower bound: -0.2991933, upper bound: 0.2987817
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 2, lower bound: -0.2991933, upper bound: 0.2987817

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4997294, 0.5040256
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4482044, 0.4401467
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3845302, 0.3842182
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3333209, 0.3329509
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3739157, 0.3737320
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3317087, 0.3355244
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5382524, 0.5417273
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3293407, 0.3227635
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3646042, 0.3671208
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4825497, 0.4827645

Time for backsubstitution: 8.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2926480, upper bound: 0.2964990
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2965162, upper bound: 0.2948407
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5000683, 0.5007550
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4421585, 0.4408654
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3857149, 0.3843990
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3337449, 0.3309809
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3744203, 0.3741822
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3324766, 0.3300863
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5387452, 0.5371383
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3241095, 0.3234736
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3656606, 0.3645431
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4818738, 0.4832615

Time for backsubstitution: 9.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2926480, upper bound: 0.2964990
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2965162, upper bound: 0.2948407
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5007551, 0.5029999
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4461926, 0.4421586
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3843991, 0.3843494
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3309809, 0.3352908
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3741822, 0.3734655
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3300863, 0.3371469
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5371385, 0.5428413
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3279948, 0.3241095
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3645431, 0.3671818
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4834405, 0.4818738

Time for backsubstitution: 8.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2948402, upper bound: 0.2965167
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2964985, upper bound: 0.2926482
time: 4.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5010937, 0.4997295
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4401467, 0.4428773
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3855838, 0.3845301
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3314050, 0.3333207
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3746868, 0.3739157
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3308541, 0.3317088
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5376310, 0.5382525
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3227635, 0.3248196
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3655996, 0.3646041
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4827645, 0.4823709

Time for backsubstitution: 9.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2948402, upper bound: 0.2965167
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2964985, upper bound: 0.2926482
time: 4.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 17.04 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.04
Output dim: 2, lower bound: -0.2926480, upper bound: 0.2964990
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.04
Output dim: 2, lower bound: -0.2965162, upper bound: 0.2948407
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.04
Output dim: 2, lower bound: -0.2926480, upper bound: 0.2964990
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.04
Output dim: 2, lower bound: -0.2965162, upper bound: 0.2948407
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.04
Output dim: 2, lower bound: -0.2948402, upper bound: 0.2965167
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.04
Output dim: 2, lower bound: -0.2964985, upper bound: 0.2926482
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.04
Output dim: 2, lower bound: -0.2948402, upper bound: 0.2965167
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.04
Output dim: 2, lower bound: -0.2964985, upper bound: 0.2926482

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4766861, 0.4901245
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4303162, 0.4273874
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3697391, 0.3723530
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3300773, 0.3291754
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3273702, 0.3415247
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3066846, 0.3122644
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5213962, 0.5217675
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3093085, 0.3085043
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3622102, 0.3628821
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4685513, 0.4732459

Time for backsubstitution: 8.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2850971, upper bound: 0.2891544
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2850971, upper bound: 0.2890449
time: 2.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4865254, 0.4809821
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4354457, 0.4222584
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3754674, 0.3694272
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3295478, 0.3297074
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3430617, 0.3271865
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3084488, 0.3116576
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5182927, 0.5259495
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3150818, 0.3027312
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3603653, 0.3647268
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4734184, 0.4687659

Time for backsubstitution: 8.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2890595, upper bound: 0.2877222
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2890978, upper bound: 0.2874343
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4770249, 0.4868538
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4242703, 0.4281062
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3709241, 0.3725340
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3305012, 0.3272053
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3278724, 0.3419747
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3074524, 0.3068263
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5218890, 0.5171787
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3040771, 0.3092145
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3632666, 0.3603044
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4678752, 0.4737430

Time for backsubstitution: 8.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2850971, upper bound: 0.2891544
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2850971, upper bound: 0.2890449
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4868642, 0.4777114
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4293997, 0.4229772
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3766523, 0.3696080
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3299717, 0.3277373
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3435639, 0.3276367
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3092166, 0.3062194
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5187855, 0.5213606
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3098505, 0.3034414
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3614219, 0.3621491
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4727423, 0.4692628

Time for backsubstitution: 8.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2890595, upper bound: 0.2877222
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2890978, upper bound: 0.2874343
time: 2.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4777116, 0.4897960
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4283043, 0.4293997
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3696080, 0.3752864
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3277373, 0.3315178
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3276367, 0.3426117
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3062195, 0.3138869
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5213604, 0.5228816
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3079624, 0.3098506
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3621491, 0.3629431
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4694418, 0.4727423

Time for backsubstitution: 8.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2874338, upper bound: 0.2890983
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2877217, upper bound: 0.2890600
time: 2.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4868540, 0.4799564
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4334333, 0.4242703
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3725341, 0.3695585
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3272053, 0.3320473
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3419747, 0.3269199
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3068262, 0.3121227
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5171785, 0.5259852
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3137356, 0.3040772
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3603044, 0.3647878
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4739220, 0.4678752

Time for backsubstitution: 9.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2890444, upper bound: 0.2850976
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2891539, upper bound: 0.2850976
time: 2.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4780501, 0.4865254
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4222584, 0.4301184
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3707929, 0.3754673
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3281612, 0.3295478
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3281389, 0.3430616
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3069873, 0.3084488
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5218532, 0.5182928
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3027312, 0.3105607
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3632057, 0.3603654
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4687659, 0.4732394

Time for backsubstitution: 8.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2874338, upper bound: 0.2890983
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2877217, upper bound: 0.2890600
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4871925, 0.4766860
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4273874, 0.4249890
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3737190, 0.3697391
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3276292, 0.3300773
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3424770, 0.3273701
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3075941, 0.3066845
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5176711, 0.5213964
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3085043, 0.3047873
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3613609, 0.3622101
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4732458, 0.4683722

Time for backsubstitution: 8.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2890444, upper bound: 0.2850976
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2891539, upper bound: 0.2850976
time: 2.85 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2850971, upper bound: 0.2891544
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2850971, upper bound: 0.2890449
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2890595, upper bound: 0.2877222
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2890978, upper bound: 0.2874343
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2850971, upper bound: 0.2891544
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2850971, upper bound: 0.2890449
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2890595, upper bound: 0.2877222
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2890978, upper bound: 0.2874343
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2874338, upper bound: 0.2890983
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2877217, upper bound: 0.2890600
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2890444, upper bound: 0.2850976
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2891539, upper bound: 0.2850976
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2874338, upper bound: 0.2890983
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2877217, upper bound: 0.2890600
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2890444, upper bound: 0.2850976
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 14.96
Output dim: 2, lower bound: -0.2891539, upper bound: 0.2850976

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 57.01 + 222.71 = 279.72 seconds
