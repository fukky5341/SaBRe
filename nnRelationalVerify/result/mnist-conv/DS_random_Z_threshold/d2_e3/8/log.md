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
execution time: IAR + RelationalAnalysis = 24.70 + 35.01 = 59.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3267234, upper bound: 0.3267232

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2320
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 2516

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2320

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3192825, upper bound: 0.3200981
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3200976, upper bound: 0.3192831
time: 4.33 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.58 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.58
Output dim: 2, lower bound: -0.3192825, upper bound: 0.3200981
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.58
Output dim: 2, lower bound: -0.3200976, upper bound: 0.3192831

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5387101, 0.5383698
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4965875, 0.5062066
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4196994, 0.4197425
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3537602, 0.3537632
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3752288, 0.3713155
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3580609, 0.3520110
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5332024, 0.5322173
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3423058, 0.3430493
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3658907, 0.3664393
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5003600, 0.4993393

Time for backsubstitution: 8.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1376

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3192710, upper bound: 0.3200531
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3191356, upper bound: 0.3200864
time: 3.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5389104, 0.5387101
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4993663, 0.4965876
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4197077, 0.4196994
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3537633, 0.3537687
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3765505, 0.3752288
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3520110, 0.3536178
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5343599, 0.5332022
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3430494, 0.3434693
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3664252, 0.3658906
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5006785, 0.5003599

Time for backsubstitution: 8.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 425

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 614

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3192050, upper bound: 0.3191599
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3199745, upper bound: 0.3183898
time: 4.62 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 16.49 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.49
Output dim: 2, lower bound: -0.3192710, upper bound: 0.3200531
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.49
Output dim: 2, lower bound: -0.3191356, upper bound: 0.3200864
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.49
Output dim: 2, lower bound: -0.3192050, upper bound: 0.3191599
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.49
Output dim: 2, lower bound: -0.3199745, upper bound: 0.3183898

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5386946, 0.5383536
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4965754, 0.5062017
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4196912, 0.4197339
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3537532, 0.3537561
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3752272, 0.3713177
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3580623, 0.3520120
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5331929, 0.5322083
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3423036, 0.3430494
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3658864, 0.3664407
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5003571, 0.4993367

Time for backsubstitution: 8.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2606

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3180522, upper bound: 0.3184763
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3178775, upper bound: 0.3185891
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5386938, 0.5383542
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4965827, 0.5061944
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4196910, 0.4197343
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3537532, 0.3537561
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3752311, 0.3713138
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3580616, 0.3520123
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5331933, 0.5322077
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3423058, 0.3430471
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3658920, 0.3664351
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5003571, 0.4993366

Time for backsubstitution: 9.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2363

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 675

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3081722, upper bound: 0.3149325
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3139826, upper bound: 0.3091220
time: 3.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5384751, 0.5384824
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4989398, 0.4963405
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4193055, 0.4193547
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3532321, 0.3532346
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3757512, 0.3742917
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3517410, 0.3533407
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5344143, 0.5332346
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3429908, 0.3434254
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3664699, 0.3657490
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5006838, 0.5003663

Time for backsubstitution: 9.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 1698

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1157

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3154878, upper bound: 0.3101951
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3102403, upper bound: 0.3154426
time: 3.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5386825, 0.5382750
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4991195, 0.4961610
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4193628, 0.4192973
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3532293, 0.3532375
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3756134, 0.3744296
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3517340, 0.3533478
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5343921, 0.5332566
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3430054, 0.3434107
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3662835, 0.3659354
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5006850, 0.5003650

Time for backsubstitution: 9.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 739

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3116336, upper bound: 0.3098328
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3116257, upper bound: 0.3099052
time: 3.34 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 16.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 2, lower bound: -0.3180522, upper bound: 0.3184763
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 2, lower bound: -0.3178775, upper bound: 0.3185891
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 2, lower bound: -0.3081722, upper bound: 0.3149325
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 2, lower bound: -0.3139826, upper bound: 0.3091220
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 2, lower bound: -0.3154878, upper bound: 0.3101951
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 2, lower bound: -0.3102403, upper bound: 0.3154426
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 2, lower bound: -0.3116336, upper bound: 0.3098328
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 2, lower bound: -0.3116257, upper bound: 0.3099052

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5351231, 0.5357223
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4957631, 0.5052933
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4178441, 0.4177076
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3498188, 0.3494220
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3744757, 0.3706075
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3552725, 0.3489980
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5328839, 0.5320196
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3387294, 0.3383998
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3632252, 0.3644271
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4970588, 0.4973285

Time for backsubstitution: 9.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1157

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3143348, upper bound: 0.3095123
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3090873, upper bound: 0.3147595
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5360630, 0.5347824
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4956670, 0.5053894
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4176649, 0.4178867
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3494190, 0.3498218
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3745170, 0.3705662
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3550483, 0.3492227
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5330040, 0.5318995
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3378864, 0.3394753
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3638728, 0.3638713
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4983489, 0.4960384

Time for backsubstitution: 9.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3095599, upper bound: 0.3102678
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3095527, upper bound: 0.3102750
time: 3.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5204071, 0.5254480
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4876654, 0.4932615
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4010273, 0.4047055
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3257559, 0.3325106
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3689103, 0.3614778
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3549556, 0.3499406
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5292125, 0.5286051
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3382074, 0.3377444
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3651715, 0.3658724
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4887388, 0.4828968

Time for backsubstitution: 9.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 614

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1849

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3004309, upper bound: 0.3071890
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3004309, upper bound: 0.3071890
time: 4.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5257878, 0.5200675
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4836497, 0.4972773
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4046623, 0.4010706
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3325077, 0.3257588
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3653952, 0.3649930
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3559898, 0.3489064
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5295908, 0.5282269
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3370032, 0.3389487
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3653292, 0.3657146
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4839175, 0.4877182

Time for backsubstitution: 9.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3056630, upper bound: 0.3007995
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3056556, upper bound: 0.3008068
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5247077, 0.5257100
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4951880, 0.4919381
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4134771, 0.4113321
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3449972, 0.3455149
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3755462, 0.3741663
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3511085, 0.3525509
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5299082, 0.5297797
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3407664, 0.3416691
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3639083, 0.3630350
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5002198, 0.5000337

Time for backsubstitution: 9.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2516

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 172

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3069964, upper bound: 0.3018431
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3069294, upper bound: 0.3018510
time: 4.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5257026, 0.5247152
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4945374, 0.4925888
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4112831, 0.4135261
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3455124, 0.3449997
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3756258, 0.3740867
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3509511, 0.3527083
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5309594, 0.5287286
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3412346, 0.3412010
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3637558, 0.3631874
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5003510, 0.4999026

Time for backsubstitution: 9.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2363

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3082352, upper bound: 0.3142477
time: 4.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3090460, upper bound: 0.3136971
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5271910, 0.5306853
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4866138, 0.4682070
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4065419, 0.4024494
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3399544, 0.3365591
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3752537, 0.3742908
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3476796, 0.3506758
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5235448, 0.5288708
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3357223, 0.3329575
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3473900, 0.3505292
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4962782, 0.4970934

Time for backsubstitution: 9.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 675

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3125

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2939401, upper bound: 0.2921248
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2939401, upper bound: 0.2921248
time: 3.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5386825, 0.5267837
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4711654, 0.4961610
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4193628, 0.4064760
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3532293, 0.3399626
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3756134, 0.3740700
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3490641, 0.3533478
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5343921, 0.5224068
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3325524, 0.3434107
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3662835, 0.3470420
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4974135, 0.5003650

Time for backsubstitution: 9.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1849

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 425

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2944012, upper bound: 0.2935099
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2944012, upper bound: 0.2935099
time: 3.00 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 15.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3143348, upper bound: 0.3095123
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3090873, upper bound: 0.3147595
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3095599, upper bound: 0.3102678
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3095527, upper bound: 0.3102750
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3004309, upper bound: 0.3071890
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3004309, upper bound: 0.3071890
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3056630, upper bound: 0.3007995
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3056556, upper bound: 0.3008068
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3069964, upper bound: 0.3018431
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3069294, upper bound: 0.3018510
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3082352, upper bound: 0.3142477
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.3090460, upper bound: 0.3136971
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.2939401, upper bound: 0.2921248
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.2939401, upper bound: 0.2921248
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.2944012, upper bound: 0.2935099
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.50
Output dim: 2, lower bound: -0.2944012, upper bound: 0.2935099

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5213560, 0.5229498
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4920108, 0.5008907
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4120156, 0.4096851
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3415837, 0.3417019
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3742708, 0.3704820
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3546400, 0.3482081
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5283780, 0.5285649
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3365051, 0.3366437
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3606634, 0.3617129
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4965946, 0.4969954

Time for backsubstitution: 9.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 739

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3117171, upper bound: 0.3088797
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3137034, upper bound: 0.3058381
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5223509, 0.5219550
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4913602, 0.5015413
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4098216, 0.4118791
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3420988, 0.3411868
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3743502, 0.3704025
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3544827, 0.3483655
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5294292, 0.5275137
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3369732, 0.3361755
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3605109, 0.3618653
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4967257, 0.4968643

Time for backsubstitution: 8.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 1849

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 410

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3028318, upper bound: 0.3088466
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3031594, upper bound: 0.3085188
time: 3.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5248795, 0.5274978
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4834964, 0.4777705
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4045854, 0.4006821
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3358338, 0.3328329
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3741643, 0.3704346
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3510612, 0.3466170
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5221863, 0.5275351
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3305858, 0.3290042
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3453252, 0.3488109
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4939491, 0.4927740

Time for backsubstitution: 8.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1698

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1849

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3018148, upper bound: 0.3025219
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3018148, upper bound: 0.3025219
time: 3.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5360630, 0.5235988
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4680481, 0.5053894
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4176649, 0.4048072
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3494190, 0.3362367
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3745170, 0.3702137
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3524426, 0.3492227
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5330040, 0.5210817
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3274152, 0.3394753
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3638728, 0.3453237
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4950844, 0.4960384

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 614

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3051405, upper bound: 0.3080777
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3073413, upper bound: 0.3057907
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5203821, 0.5253030
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4876306, 0.4936945
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4012167, 0.4046382
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3260061, 0.3324846
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3688960, 0.3615685
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3548710, 0.3493014
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5291274, 0.5284300
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3378618, 0.3377029
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3650396, 0.3653563
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4887457, 0.4828402

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2606

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1698

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2775500, upper bound: 0.2855123
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2775500, upper bound: 0.2855123
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5204071, 0.5254228
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4876654, 0.4932265
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4009598, 0.4047055
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3257299, 0.3325106
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3689103, 0.3614634
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3549556, 0.3498559
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5292125, 0.5285201
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3381660, 0.3377444
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3651715, 0.3657405
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4886823, 0.4828968

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 739

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2964146, upper bound: 0.3065568
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2997983, upper bound: 0.3045709
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5146052, 0.5127840
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4714787, 0.4696579
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3915828, 0.3838662
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3189232, 0.3087708
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3650424, 0.3648612
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3519841, 0.3462820
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5187731, 0.5238624
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3297025, 0.3284776
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3467814, 0.3506540
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4794961, 0.4844322

Time for backsubstitution: 8.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 2606

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 60

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3012116, upper bound: 0.2985842
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3034572, upper bound: 0.2964080
time: 4.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5257878, 0.5088849
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4560306, 0.4972773
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4046623, 0.3879913
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3325077, 0.3121742
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3653952, 0.3646404
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3533655, 0.3489064
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5295908, 0.5174091
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3265320, 0.3389487
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3653292, 0.3471668
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4806314, 0.4877182

Time for backsubstitution: 9.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 425

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2363

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3038667, upper bound: 0.2996140
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3044836, upper bound: 0.2988524
time: 3.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5132164, 0.5180250
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4827394, 0.4639840
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4006558, 0.3943858
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3317223, 0.3288363
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3751865, 0.3740275
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3470571, 0.3498788
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5190609, 0.5253949
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3334838, 0.3312160
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3450149, 0.3476560
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4958130, 0.4967622

Time for backsubstitution: 9.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3125

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2892130, upper bound: 0.2839603
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2892130, upper bound: 0.2839603
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5247077, 0.5142187
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4672339, 0.4919381
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4134771, 0.3985109
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3449972, 0.3322401
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3755462, 0.3738067
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3484386, 0.3525509
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5299082, 0.5189300
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3303133, 0.3416691
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3639083, 0.3441415
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4969481, 0.5000337

Time for backsubstitution: 8.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 2363

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2516

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3061613, upper bound: 0.3010848
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3064366, upper bound: 0.3013580
time: 3.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5242248, 0.5214063
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4939754, 0.4918358
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4082745, 0.4111882
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3437407, 0.3431586
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3755794, 0.3740678
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3495188, 0.3500537
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5295787, 0.5251076
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3408738, 0.3409255
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3626213, 0.3615576
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4991765, 0.4993744

Time for backsubstitution: 9.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 675

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1376

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3082246, upper bound: 0.3141239
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3080293, upper bound: 0.3142380
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5223937, 0.5232372
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4937921, 0.4920192
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4089449, 0.4105177
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3436712, 0.3432282
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3756070, 0.3740402
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3482907, 0.3512817
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5273385, 0.5273479
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3409575, 0.3408418
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3621261, 0.3620527
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4998231, 0.4987278

Time for backsubstitution: 9.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2606

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3075874, upper bound: 0.3122966
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074731, upper bound: 0.3124679
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5386613, 0.5267217
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4709175, 0.4961606
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4187145, 0.4061461
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3532286, 0.3392671
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3754009, 0.3737669
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3487912, 0.3528750
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5343921, 0.5223271
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3322608, 0.3434103
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3662813, 0.3463125
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4972650, 0.5003043

Time for backsubstitution: 9.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1849

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3125

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2771189, upper bound: 0.2762474
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2771189, upper bound: 0.2762474
time: 3.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5387483, 0.5266345
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4710405, 0.4960376
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4190329, 0.4058276
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3529081, 0.3395878
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3753103, 0.3738617
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3485912, 0.3530750
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5343494, 0.5223698
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3324264, 0.3432447
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3658419, 0.3467520
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4973527, 0.5002165

Time for backsubstitution: 9.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 1376
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 1157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 900

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2859055, upper bound: 0.2850330
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2859055, upper bound: 0.2850330
time: 3.01 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3117171, upper bound: 0.3088797
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3137034, upper bound: 0.3058381
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3028318, upper bound: 0.3088466
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3031594, upper bound: 0.3085188
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3018148, upper bound: 0.3025219
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3018148, upper bound: 0.3025219
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3051405, upper bound: 0.3080777
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3073413, upper bound: 0.3057907
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2775500, upper bound: 0.2855123
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2775500, upper bound: 0.2855123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2964146, upper bound: 0.3065568
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2997983, upper bound: 0.3045709
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3012116, upper bound: 0.2985842
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3034572, upper bound: 0.2964080
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3038667, upper bound: 0.2996140
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3044836, upper bound: 0.2988524
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2892130, upper bound: 0.2839603
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2892130, upper bound: 0.2839603
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3061613, upper bound: 0.3010848
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3064366, upper bound: 0.3013580
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3082246, upper bound: 0.3141239
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3080293, upper bound: 0.3142380
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3075874, upper bound: 0.3122966
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.3074731, upper bound: 0.3124679
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2771189, upper bound: 0.2762474
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2771189, upper bound: 0.2762474
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2859055, upper bound: 0.2850330
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.56
Output dim: 2, lower bound: -0.2859055, upper bound: 0.2850330

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5209732, 0.5226378
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4919885, 0.5008739
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4084844, 0.4063938
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3395655, 0.3398421
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3704715, 0.3662379
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3541917, 0.3476647
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5280147, 0.5284171
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3346888, 0.3345165
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3603612, 0.3614866
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4957246, 0.4959472

Time for backsubstitution: 9.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 425

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2363

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3099411, upper bound: 0.3077116
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3105394, upper bound: 0.3067186
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5210443, 0.5225668
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4919940, 0.5008684
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4087244, 0.4061537
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3397238, 0.3396838
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3700264, 0.3666829
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3540965, 0.3477556
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5282304, 0.5282014
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3343780, 0.3348273
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3604372, 0.3614107
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4955465, 0.4961255

Time for backsubstitution: 8.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1698

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2916566, upper bound: 0.2837259
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2916566, upper bound: 0.2837259
time: 3.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4835093, 0.4841361
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4348662, 0.4430337
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3758289, 0.3777553
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3220752, 0.3188231
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3722219, 0.3685407
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3333357, 0.3255961
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5338123, 0.5307828
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3183191, 0.3161754
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3597466, 0.3610399
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4784191, 0.4794482

Time for backsubstitution: 9.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 739

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3002144, upper bound: 0.3082141
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3022003, upper bound: 0.3051739
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.4845347, 0.4831134
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4328544, 0.4450473
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.3756977, 0.3778865
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3197353, 0.3211631
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3724884, 0.3682742
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3317118, 0.3272186
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5326984, 0.5318973
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3169731, 0.3175213
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3596855, 0.3611009
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4793096, 0.4785575

Time for backsubstitution: 8.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 172
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2516

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1698

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2813059, upper bound: 0.2865044
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.2813059, upper bound: 0.2865044
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5248542, 0.5273526
1: -11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4834616, 0.4782036
2: 6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4047750, 0.4006149
3: -4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3360841, 0.3328069
4: -12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3741502, 0.3705254
5: -13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3509766, 0.3459779
6: -10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5221014, 0.5273600
7: -1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3302402, 0.3289628
8: -0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3451933, 0.3482948
9: -10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.4939562, 0.4927176

Time for backsubstitution: 8.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1157
type: DSZ, layer: 3, pos: 739
type: DSZ, layer: 3, pos: 60
type: DSZ, layer: 3, pos: 410
type: DSZ, layer: 3, pos: 900
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 425
type: DSZ, layer: 3, pos: 2363
type: DSZ, layer: 3, pos: 614
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 675

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2908546, upper bound: 0.2973699
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2966626, upper bound: 0.2915619
time: 3.49 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 15.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.3099411, upper bound: 0.3077116
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.3105394, upper bound: 0.3067186
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.2916566, upper bound: 0.2837259
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.2916566, upper bound: 0.2837259
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.3002144, upper bound: 0.3082141
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.3022003, upper bound: 0.3051739
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.2813059, upper bound: 0.2865044
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.2813059, upper bound: 0.2865044
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.2908546, upper bound: 0.2973699
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.45
Output dim: 2, lower bound: -0.2966626, upper bound: 0.2915619
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3018148, upper bound: 0.3025219
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3051405, upper bound: 0.3080777
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3073413, upper bound: 0.3057907
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.2964146, upper bound: 0.3065568
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.2997983, upper bound: 0.3045709
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3012116, upper bound: 0.2985842
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3034572, upper bound: 0.2964080
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3038667, upper bound: 0.2996140
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3044836, upper bound: 0.2988524
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3061613, upper bound: 0.3010848
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3064366, upper bound: 0.3013580
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3082246, upper bound: 0.3141239
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3080293, upper bound: 0.3142380
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3075874, upper bound: 0.3122966
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.45
Output dim: 2, lower bound: -0.3074731, upper bound: 0.3124679

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.71 + 546.78 = 606.49 seconds
