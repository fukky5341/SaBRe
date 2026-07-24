## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.46374671450000005


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7471986, 0.7471986)
1: (-8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9235189, 0.9235187)
2: (-2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7697604, 0.7697604)
3: (-10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8960600, 0.8960595)
4: (-8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8202255, 0.8202257)
5: (-5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6728578, 0.6728578)
6: (-1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8079462, 0.8079464)
7: (-8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.9045877, 0.9045880)
8: (-1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7171779, 0.7171779)
9: (-6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.8136141, 0.8136144)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.57 + 32.94 = 56.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.4660768, upper bound: 0.4660770

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6126

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4660691, upper bound: 0.4606142
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4606140, upper bound: 0.4660693
time: 2.79 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.80
Output dim: 0, lower bound: -0.4660691, upper bound: 0.4606142
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.80
Output dim: 0, lower bound: -0.4606140, upper bound: 0.4660693

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7355590, 0.7316821
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8596187, 0.8755634
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7687922, 0.7684705
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8451262, 0.8281772
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7550886, 0.7713597
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6735132, 0.6688547
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7919145, 0.7984667
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.9048138, 0.9047718
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7160511, 0.7187610
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7546301, 0.7350062

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4590630, upper bound: 0.4603530
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4658080, upper bound: 0.4536080
time: 2.93 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7316821, 0.7355590
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8755634, 0.8596187
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7684705, 0.7687922
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8281765, 0.8451259
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7713594, 0.7550886
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6688545, 0.6735134
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7984667, 0.7919147
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.9047718, 0.9048135
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7187614, 0.7160506
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7350063, 0.7546302

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4658081
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4590631
time: 2.96 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.47 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 27.47
Output dim: 0, lower bound: -0.4590630, upper bound: 0.4603530
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.47
Output dim: 0, lower bound: -0.4658080, upper bound: 0.4536080
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.47
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4658081
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 27.47
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4590631

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7289927, 0.7228951
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8527918, 0.8704414
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7612095, 0.7627854
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8457694, 0.8296723
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7528696, 0.7696955
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6688960, 0.6627007
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7920957, 0.7982843
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8973222, 0.8991568
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7152882, 0.7181890
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7547541, 0.7353501

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 136

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4651423, upper bound: 0.4534815
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4656816, upper bound: 0.4529423
time: 2.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7228951, 0.7289927
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8704414, 0.8527918
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7627854, 0.7612095
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8296723, 0.8457694
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7696955, 0.7528696
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6627004, 0.6688960
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7982845, 0.7920957
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8991570, 0.8973224
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7181888, 0.7152884
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7353501, 0.7547541

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4525781, upper bound: 0.4657909
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4535906, upper bound: 0.4647784
time: 2.90 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.93 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 0, lower bound: -0.4651423, upper bound: 0.4534815
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 0, lower bound: -0.4656816, upper bound: 0.4529423
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 0, lower bound: -0.4525781, upper bound: 0.4657909
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 0, lower bound: -0.4535906, upper bound: 0.4647784

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7289925, 0.7228951
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8527911, 0.8704388
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7612097, 0.7627854
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8457689, 0.8296719
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7528684, 0.7696912
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6688960, 0.6627004
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7920942, 0.7982833
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8973234, 0.8991568
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7152882, 0.7181888
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7547522, 0.7353487

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 495

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 541

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4648449, upper bound: 0.4534806
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4648457, upper bound: 0.4526459
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7289929, 0.7228947
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8527892, 0.8704410
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7612095, 0.7627857
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8457689, 0.8296723
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7528656, 0.7696946
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6688960, 0.6627004
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7920942, 0.7982833
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8973224, 0.8991578
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7152882, 0.7181888
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7547526, 0.7353482

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4656723, upper bound: 0.4407673
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4535066, upper bound: 0.4529331
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7219682, 0.7327774
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8736989, 0.8519938
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7646475, 0.7607539
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8287477, 0.8495433
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7705905, 0.7526503
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6640973, 0.6685541
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8017969, 0.7912362
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8983521, 0.9006176
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7188025, 0.7151380
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7363205, 0.7545167

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 136

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4519123, upper bound: 0.4656646
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524517, upper bound: 0.4651253
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7228951, 0.7280660
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8696434, 0.8527918
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7623301, 0.7612095
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8296723, 0.8448446
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7694762, 0.7528696
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6623588, 0.6688960
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7974253, 0.7920957
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8991570, 0.8965175
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7180386, 0.7152884
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7351129, 0.7547541

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 136

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4529248, upper bound: 0.4646522
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4534641, upper bound: 0.4641129
time: 2.76 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.74 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 0, lower bound: -0.4648449, upper bound: 0.4534806
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 0, lower bound: -0.4648457, upper bound: 0.4526459
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 0, lower bound: -0.4656723, upper bound: 0.4407673
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 27.74
Output dim: 0, lower bound: -0.4535066, upper bound: 0.4529331
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 0, lower bound: -0.4519123, upper bound: 0.4656646
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 0, lower bound: -0.4524517, upper bound: 0.4651253
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 0, lower bound: -0.4529248, upper bound: 0.4646522
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 0, lower bound: -0.4534641, upper bound: 0.4641129

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7278099, 0.7233679
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8498807, 0.8665140
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7618799, 0.7639785
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8461571, 0.8291695
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7527702, 0.7689152
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6658320, 0.6586001
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7940722, 0.8009441
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8901622, 0.8937836
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7117476, 0.7134686
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7378230, 0.7226459

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4638152, upper bound: 0.4534641
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4648276, upper bound: 0.4524516
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7294652, 0.7217128
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8488662, 0.8675287
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7624018, 0.7634556
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8452663, 0.8300593
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7520924, 0.7695928
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6647954, 0.6596365
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7947545, 0.8002610
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8919485, 0.8919957
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7105680, 0.7146480
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7420485, 0.7184197

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 495

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4648364, upper bound: 0.4404709
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4526706, upper bound: 0.4526366
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7346318, 0.7269926
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8463097, 0.8614256
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7565207, 0.7565393
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8392367, 0.8248284
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7539237, 0.7691278
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6607904, 0.6566136
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7873182, 0.7947004
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8964896, 0.8980477
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7162156, 0.7193575
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7575166, 0.7373857

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4646426, upper bound: 0.4407501
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4656551, upper bound: 0.4397376
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7219677, 0.7327771
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8736985, 0.8519912
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7646475, 0.7607539
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8287477, 0.8495429
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7705896, 0.7526460
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6640968, 0.6685541
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8017960, 0.7912350
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8983529, 0.9006176
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7188020, 0.7151377
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7363186, 0.7545154

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4519031, upper bound: 0.4534894
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4397373, upper bound: 0.4656552
time: 2.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7219682, 0.7327766
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8736963, 0.8519931
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7646472, 0.7607541
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8287477, 0.8495429
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7705863, 0.7526493
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6640968, 0.6685541
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8017960, 0.7912347
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8983519, 0.9006186
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7188020, 0.7151377
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7363191, 0.7545149

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 541

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4516159, upper bound: 0.4648287
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524507, upper bound: 0.4648279
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7228947, 0.7280660
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8696430, 0.8527892
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7623301, 0.7612095
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8296719, 0.8448446
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7694752, 0.7528656
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6623583, 0.6688957
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7974234, 0.7920947
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8991573, 0.8965178
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7180386, 0.7152882
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7351110, 0.7547526

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4529155, upper bound: 0.4524770
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4407497, upper bound: 0.4646428
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7228951, 0.7280655
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8696408, 0.8527911
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7623298, 0.7612097
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8296719, 0.8448446
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7694719, 0.7528687
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6623583, 0.6688960
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7974238, 0.7920945
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8991568, 0.8965185
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7180386, 0.7152882
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7351115, 0.7547522

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 541

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4526284, upper bound: 0.4638162
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4534631, upper bound: 0.4638155
time: 2.82 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.05 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4638152, upper bound: 0.4534641
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4648276, upper bound: 0.4524516
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4648364, upper bound: 0.4404709
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4526706, upper bound: 0.4526366
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4646426, upper bound: 0.4407501
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4656551, upper bound: 0.4397376
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4519031, upper bound: 0.4534894
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4397373, upper bound: 0.4656552
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4516159, upper bound: 0.4648287
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4524507, upper bound: 0.4648279
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4529155, upper bound: 0.4524770
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4407497, upper bound: 0.4646428
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4526284, upper bound: 0.4638162
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.05
Output dim: 0, lower bound: -0.4534631, upper bound: 0.4638155

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7268836, 0.7271531
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8531382, 0.8657160
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7637417, 0.7635231
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8452330, 0.8329432
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7536650, 0.7686956
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6672287, 0.6582584
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7975850, 0.8000846
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8893578, 0.8970795
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7123618, 0.7133191
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7387934, 0.7224087

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4638059, upper bound: 0.4412890
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4516402, upper bound: 0.4534541
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7278099, 0.7224419
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8490827, 0.8665140
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7614243, 0.7639785
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8461571, 0.8282449
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7525506, 0.7689152
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6654902, 0.6586001
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7932124, 0.8009441
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8901622, 0.8929794
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7115979, 0.7134686
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7375858, 0.7226459

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4648184, upper bound: 0.4402759
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4526526, upper bound: 0.4524417
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7351043, 0.7258110
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8423862, 0.8585129
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7577136, 0.7572103
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8387337, 0.8252151
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7531505, 0.7690263
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6566916, 0.6535504
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7899776, 0.7966781
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8911157, 0.8908856
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7114959, 0.7158172
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7448120, 0.7204567

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4638067, upper bound: 0.4404535
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4648192, upper bound: 0.4394412
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7337050, 0.7307775
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8495674, 0.8606279
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7583823, 0.7560835
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8383121, 0.8286018
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7548187, 0.7689083
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6621876, 0.6562722
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7908297, 0.7938402
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8956852, 0.9013433
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7168288, 0.7192070
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7584870, 0.7371485

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 541

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 541

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4638069, upper bound: 0.4404535
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4646416, upper bound: 0.4404527
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7346318, 0.7260661
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8455119, 0.8614256
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7560649, 0.7565393
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8392367, 0.8239033
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7537043, 0.7691278
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6604495, 0.6566136
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7864580, 0.7947004
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8964896, 0.8972430
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7160649, 0.7193575
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7572794, 0.7373857

Time for backsubstitution: 22.72 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.51 + 556.27 = 612.78 seconds
