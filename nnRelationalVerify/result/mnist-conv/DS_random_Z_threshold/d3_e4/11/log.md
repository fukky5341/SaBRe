## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1665576092


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7390966, 2.7390966)
1: (-6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2134962, 2.2134960)
2: (8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2642365, 2.2642365)
3: (-6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9247456, 2.9247451)
4: (-11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9852571, 2.9852571)
5: (-13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.5011797, 2.5011792)
6: (-15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3238053, 2.3238053)
7: (-5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2520328, 3.2520332)
8: (-1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0681491, 2.0681491)
9: (-7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7144065, 2.7144065)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.68 + 35.85 = 59.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -1.1688954, upper bound: 1.1688953

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 520

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 536

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688927, upper bound: 1.1673714
time: 5.66 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673714, upper bound: 1.1688950
time: 6.07 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.74 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.74
Output dim: 2, lower bound: -1.1688927, upper bound: 1.1673714
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.74
Output dim: 2, lower bound: -1.1673714, upper bound: 1.1688950

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7260418, 2.7297492
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2116766, 2.2109587
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2601390, 2.2585118
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9206395, 2.9190102
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9849691, 2.9850478
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4980087, 2.4967480
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3081040, 2.3125625
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2486401, 3.2472916
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0599680, 2.0622911
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7106700, 2.7117310

Time for backsubstitution: 20.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 520

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688910, upper bound: 1.1664370
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1679585, upper bound: 1.1673697
time: 10.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7297497, 2.7260413
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2109585, 2.2116771
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2585115, 2.2601392
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9190106, 2.9206405
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9850473, 2.9849691
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4967480, 2.4980087
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3125615, 2.3081031
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2472916, 3.2486405
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0622911, 2.0599680
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7117314, 2.7106700

Time for backsubstitution: 20.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6191

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673712, upper bound: 1.1684688
time: 5.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1669502, upper bound: 1.1688925
time: 5.05 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.28 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.28
Output dim: 2, lower bound: -1.1688910, upper bound: 1.1664370
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.28
Output dim: 2, lower bound: -1.1679585, upper bound: 1.1673697
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.28
Output dim: 2, lower bound: -1.1673712, upper bound: 1.1684688
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.28
Output dim: 2, lower bound: -1.1669502, upper bound: 1.1688925

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7237062, 2.7278886
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2102842, 2.2107155
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2544870, 2.2520528
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9210844, 2.9191055
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9781971, 2.9773078
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.5004807, 2.4997602
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3062782, 2.3111095
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2493849, 3.2484913
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0587478, 2.0613823
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7100000, 2.7109652

Time for backsubstitution: 21.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 5843

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6231

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688890, upper bound: 1.1653879
time: 24.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1678415, upper bound: 1.1664355
time: 17.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7241812, 2.7274137
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2114334, 2.2095659
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2536802, 2.2528598
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9207354, 2.9194546
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9772291, 2.9782758
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.5010204, 2.4992189
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3066502, 2.3107376
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2498407, 3.2480359
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0590591, 2.0610709
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7099037, 2.7110610

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 929

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6231

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1679565, upper bound: 1.1663210
time: 7.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1669090, upper bound: 1.1673675
time: 8.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7211170, 2.7233706
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.1920719, 2.1951556
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2627420, 2.2600720
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.8491974, 2.8601284
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9733243, 2.9715714
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4981012, 2.4996758
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3107505, 2.3065186
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2410765, 3.2378702
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0579462, 2.0527530
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7175550, 2.7178535

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673686, upper bound: 1.1678601
time: 8.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1667624, upper bound: 1.1684662
time: 7.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7270784, 2.7174091
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.1944370, 2.1927903
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2584443, 2.2643700
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.8584976, 2.8508277
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9716506, 2.9732451
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4984159, 2.4993625
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3109775, 2.3062916
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2365217, 3.2424254
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0550761, 2.0556231
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7189150, 2.7164941

Time for backsubstitution: 21.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6231

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1669481, upper bound: 1.1678453
time: 23.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1659007, upper bound: 1.1688905
time: 9.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 55.13 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 55.13
Output dim: 2, lower bound: -1.1688890, upper bound: 1.1653879
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 55.13
Output dim: 2, lower bound: -1.1678415, upper bound: 1.1664355
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 55.13
Output dim: 2, lower bound: -1.1679565, upper bound: 1.1663210
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 55.13
Output dim: 2, lower bound: -1.1669090, upper bound: 1.1673675
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 55.13
Output dim: 2, lower bound: -1.1673686, upper bound: 1.1678601
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 55.13
Output dim: 2, lower bound: -1.1667624, upper bound: 1.1684662
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 55.13
Output dim: 2, lower bound: -1.1669481, upper bound: 1.1678453
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 55.13
Output dim: 2, lower bound: -1.1659007, upper bound: 1.1688905

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7118130, 2.7174811
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2108898, 2.2112188
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2496939, 2.2465761
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9231896, 2.9218206
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9741082, 2.9721642
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4863386, 2.4834952
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3072796, 2.3123140
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2498245, 3.2493653
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0572200, 2.0588007
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7074394, 2.7078409

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6191

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688888, upper bound: 1.1649664
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1684649, upper bound: 1.1653877
time: 6.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7132988, 2.7159963
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2107868, 2.2113214
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2490106, 2.2472591
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9238000, 2.9212098
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9730535, 2.9732203
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4842157, 2.4856191
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3074837, 2.3121109
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2502594, 3.2489305
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0561662, 2.0598545
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7068758, 2.7084045

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 4666

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1678385, upper bound: 1.1648071
time: 5.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1662145, upper bound: 1.1664316
time: 5.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7122879, 2.7170062
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2120390, 2.2100692
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2488866, 2.2473831
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9228396, 2.9221702
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9731412, 2.9731321
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4868803, 2.4829545
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3076515, 2.3119421
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2502785, 3.2489104
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0575314, 2.0584893
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7073431, 2.7079368

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 498

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6191

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1679564, upper bound: 1.1658989
time: 5.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675324, upper bound: 1.1663198
time: 7.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7137737, 2.7155213
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2119370, 2.2101717
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2482033, 2.2480664
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9234509, 2.9215593
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9720845, 2.9741879
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4847555, 2.4850783
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3078547, 2.3117390
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2507133, 3.2484751
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0564775, 2.0595431
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7067804, 2.7085004

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1650907, upper bound: 1.1654944
time: 5.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1650359, upper bound: 1.1655488
time: 5.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7189398, 2.7203026
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.1914549, 2.1942844
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2611785, 2.2578609
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.8321581, 2.8480825
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9723306, 2.9708738
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4932251, 2.4927750
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3087468, 2.3051014
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2372999, 3.2325268
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0575709, 2.0524883
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7159271, 2.7167053

Time for backsubstitution: 22.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4666

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1654935, upper bound: 1.1678550
time: 7.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673633, upper bound: 1.1659847
time: 4.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7180481, 2.7211938
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.1912003, 2.1945388
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2605309, 2.2585084
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.8371525, 2.8430886
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9726262, 2.9705787
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4912014, 2.4947996
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3093343, 2.3045149
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2357330, 3.2340932
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0576811, 2.0523782
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7164068, 2.7162256

Time for backsubstitution: 23.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6231

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 498

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1667574, upper bound: 1.1663272
time: 11.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1646245, upper bound: 1.1684614
time: 5.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7151871, 2.7070022
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.1950436, 2.1932943
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2536507, 2.2588930
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.8606033, 2.8535442
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9675627, 2.9681020
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4842758, 2.4830990
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3119788, 2.3074970
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2369595, 3.2432990
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0535483, 2.0530410
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7163544, 2.7133694

Time for backsubstitution: 22.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1669474, upper bound: 1.1665332
time: 8.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1656292, upper bound: 1.1678427
time: 8.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7166710, 2.7055168
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.1949415, 2.1933970
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2529678, 2.2595763
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.8612146, 2.8529329
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9665070, 2.9691582
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4821510, 2.4852228
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3121829, 2.3072934
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2373943, 3.2428637
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0524940, 2.0540948
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7157898, 2.7139330

Time for backsubstitution: 22.57 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.53 + 545.49 = 605.02 seconds
