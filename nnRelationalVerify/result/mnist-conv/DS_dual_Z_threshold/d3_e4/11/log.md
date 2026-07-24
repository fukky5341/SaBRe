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
execution time: IAR + RelationalAnalysis = 21.74 + 35.48 = 57.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -1.1688954, upper bound: 1.1688953

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4616
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4616

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1664048, upper bound: 1.1688888
time: 11.49 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688890, upper bound: 1.1664070
time: 6.44 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 18.03 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 18.03
Output dim: 2, lower bound: -1.1664048, upper bound: 1.1688888
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 18.03
Output dim: 2, lower bound: -1.1688890, upper bound: 1.1664070

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7364502, 2.7387762
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2109928, 2.2131789
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2639275, 2.2641964
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9247379, 2.9247446
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9850931, 2.9839573
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.5010138, 2.4999475
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3236780, 2.3225131
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2518144, 3.2520065
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0681386, 2.0681653
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7131419, 2.7142630

Time for backsubstitution: 20.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1664012, upper bound: 1.1609270
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1584428, upper bound: 1.1688854
time: 7.63 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7387762, 2.7364507
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2131786, 2.2109931
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2641964, 2.2639275
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9247446, 2.9247379
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9839573, 2.9850931
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4999475, 2.5010138
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3225136, 2.3236780
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2520061, 3.2518139
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0681653, 2.0681386
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7142625, 2.7131419

Time for backsubstitution: 20.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688854, upper bound: 1.1584448
time: 32.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1609270, upper bound: 1.1664015
time: 6.47 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 59.81 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 59.81
Output dim: 2, lower bound: -1.1664012, upper bound: 1.1609270
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 59.81
Output dim: 2, lower bound: -1.1584428, upper bound: 1.1688854
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 59.81
Output dim: 2, lower bound: -1.1688854, upper bound: 1.1584448
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 59.81
Output dim: 2, lower bound: -1.1609270, upper bound: 1.1664015

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7233887, 2.7238550
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2109261, 2.2134128
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2464342, 2.2488852
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9104190, 2.9083891
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9621816, 2.9638987
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4927416, 2.4905000
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3119936, 2.3091683
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2528296, 3.2546687
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0504684, 2.0479751
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7168274, 2.7190285

Time for backsubstitution: 20.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 498

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1584379, upper bound: 1.1667474
time: 7.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1563051, upper bound: 1.1688805
time: 11.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7238550, 2.7233882
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2134123, 2.2109263
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2488852, 2.2464340
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9083886, 2.9104185
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9638982, 2.9621820
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4905005, 2.4927425
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3091688, 2.3119931
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2546692, 3.2528296
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0479751, 2.0504684
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7190285, 2.7168274

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 498

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688805, upper bound: 1.1563045
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1667475, upper bound: 1.1584378
time: 12.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 39.37 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.37
Output dim: 2, lower bound: -1.1584379, upper bound: 1.1667474
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.37
Output dim: 2, lower bound: -1.1563051, upper bound: 1.1688805
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.37
Output dim: 2, lower bound: -1.1688805, upper bound: 1.1563045
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.37
Output dim: 2, lower bound: -1.1667475, upper bound: 1.1584378

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7226362, 2.7191114
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2104473, 2.2103829
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2463093, 2.2481089
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9098754, 2.9083037
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9554448, 2.9628563
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4817591, 2.4887538
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3077269, 2.3084893
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2515440, 3.2544641
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0498638, 2.0440979
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7166634, 2.7179995

Time for backsubstitution: 23.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1584349, upper bound: 1.1651204
time: 9.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1568126, upper bound: 1.1667442
time: 4.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7186451, 2.7231026
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2078972, 2.2129343
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2456574, 2.2487605
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9103332, 2.9078460
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9611402, 2.9571619
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4909964, 2.4795165
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3113146, 2.3049030
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2526245, 3.2533841
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0465913, 2.0473700
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7157993, 2.7188640

Time for backsubstitution: 23.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1563022, upper bound: 1.1672530
time: 7.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1546797, upper bound: 1.1688764
time: 4.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7231026, 2.7186446
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2129345, 2.2078965
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2487602, 2.2456577
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9078450, 2.9103336
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9571614, 2.9611397
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4795160, 2.4909964
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3049021, 2.3113146
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2533836, 3.2526250
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0473704, 2.0465913
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7188644, 2.7157989

Time for backsubstitution: 23.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688775, upper bound: 1.1546800
time: 11.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672536, upper bound: 1.1563016
time: 4.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7191114, 2.7226362
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2103825, 2.2104478
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2481089, 2.2463093
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9083028, 2.9098754
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9628549, 2.9554453
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4887533, 2.4817586
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3084898, 2.3077283
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2544641, 3.2515450
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0440979, 2.0498638
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7180004, 2.7166634

Time for backsubstitution: 23.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6219
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1667446, upper bound: 1.1568147
time: 15.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1651206, upper bound: 1.1584344
time: 12.91 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 52.18 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 52.18
Output dim: 2, lower bound: -1.1584349, upper bound: 1.1651204
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 52.18
Output dim: 2, lower bound: -1.1568126, upper bound: 1.1667442
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 52.18
Output dim: 2, lower bound: -1.1563022, upper bound: 1.1672530
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 52.18
Output dim: 2, lower bound: -1.1546797, upper bound: 1.1688764
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 52.18
Output dim: 2, lower bound: -1.1688775, upper bound: 1.1546800
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 52.18
Output dim: 2, lower bound: -1.1672536, upper bound: 1.1563016
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 52.18
Output dim: 2, lower bound: -1.1667446, upper bound: 1.1568147
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 52.18
Output dim: 2, lower bound: -1.1651206, upper bound: 1.1584344

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7076721, 2.7020092
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2155137, 2.2144947
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2330155, 2.2376928
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9136953, 2.9114017
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9454260, 2.9514084
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4871273, 2.4931092
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.2833319, 2.2806082
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2558765, 3.2579799
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0282598, 2.0194016
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7176809, 2.7192550

Time for backsubstitution: 23.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1568105, upper bound: 1.1644314
time: 9.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1544974, upper bound: 1.1667422
time: 6.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7015429, 2.7081385
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2120080, 2.2180004
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2352419, 2.2354667
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9134321, 2.9116654
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9496918, 2.9471426
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4953518, 2.4848838
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.2834330, 2.2805071
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2561398, 3.2577167
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0218949, 2.0257664
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7170534, 2.7198820

Time for backsubstitution: 23.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1563002, upper bound: 1.1649369
time: 18.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1539884, upper bound: 1.1672509
time: 5.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7036810, 2.7060003
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2129626, 2.2170460
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2323637, 2.2383442
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9141531, 2.9109440
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9511194, 2.9457145
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4963646, 2.4838715
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.2869177, 2.2770219
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2569561, 3.2568998
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0249872, 2.0226736
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7168169, 2.7201190

Time for backsubstitution: 25.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 6170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1546777, upper bound: 1.1665639
time: 7.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1523645, upper bound: 1.1688751
time: 5.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7060003, 2.7036810
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2170463, 2.2129626
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2383442, 2.2323639
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9109430, 2.9141536
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9457140, 2.9511204
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4838715, 2.4963641
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.2770224, 2.2869186
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2568989, 3.2569575
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0226736, 2.0249872
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7201185, 2.7168169

Time for backsubstitution: 25.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 4639
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 6191
type: DSZ, layer: 1, pos: 4666
type: DSZ, layer: 1, pos: 6111
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 6231
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688751, upper bound: 1.1523643
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1665642, upper bound: 1.1546772
time: 10.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7081385, 2.7015424
1: -6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2179999, 2.2120082
2: 8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2354670, 2.2352417
3: -6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9116659, 2.9134316
4: -11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9471426, 2.9496922
5: -13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.4848843, 2.4953513
6: -15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.2805071, 2.2834334
7: -5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2577152, 3.2561407
8: -1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0257664, 2.0218949
9: -7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7198820, 2.7170539

Time for backsubstitution: 25.63 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.22 + 550.83 = 608.05 seconds
