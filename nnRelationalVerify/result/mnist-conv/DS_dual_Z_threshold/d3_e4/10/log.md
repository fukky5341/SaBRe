## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0070000638


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4699616, 2.4699612)
1: (-10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0266666, 2.0266664)
2: (-10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3282871, 2.3282874)
3: (-12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9452214, 1.9452219)
4: (5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2497125, 2.2497125)
5: (-8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9607882, 1.9607880)
6: (-12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2138529, 2.2138529)
7: (-6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7246361, 2.7246356)
8: (-3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2275991, 2.2275991)
9: (-5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6726398, 1.6726398)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.34 + 34.16 = 57.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.0090181, upper bound: 1.0090208

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5735

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090104, upper bound: 0.9993711
time: 5.00 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9993716, upper bound: 1.0090098
time: 4.24 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.34 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.34
Output dim: 4, lower bound: -1.0090104, upper bound: 0.9993711
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.34
Output dim: 4, lower bound: -0.9993716, upper bound: 1.0090098

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4899273, 2.4934692
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0377250, 2.0394964
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3301878, 2.3304927
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9555044, 1.9540854
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2282820, 2.2252259
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9648390, 1.9654880
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2081008, 2.2072787
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7381744, 2.7363043
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2263489, 2.2261701
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6428366, 1.6465614

Time for backsubstitution: 22.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4560

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088818, upper bound: 0.9993710
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090102, upper bound: 0.9992429
time: 5.67 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4934692, 2.4899273
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0394964, 2.0377252
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3304930, 2.3301883
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9540854, 1.9555044
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2252264, 2.2282829
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9654884, 1.9648387
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2072792, 2.2081003
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7363033, 2.7381754
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2261705, 2.2263489
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6465611, 1.6428363

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4560

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9992433, upper bound: 1.0090096
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9993712, upper bound: 1.0088845
time: 4.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.09 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.09
Output dim: 4, lower bound: -1.0088818, upper bound: 0.9993710
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.09
Output dim: 4, lower bound: -1.0090102, upper bound: 0.9992429
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.09
Output dim: 4, lower bound: -0.9992433, upper bound: 1.0090096
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.09
Output dim: 4, lower bound: -0.9993712, upper bound: 1.0088845

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4749060, 2.4824171
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0320702, 2.0318089
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3258123, 2.3245423
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9537978, 1.9517546
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2251778, 2.2210140
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9591479, 1.9577439
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2071581, 2.2065978
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7349958, 2.7319541
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2170725, 2.2135553
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6323757, 1.6388640

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 494

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0078844, upper bound: 0.9993725
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088807, upper bound: 0.9983732
time: 4.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4788752, 2.4784474
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0300379, 2.0338411
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3242378, 2.3261166
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9531732, 1.9523787
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2240705, 2.2221193
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9570947, 1.9597969
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2074189, 2.2063365
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7338247, 2.7331247
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2137346, 2.2168941
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6351390, 1.6361008

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 494

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0080125, upper bound: 0.9992446
time: 4.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090091, upper bound: 0.9982476
time: 4.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4784479, 2.4788756
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0338411, 2.0300376
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3261166, 2.3242378
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9523787, 1.9531734
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2221193, 2.2240705
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9597969, 1.9570947
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2063365, 2.2074189
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7331247, 2.7338252
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2168941, 2.2137341
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6361008, 1.6351390

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 494

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982451, upper bound: 1.0090116
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9992421, upper bound: 1.0080116
time: 4.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4824171, 2.4749055
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0318089, 2.0320704
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3245420, 2.3258119
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9517546, 1.9537978
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2210140, 2.2251773
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9577436, 1.9591477
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2065978, 2.2071576
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7319536, 2.7349963
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2135553, 2.2170730
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6388640, 1.6323757

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 494

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9983731, upper bound: 1.0088833
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9993700, upper bound: 1.0078838
time: 4.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.48 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.48
Output dim: 4, lower bound: -1.0078844, upper bound: 0.9993725
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.48
Output dim: 4, lower bound: -1.0088807, upper bound: 0.9983732
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.48
Output dim: 4, lower bound: -1.0080125, upper bound: 0.9992446
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.48
Output dim: 4, lower bound: -1.0090091, upper bound: 0.9982476
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.48
Output dim: 4, lower bound: -0.9982451, upper bound: 1.0090116
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.48
Output dim: 4, lower bound: -0.9992421, upper bound: 1.0080116
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.48
Output dim: 4, lower bound: -0.9983731, upper bound: 1.0088833
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.48
Output dim: 4, lower bound: -0.9993700, upper bound: 1.0078838

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4610176, 2.4713364
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0383906, 2.0394664
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3334808, 2.3312080
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9542317, 1.9522460
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2196679, 2.2161932
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9426899, 1.9389360
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1772447, 2.1804228
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7307882, 2.7282710
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2089319, 2.2064295
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6204348, 1.6296759

Time for backsubstitution: 21.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0078802, upper bound: 0.9980896
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9969791, upper bound: 0.9980927
time: 5.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4638252, 2.4685297
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0397277, 2.0381293
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3324776, 2.3322108
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9542894, 1.9521885
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2203574, 2.2155046
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9403400, 1.9412861
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1809831, 2.1766844
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7313137, 2.7277465
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2099466, 2.2054148
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6231875, 1.6269231

Time for backsubstitution: 21.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088765, upper bound: 0.9970953
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9979753, upper bound: 0.9970983
time: 4.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4649878, 2.4673667
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0363579, 2.0414987
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3319054, 2.3327823
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9536071, 1.9528704
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2185616, 2.2172990
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9406366, 1.9409890
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1775060, 2.1801615
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7296171, 2.7294416
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2055931, 2.2097683
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6231980, 1.6269126

Time for backsubstitution: 21.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0080075, upper bound: 0.9979600
time: 5.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9971116, upper bound: 0.9979644
time: 5.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4677954, 2.4645596
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0376949, 2.0401614
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3309031, 2.3337851
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9536648, 1.9528127
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2192502, 2.2166104
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9382868, 1.9433389
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1812439, 2.1764231
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7301426, 2.7289166
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2066088, 2.2087536
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6259508, 1.6241601

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090041, upper bound: 0.9969637
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9981086, upper bound: 0.9969709
time: 4.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4645596, 2.4677954
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0401611, 2.0376949
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3337841, 2.3309033
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9528127, 1.9536648
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2166104, 2.2192507
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9433393, 1.9382868
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1764231, 2.1812439
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7289171, 2.7301421
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2087536, 2.2066083
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6241598, 1.6259508

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9969684, upper bound: 0.9981079
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9969640, upper bound: 1.0090034
time: 4.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4673662, 2.4649882
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0414987, 2.0363579
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3327827, 2.3319061
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9528704, 1.9536073
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2172990, 2.2185616
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9409890, 1.9406366
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1801615, 2.1775055
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7294416, 2.7296176
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2097683, 2.2055931
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6269126, 1.6231980

Time for backsubstitution: 22.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9979647, upper bound: 0.9971115
time: 6.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9979603, upper bound: 1.0080100
time: 6.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4685297, 2.4638252
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0381293, 2.0397279
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3322105, 2.3324773
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9521885, 1.9542894
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2155051, 2.2203569
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9412861, 1.9403398
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1766844, 2.1809826
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7277460, 2.7313132
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2054148, 2.2099471
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6269231, 1.6231875

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9970958, upper bound: 0.9979747
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9970928, upper bound: 1.0088754
time: 4.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4713364, 2.4610176
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0394664, 2.0383906
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3312073, 2.3334804
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9522457, 1.9542317
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2161937, 2.2196684
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9389362, 1.9426899
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1804228, 2.1772442
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7282715, 2.7307887
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2064295, 2.2089324
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6296759, 1.6204350

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9980928, upper bound: 0.9969782
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9980898, upper bound: 1.0078793
time: 4.34 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.68 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 4, lower bound: -1.0078802, upper bound: 0.9980896
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9969791, upper bound: 0.9980927
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 4, lower bound: -1.0088765, upper bound: 0.9970953
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9979753, upper bound: 0.9970983
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 4, lower bound: -1.0080075, upper bound: 0.9979600
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9971116, upper bound: 0.9979644
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 4, lower bound: -1.0090041, upper bound: 0.9969637
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9981086, upper bound: 0.9969709
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9969684, upper bound: 0.9981079
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9969640, upper bound: 1.0090034
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9979647, upper bound: 0.9971115
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9979603, upper bound: 1.0080100
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9970958, upper bound: 0.9979747
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9970928, upper bound: 1.0088754
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9980928, upper bound: 0.9969782
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.68
Output dim: 4, lower bound: -0.9980898, upper bound: 1.0078793

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4623733, 2.4729919
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0404086, 2.0419300
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3349428, 2.3329940
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9555683, 1.9533405
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2142239, 2.2099719
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9431930, 1.9395502
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1763487, 2.1793985
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7312279, 2.7286310
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2087669, 2.2062402
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6132259, 1.6233683

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0078756, upper bound: 0.9938708
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0036623, upper bound: 0.9980843
time: 4.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4651799, 2.4701848
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0417457, 2.0405927
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3339405, 2.3339968
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9556260, 1.9532831
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2149124, 2.2092834
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9408431, 1.9419003
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1800871, 2.1756601
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7317533, 2.7281065
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2097816, 2.2052255
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6159782, 1.6206155

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088718, upper bound: 0.9928737
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0046585, upper bound: 0.9970872
time: 3.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4663424, 2.4690223
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0383754, 2.0439615
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3333693, 2.3345680
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9549441, 1.9539652
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2131166, 2.2110777
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9411397, 1.9416032
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1766100, 2.1791372
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7300568, 2.7298021
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2054281, 2.2095795
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6159892, 1.6206050

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0080028, upper bound: 0.9937420
time: 4.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0037894, upper bound: 0.9979553
time: 4.32 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 31.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.65
Output dim: 4, lower bound: -1.0078756, upper bound: 0.9938708
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.65
Output dim: 4, lower bound: -1.0036623, upper bound: 0.9980843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.65
Output dim: 4, lower bound: -1.0088718, upper bound: 0.9928737
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.65
Output dim: 4, lower bound: -1.0046585, upper bound: 0.9970872
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.65
Output dim: 4, lower bound: -1.0080028, upper bound: 0.9937420
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.65
Output dim: 4, lower bound: -1.0037894, upper bound: 0.9979553
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.65
Output dim: 4, lower bound: -1.0090041, upper bound: 0.9969637
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.65
Output dim: 4, lower bound: -0.9969640, upper bound: 1.0090034
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.65
Output dim: 4, lower bound: -0.9979603, upper bound: 1.0080100
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.65
Output dim: 4, lower bound: -0.9970928, upper bound: 1.0088754
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.65
Output dim: 4, lower bound: -0.9980898, upper bound: 1.0078793

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.50 + 546.52 = 604.02 seconds
