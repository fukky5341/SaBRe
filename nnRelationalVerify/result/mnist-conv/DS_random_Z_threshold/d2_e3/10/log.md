## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2745331572


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.6024015, 0.6024015)
1: (-7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5952206, 0.5952203)
2: (-2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5117292, 0.5117292)
3: (5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5401940, 0.5401940)
4: (-11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5841854, 0.5841851)
5: (-2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5143909, 0.5143909)
6: (-9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6629021, 0.6629021)
7: (-7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6346526, 0.6346531)
8: (-2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5288428, 0.5288427)
9: (-4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4562324, 0.4562323)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.19 + 34.03 = 58.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2756352, upper bound: 0.2756360

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756345, upper bound: 0.2754312
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2754305, upper bound: 0.2756345
time: 4.55 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.85 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 3, lower bound: -0.2756345, upper bound: 0.2754312
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 3, lower bound: -0.2754305, upper bound: 0.2756345

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.6021318, 0.6019716
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5906477, 0.5922358
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5103335, 0.5108136
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5395820, 0.5392618
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5793190, 0.5810037
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5134273, 0.5137666
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6616356, 0.6609461
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6342816, 0.6344087
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5250968, 0.5263948
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4555064, 0.4557574

Time for backsubstitution: 22.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756342, upper bound: 0.2741062
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2743096, upper bound: 0.2754300
time: 4.08 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.6019716, 0.6021318
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5922358, 0.5906475
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5108137, 0.5103337
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5392618, 0.5395820
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5810037, 0.5793190
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5137668, 0.5134273
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6609464, 0.6616356
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6344085, 0.6342816
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5263947, 0.5250968
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4557575, 0.4555066

Time for backsubstitution: 22.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5842

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2754042, upper bound: 0.2756209
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2754169, upper bound: 0.2756089
time: 3.51 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.37 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.37
Output dim: 3, lower bound: -0.2756342, upper bound: 0.2741062
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.37
Output dim: 3, lower bound: -0.2743096, upper bound: 0.2754300
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.37
Output dim: 3, lower bound: -0.2754042, upper bound: 0.2756209
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.37
Output dim: 3, lower bound: -0.2754169, upper bound: 0.2756089

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.6041567, 0.6036835
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5856049, 0.5885854
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5072269, 0.5083218
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5419104, 0.5412285
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5767164, 0.5795627
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5131595, 0.5133924
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6626499, 0.6621468
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6290808, 0.6300759
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5220771, 0.5241153
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4569681, 0.4580458

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5842

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756074, upper bound: 0.2740925
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756206, upper bound: 0.2740858
time: 3.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.6038435, 0.6039970
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5869973, 0.5871930
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5078416, 0.5077070
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5415487, 0.5415901
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5778778, 0.5784013
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5130532, 0.5134990
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6628361, 0.6619606
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6299486, 0.6292081
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5228171, 0.5233750
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4577949, 0.4572191

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2743090, upper bound: 0.2745824
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2734613, upper bound: 0.2754294
time: 4.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5936615, 0.5952065
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5922356, 0.5908065
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5066974, 0.5074762
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5447145, 0.5438455
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5810027, 0.5798838
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5079116, 0.5063869
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6565320, 0.6583371
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6189322, 0.6213841
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5242950, 0.5235736
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4574853, 0.4575433

Time for backsubstitution: 23.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 942

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2754036, upper bound: 0.2747733
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2745564, upper bound: 0.2756202
time: 5.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5950465, 0.5938215
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5923946, 0.5906472
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5079563, 0.5062176
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5435252, 0.5450349
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5815687, 0.5793178
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5067264, 0.5075719
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6576478, 0.6572213
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6215110, 0.6188052
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5248717, 0.5229969
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4577943, 0.4572343

Time for backsubstitution: 23.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2754163, upper bound: 0.2747605
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2745691, upper bound: 0.2756083
time: 3.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.38 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 3, lower bound: -0.2756074, upper bound: 0.2740925
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 3, lower bound: -0.2756206, upper bound: 0.2740858
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 3, lower bound: -0.2743090, upper bound: 0.2745824
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 3, lower bound: -0.2734613, upper bound: 0.2754294
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 3, lower bound: -0.2754036, upper bound: 0.2747733
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 3, lower bound: -0.2745564, upper bound: 0.2756202
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 3, lower bound: -0.2754163, upper bound: 0.2747605
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.38
Output dim: 3, lower bound: -0.2745691, upper bound: 0.2756083

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5958464, 0.5967581
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5856056, 0.5887442
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5031068, 0.5054597
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5473614, 0.5454905
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5767155, 0.5801270
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5073040, 0.5063521
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6582353, 0.6588480
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6136019, 0.6171753
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5199742, 0.5225891
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4586955, 0.4600816

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756063, upper bound: 0.2727292
time: 4.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742435, upper bound: 0.2740914
time: 3.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5972309, 0.5953732
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5857649, 0.5885861
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5043652, 0.5042015
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5461724, 0.5466799
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5772815, 0.5795615
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5061190, 0.5075371
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6593511, 0.6577322
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6161807, 0.6145968
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5205507, 0.5220125
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4590045, 0.4597733

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756196, upper bound: 0.2727224
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742568, upper bound: 0.2740848
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5991900, 0.6001136
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5847945, 0.5853572
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.4997435, 0.5009549
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5402634, 0.5400501
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5767303, 0.5770044
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5119660, 0.5121964
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6618576, 0.6607862
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6245005, 0.6246655
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5206853, 0.5215977
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4580433, 0.4574081

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2743080, upper bound: 0.2732190
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2729453, upper bound: 0.2745814
time: 4.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5999601, 0.5993435
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5851612, 0.5849900
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5010896, 0.4996086
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5400088, 0.5403047
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5764806, 0.5772541
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5117505, 0.5124118
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6616616, 0.6609819
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6254063, 0.6237597
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5210398, 0.5212432
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4579837, 0.4574676

Time for backsubstitution: 23.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 928

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2734172, upper bound: 0.2745092
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2733949, upper bound: 0.2753767
time: 3.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5890086, 0.5913239
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5900323, 0.5889697
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.4985989, 0.5007240
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5434289, 0.5423055
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5798554, 0.5784869
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5068241, 0.5050839
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6555536, 0.6571627
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6134834, 0.6168411
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5221629, 0.5217961
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4577339, 0.4577323

Time for backsubstitution: 23.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2754029, upper bound: 0.2747698
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2745524, upper bound: 0.2747733
time: 3.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5897791, 0.5905538
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5903990, 0.5886030
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.4999453, 0.4993776
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5431745, 0.5425600
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5796056, 0.5787368
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5066086, 0.5052993
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6553576, 0.6573584
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6143894, 0.6159353
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5225174, 0.5214416
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4576743, 0.4577919

Time for backsubstitution: 23.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 928

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2745136, upper bound: 0.2755636
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2745117, upper bound: 0.2755635
time: 4.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5903938, 0.5899386
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5901911, 0.5888109
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.4998578, 0.4994652
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5422397, 0.5434948
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5804214, 0.5779209
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5056387, 0.5062690
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6566694, 0.6560469
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6160624, 0.6142623
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5227396, 0.5212194
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4580429, 0.4574233

Time for backsubstitution: 23.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2754156, upper bound: 0.2747571
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2745651, upper bound: 0.2747605
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5911639, 0.5891685
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5905583, 0.5884438
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5012039, 0.4981190
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5419850, 0.5437493
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5801721, 0.5781703
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5054232, 0.5064844
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6564734, 0.6562428
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6169682, 0.6133566
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5230942, 0.5208648
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4579833, 0.4574829

Time for backsubstitution: 23.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2745680, upper bound: 0.2742444
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2732056, upper bound: 0.2756065
time: 3.56 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2756063, upper bound: 0.2727292
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2742435, upper bound: 0.2740914
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2756196, upper bound: 0.2727224
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2742568, upper bound: 0.2740848
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2743080, upper bound: 0.2732190
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2729453, upper bound: 0.2745814
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2734172, upper bound: 0.2745092
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2733949, upper bound: 0.2753767
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2754029, upper bound: 0.2747698
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2745524, upper bound: 0.2747733
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2745136, upper bound: 0.2755636
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2745117, upper bound: 0.2755635
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2754156, upper bound: 0.2747571
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2745651, upper bound: 0.2747605
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2745680, upper bound: 0.2742444
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 3, lower bound: -0.2732056, upper bound: 0.2756065

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5896373, 0.5926471
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5771985, 0.5760510
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5029154, 0.5051731
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5446489, 0.5413960
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5743096, 0.5785329
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5049379, 0.5027800
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6562953, 0.6559172
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6069980, 0.6128013
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5183942, 0.5202042
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4531198, 0.4563905

Time for backsubstitution: 23.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756057, upper bound: 0.2718806
time: 4.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2747580, upper bound: 0.2727286
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5910220, 0.5912619
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5773575, 0.5758929
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5041738, 0.5039150
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5434597, 0.5425854
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5748756, 0.5779676
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5037527, 0.5039651
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6574111, 0.6548014
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6095769, 0.6102228
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5189710, 0.5196275
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4534287, 0.4560822

Time for backsubstitution: 23.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756189, upper bound: 0.2718746
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2747712, upper bound: 0.2727212
time: 3.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5950792, 0.5939047
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5721018, 0.5769491
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.4994569, 0.5007634
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5361691, 0.5373374
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5751362, 0.5745983
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5083941, 0.5098302
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6589272, 0.6588461
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6201262, 0.6180618
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5183005, 0.5200176
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4543519, 0.4518319

Time for backsubstitution: 23.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 928

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2729011, upper bound: 0.2745090
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2728789, upper bound: 0.2745282
time: 5.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5980110, 0.5962543
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5827830, 0.5834882
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5010849, 0.5003153
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5385084, 0.5393565
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5745270, 0.5741622
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5100632, 0.5113465
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6575108, 0.6544061
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6254051, 0.6239796
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5142473, 0.5104786
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4562088, 0.4563477

Time for backsubstitution: 25.39 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.22 + 565.07 = 623.29 seconds
