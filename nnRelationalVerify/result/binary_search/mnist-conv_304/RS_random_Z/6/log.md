## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.1823463684
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.8861728, 3.8861723)
1: (-10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.7987485, 2.7987485)
2: (-6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.8586493, 2.8586493)
3: (-2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.5133810, 2.5133805)
4: (-6.9938774, -2.8966291, -6.9938774, -2.8966291, -4.0211811, 4.0211816)
5: (-8.9602108, -5.7368851, -8.9602108, -5.7368851, -3.2127132, 3.2127137)
6: (-19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.8937092, 3.8937092)
7: (4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786)
8: (-7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.7680058, 2.7680058)
9: (-7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.4328909, 3.4328909)

## BASE Result
execution time: IAR + LP analysis = 15.45 + 33.09 = 48.54 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.46 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.7229785919189453
rel_dist={7: [-1.52484302938982, 1.5248425766176732]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.7229785919189453
rel_dist={7: [-1.184716116461333, 1.184715436374983]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.7229785919189453
rel_dist={7: [-0.9113997367826396, 0.9113981821144339]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.7229785919189453
rel_dist={7: [-1.0527576805321806, 1.0527567151637864]}

## Binary Search Result
Binary search time: 207.91 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3343.55 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265306, upper bound: 1.6217980
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6217981, upper bound: 1.6265307
time: 4.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.31
Output dim: 7, lower bound: -1.6265306, upper bound: 1.6217980
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.31
Output dim: 7, lower bound: -1.6217981, upper bound: 1.6265307

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924773, 3.0760870
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4975448, 2.4887409
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5852017, 2.6031280
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1076417, 2.1179776
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5288448, 3.5224857
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7569017, 2.7659807
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6694450, 3.6598468
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6881680, 2.6846516
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0400882, 3.0309520

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265306, upper bound: 1.6137109
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6184428, upper bound: 1.6217979
time: 4.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0760875, 3.0924778
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4887404, 2.4975448
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6031280, 2.5852013
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1179776, 2.1076417
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5224857, 3.5288448
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7659807, 2.7569017
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6598463, 3.6694450
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6846519, 2.6881683
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0309520, 3.0400882

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6217799, upper bound: 1.6166356
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6119042, upper bound: 1.6265121
time: 4.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.95
Output dim: 7, lower bound: -1.6265306, upper bound: 1.6137109
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.95
Output dim: 7, lower bound: -1.6184428, upper bound: 1.6217979
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.95
Output dim: 7, lower bound: -1.6217799, upper bound: 1.6166356
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.95
Output dim: 7, lower bound: -1.6119042, upper bound: 1.6265121

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924692, 3.0760810
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4975371, 2.4887347
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5851979, 2.6031237
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1076241, 2.1179650
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5288448, 3.5224857
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7568865, 2.7659698
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6694336, 3.6598363
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6881585, 2.6846387
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0400839, 3.0309486

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6197536, upper bound: 1.6060426
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6188714, upper bound: 1.6069150
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924721, 3.0760789
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4975381, 2.4887328
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5851979, 2.6031246
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1076269, 2.1179605
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5288448, 3.5224857
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7568884, 2.7659650
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6694355, 3.6598334
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6881557, 2.6846421
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0400848, 3.0309477

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6184411, upper bound: 1.6136859
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6184244, upper bound: 1.6217958
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0749354, 3.0925355
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4889212, 2.4938893
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5968537, 2.5855236
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1167269, 2.1077023
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5232325, 3.5138559
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7658901, 2.7569175
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6576204, 3.6695690
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6808710, 2.6883574
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0310450, 3.0383186

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6167300, upper bound: 1.6152785
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6203516, upper bound: 1.6115995
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0760875, 3.0913265
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4850855, 2.4975448
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6031280, 2.5789270
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1179776, 2.1063910
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5074959, 3.5288448
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7659807, 2.7568116
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6598463, 3.6672192
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6846519, 2.6843874
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0291824, 3.0400882

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6119044, upper bound: 1.6184246
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6038217, upper bound: 1.6265118
time: 6.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.6197536, upper bound: 1.6060426
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.6188714, upper bound: 1.6069150
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.6184411, upper bound: 1.6136859
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.6184244, upper bound: 1.6217958
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.6167300, upper bound: 1.6152785
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.6203516, upper bound: 1.6115995
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.6119044, upper bound: 1.6184246
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.6038217, upper bound: 1.6265118

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924683, 3.0760841
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4975367, 2.4887376
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5851979, 2.6031251
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1076241, 2.1179647
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5288401, 3.5224829
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7568855, 2.7659698
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6694336, 3.6598363
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6881595, 2.6846387
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0400839, 3.0309486

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6197222, upper bound: 1.6038178
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6166358, upper bound: 1.6039923
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924692, 3.0760798
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4975371, 2.4887352
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5851979, 2.6031232
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1076241, 2.1179650
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5288420, 3.5224857
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7568865, 2.7659683
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6694336, 3.6598363
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6881585, 2.6846387
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0400839, 3.0309486

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6180355, upper bound: 1.6069004
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6188566, upper bound: 1.6060579
time: 5.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0868163, 3.0632923
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.5005322, 2.4910016
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5691385, 2.5917382
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0590491, 2.0835516
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5226440, 3.5137415
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7065415, 2.7302961
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6519384, 3.6351418
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6721964, 2.6596854
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0358076, 3.0249090

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6184229, upper bound: 1.6037947
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085468, upper bound: 1.6136659
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0796847, 3.0704243
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4998064, 2.4917293
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5738087, 2.5870652
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0732207, 2.0693824
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5201006, 3.5162840
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7212224, 2.7156167
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6447439, 3.6423383
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6631985, 2.6686835
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0340462, 3.0266705

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6133136, upper bound: 1.6203664
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6169282, upper bound: 1.6167458
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0782633, 3.0923991
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4873962, 2.4902058
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5953374, 2.5848923
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1164193, 2.1075709
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5211449, 3.5088339
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7658758, 2.7571840
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6566868, 3.6691723
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6789565, 2.6836820
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0282583, 3.0314865

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6167282, upper bound: 1.6070845
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085356, upper bound: 1.6152768
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0747995, 3.0958624
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4852381, 2.4923639
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5962224, 2.5840068
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1165953, 2.1073952
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5182114, 3.5117679
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7661572, 2.7569032
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6572237, 3.6686349
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6761951, 2.6864431
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0242128, 3.0355320

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6203516, upper bound: 1.6034616
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6121775, upper bound: 1.6116017
time: 5.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0760794, 3.0913205
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4850779, 2.4975376
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6031251, 2.5789237
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1179605, 2.1063762
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5074959, 3.5288448
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7659655, 2.7567992
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6598339, 3.6672072
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6846423, 2.6843750
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0291786, 3.0400848

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6119026, upper bound: 1.6184060
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6037948, upper bound: 1.6184229
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0760813, 3.0913186
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4850798, 2.4975367
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6031241, 2.5789247
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1179647, 2.1063738
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5074959, 3.5288448
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7659693, 2.7567959
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6598358, 3.6672063
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6846385, 2.6843784
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0291805, 3.0400839

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6038199, upper bound: 1.6184296
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6037819, upper bound: 1.6265107
time: 5.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6197222, upper bound: 1.6038178
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6166358, upper bound: 1.6039923
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6180355, upper bound: 1.6069004
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6188566, upper bound: 1.6060579
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6184229, upper bound: 1.6037947
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6085468, upper bound: 1.6136659
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6133136, upper bound: 1.6203664
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6169282, upper bound: 1.6167458
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6167282, upper bound: 1.6070845
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6085356, upper bound: 1.6152768
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6203516, upper bound: 1.6034616
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6121775, upper bound: 1.6116017
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6119026, upper bound: 1.6184060
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6037948, upper bound: 1.6184229
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6038199, upper bound: 1.6184296
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 7, lower bound: -1.6037819, upper bound: 1.6265107

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0913172, 3.0761414
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4977188, 2.4850836
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5789237, 2.6034479
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1063733, 2.1180260
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5295935, 3.5074944
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7567959, 2.7659869
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6672058, 3.6599588
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6843791, 2.6848283
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0401783, 3.0291805

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6188806, upper bound: 1.6038032
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6197076, upper bound: 1.6029619
time: 5.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924683, 3.0749323
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4938831, 2.4887376
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5851979, 2.5968518
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1076241, 2.1167138
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5138521, 3.5224829
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7568855, 2.7658806
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6694336, 3.6576090
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6881595, 2.6808584
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0383158, 3.0309486

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6115998, upper bound: 1.6024959
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6152788, upper bound: 1.5988811
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0897198, 3.0804882
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4966373, 2.4901772
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5869522, 2.6020336
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1127939, 2.1147478
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5277262, 3.5242710
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7556200, 2.7680125
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6711102, 3.6588073
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6907320, 2.6830356
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0378046, 3.0346498

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6131194, upper bound: 1.6068979
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6180329, upper bound: 1.6019957
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924692, 3.0733294
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4975371, 2.4878364
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5841093, 2.6031232
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1044073, 2.1179650
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5288420, 3.5213695
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7568865, 2.7647033
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6684055, 3.6598363
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6865554, 2.6846387
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0400839, 3.0286689

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6167916, upper bound: 1.6029656
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6166170, upper bound: 1.6060257
time: 5.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0856643, 3.0633502
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.5007124, 2.4873452
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5628633, 2.5920591
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0577974, 2.0836122
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5233908, 3.4987526
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7064519, 2.7303128
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6497107, 3.6352639
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6684151, 2.6598740
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0359011, 3.0231395

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6133122, upper bound: 1.6022989
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6169286, upper bound: 1.5986839
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0868163, 3.0621412
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4968758, 2.4910016
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5691385, 2.5854630
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0590491, 2.0823007
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5076542, 3.5137415
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7065415, 2.7302065
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6519384, 3.6329141
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6721964, 2.6559041
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0340376, 3.0249090

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085467, upper bound: 1.6039667
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085429, upper bound: 1.6068571
time: 6.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0830116, 3.0702877
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4982800, 2.4880452
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5722923, 2.5864348
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0729136, 2.0692511
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5180130, 3.5112619
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7212071, 2.7158823
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6438074, 3.6419406
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6612830, 2.6640067
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0312576, 3.0198369

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6116020, upper bound: 1.6203644
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6117590, upper bound: 1.6150173
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0795479, 3.0737510
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4961219, 2.4902034
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5731783, 2.5855494
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0730891, 2.0690753
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5150795, 3.5141954
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7214894, 2.7156010
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6443453, 3.6414032
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6585217, 2.6667676
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0272131, 3.0238819

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6169119, upper bound: 1.6068597
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6070612, upper bound: 1.6167280
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0725923, 3.0796008
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4903903, 2.4924726
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5792780, 2.5735016
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0678406, 2.0731604
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5149422, 3.5000892
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7155280, 2.7215166
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6391821, 3.6444769
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6629753, 2.6587029
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0239792, 3.0254483

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6099434, upper bound: 1.6070817
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6070312, upper bound: 1.6070868
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0654645, 3.0867314
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4896617, 2.4932013
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5839510, 2.5688334
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0820093, 2.0589914
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5123997, 3.5026321
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7302070, 2.7068367
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6319914, 3.6516724
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6539774, 2.6677008
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0222206, 3.0272093

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6017401, upper bound: 1.6152725
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5988541, upper bound: 1.6152768
time: 5.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0747910, 3.0958560
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4852300, 2.4923573
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5962195, 2.5840030
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1165776, 2.1073799
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5182114, 3.5117679
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7661409, 2.7568903
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6572094, 3.6686230
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6761861, 2.6864307
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0242095, 3.0355291

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6203498, upper bound: 1.6034434
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6121503, upper bound: 1.6034600
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0747929, 3.0958538
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4852319, 2.4923563
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5962186, 2.5840039
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1165824, 2.1073775
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5182114, 3.5117679
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7661457, 2.7568874
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6572123, 3.6686220
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6761832, 2.6864340
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0242105, 3.0355282

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5978455, upper bound: 1.6115849
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6121625, upper bound: 1.6107761
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0704246, 3.0785341
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4880733, 2.4998064
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5870657, 2.5675335
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0693822, 2.0719695
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5012941, 3.5201011
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7156167, 2.7211332
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6423388, 3.6425152
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6686835, 2.6594174
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0249014, 3.0340462

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6110542, upper bound: 1.6183921
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6118867, upper bound: 1.6175163
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0632930, 3.0856645
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4873457, 2.5005326
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5917377, 2.5628629
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0835519, 2.0577979
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.4987526, 3.5226440
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7302957, 2.7064514
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6351423, 3.6497111
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6596851, 2.6684153
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0231400, 3.0358071

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5989120, upper bound: 1.6184223
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6037923, upper bound: 1.6135142
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0704226, 3.0785317
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4880714, 2.4998055
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5870647, 2.5675325
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0693870, 2.0719647
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5012941, 3.5201011
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7156205, 2.7211285
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6423349, 3.6425142
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6686802, 2.6594207
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0249004, 3.0340452

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6038198, upper bound: 1.6087244
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6038160, upper bound: 1.6116288
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0632949, 3.0856667
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4873476, 2.5005345
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5917397, 2.5628638
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0835567, 2.0577955
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.4987526, 3.5226440
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7303014, 2.7064486
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6351442, 3.6497140
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6596823, 2.6684186
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0231409, 3.0358081

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6037818, upper bound: 1.6168042
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6037781, upper bound: 1.6197200
time: 4.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6188806, upper bound: 1.6038032
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6197076, upper bound: 1.6029619
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6115998, upper bound: 1.6024959
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6152788, upper bound: 1.5988811
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6131194, upper bound: 1.6068979
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6180329, upper bound: 1.6019957
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6167916, upper bound: 1.6029656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6166170, upper bound: 1.6060257
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6133122, upper bound: 1.6022989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6169286, upper bound: 1.5986839
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6085467, upper bound: 1.6039667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6085429, upper bound: 1.6068571
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6116020, upper bound: 1.6203644
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6117590, upper bound: 1.6150173
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6169119, upper bound: 1.6068597
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6070612, upper bound: 1.6167280
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6099434, upper bound: 1.6070817
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6070312, upper bound: 1.6070868
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6017401, upper bound: 1.6152725
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.5988541, upper bound: 1.6152768
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6203498, upper bound: 1.6034434
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6121503, upper bound: 1.6034600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.5978455, upper bound: 1.6115849
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6121625, upper bound: 1.6107761
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6110542, upper bound: 1.6183921
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6118867, upper bound: 1.6175163
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.5989120, upper bound: 1.6184223
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6037923, upper bound: 1.6135142
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6038198, upper bound: 1.6087244
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6038160, upper bound: 1.6116288
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6037818, upper bound: 1.6168042
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 7, lower bound: -1.6037781, upper bound: 1.6197200
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.7229785919189453
rel_dist={7: [-1.6265335290613763, 1.626531610562398]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3105138, upper bound: 1.3058958
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3058989, upper bound: 1.3105153
time: 5.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.27
Output dim: 7, lower bound: -1.3105138, upper bound: 1.3058958
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.27
Output dim: 7, lower bound: -1.3058989, upper bound: 1.3105153

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6713529, 2.6672802
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3257408, 2.3253241
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.4239807, 2.4266481
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8699694, 1.8780663
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2477980, 3.2463455
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4685812, 2.4769702
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2974596, 3.2933512
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4456329, 2.4404914
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7728672, 2.7718625

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3077368, upper bound: 1.3021290
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3021300, upper bound: 1.3031204
time: 5.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6672807, 2.6713533
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3253241, 2.3257403
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.4266481, 2.4239802
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8780661, 1.8699696
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2463455, 3.2477984
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4769707, 2.4685812
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2933512, 3.2974596
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4404912, 2.4456332
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7718620, 2.7728677

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3058956, upper bound: 1.3077971
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3031766, upper bound: 1.3105139
time: 5.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.32
Output dim: 7, lower bound: -1.3077368, upper bound: 1.3021290
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.32
Output dim: 7, lower bound: -1.3021300, upper bound: 1.3031204
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.32
Output dim: 7, lower bound: -1.3058956, upper bound: 1.3077971
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.32
Output dim: 7, lower bound: -1.3031766, upper bound: 1.3105139

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6713519, 2.6672816
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3257408, 2.3253260
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.4239788, 2.4266481
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8699694, 1.8780661
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2477961, 3.2463450
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4685812, 2.4769707
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2974586, 3.2933502
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4456325, 2.4404905
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7728677, 2.7718625

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3048843, upper bound: 1.3021275
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3077353, upper bound: 1.2992681
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6713529, 2.6672792
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3257408, 2.3253241
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.4239807, 2.4266467
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8699694, 1.8780663
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2477980, 3.2463455
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4685812, 2.4769702
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2974596, 3.2933512
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4456320, 2.4404914
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7728677, 2.7718625

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3067489, upper bound: 1.3003992
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3040340, upper bound: 1.3031205
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6203737, 2.6150806
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3002086, 2.2955933
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3671899, 2.3747654
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8440938, 1.8419034
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2300892, 3.2279086
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4547715, 2.4515705
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2658501, 3.2644739
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4380369, 2.4411693
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7508168, 2.7466016

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3058886, upper bound: 1.3017122
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2970893, upper bound: 1.3077913
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6110077, 2.6244483
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2951770, 2.3006248
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3774357, 2.3645220
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8500004, 1.8359971
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2264557, 3.2315431
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4599586, 2.4463825
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2603655, 3.2699614
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4360275, 2.4431787
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7455959, 2.7518229

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3003996, upper bound: 1.3067477
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2994074, upper bound: 1.3077348
time: 5.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -1.3048843, upper bound: 1.3021275
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -1.3077353, upper bound: 1.2992681
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -1.3067489, upper bound: 1.3003992
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -1.3040340, upper bound: 1.3031205
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -1.3058886, upper bound: 1.3017122
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -1.2970893, upper bound: 1.3077913
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -1.3003996, upper bound: 1.3067477
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 7, lower bound: -1.2994074, upper bound: 1.3077348

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6699433, 2.6661046
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2776885, 2.2676535
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3296447, 2.3480535
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8491306, 1.8606997
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2087545, 3.1994896
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4568634, 2.4622087
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2790394, 3.2833114
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4001145, 2.3800485
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7783360, 2.7660980

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3048832, upper bound: 1.2994049
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3021659, upper bound: 1.3021247
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6701751, 2.6658726
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2680678, 2.2772732
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3453846, 2.3323135
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8526030, 1.8572273
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2009411, 3.2073030
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4538193, 2.4652534
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2874203, 3.2749310
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3851905, 2.3949726
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7671037, 2.7773299

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 6209

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3077353, upper bound: 1.2992534
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3031110, upper bound: 1.2992661
time: 6.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6244478, 2.6110063
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3006244, 2.2951775
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3645220, 2.3774347
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8359971, 1.8499999
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2315426, 3.2264557
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4463830, 2.4599581
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2699614, 3.2603655
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4431791, 2.4360275
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7518234, 2.7455964

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3067489, upper bound: 1.3003912
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3021182, upper bound: 1.3003986
time: 5.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6150799, 2.6203725
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2955937, 2.3002086
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3747654, 2.3671885
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8419037, 1.8440936
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2279081, 3.2300897
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4515710, 2.4547706
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2644739, 3.2658501
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4411697, 2.4380369
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7466016, 2.7508168

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3008894, upper bound: 1.3018561
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3027761, upper bound: 1.2999715
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6192226, 2.6146202
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2987442, 2.2919374
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3609147, 2.3722596
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8428426, 1.8414018
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2240925, 3.2129192
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4546809, 2.4515409
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2636232, 3.2635894
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4342570, 2.4396577
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7501125, 2.7448330

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2942293, upper bound: 1.3017113
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2970878, upper bound: 1.2988554
time: 6.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6199131, 2.6139293
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2965527, 2.2941294
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3646836, 2.3684902
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8435922, 1.8406525
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2151003, 3.2219114
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4547420, 2.4514804
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2649660, 3.2622466
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4365258, 2.4373891
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7490482, 2.7458973

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2966678, upper bound: 1.3065461
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2939517, upper bound: 1.3046577
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6110058, 2.6244495
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2951775, 2.3006263
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3774347, 2.3645225
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8499999, 1.8359966
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2264547, 3.2315426
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4599586, 2.4463825
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2603645, 3.2699623
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4360280, 2.4431789
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7455959, 2.7518229

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2972516, upper bound: 1.3054896
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2991366, upper bound: 1.3036065
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6110077, 2.6244471
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2951770, 2.3006248
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3774357, 2.3645210
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8500004, 1.8359969
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2264547, 3.2315431
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4599586, 2.4463820
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2603645, 3.2699614
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4360275, 2.4431787
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7455959, 2.7518229

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2965454, upper bound: 1.3077334
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2994081, upper bound: 1.3048818
time: 5.18 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.3048832, upper bound: 1.2994049
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.3021659, upper bound: 1.3021247
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.3077353, upper bound: 1.2992534
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.3031110, upper bound: 1.2992661
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.3067489, upper bound: 1.3003912
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.3021182, upper bound: 1.3003986
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.3008894, upper bound: 1.3018561
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.3027761, upper bound: 1.2999715
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.2942293, upper bound: 1.3017113
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.2970878, upper bound: 1.2988554
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.2966678, upper bound: 1.3065461
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.2939517, upper bound: 1.3046577
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.2972516, upper bound: 1.3054896
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.2991366, upper bound: 1.3036065
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.2965454, upper bound: 1.3077334
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.41
Output dim: 7, lower bound: -1.2994081, upper bound: 1.3048818

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6230383, 2.6098316
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2525725, 2.2375069
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2701883, 2.2988429
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8151588, 1.8326340
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1924982, 3.1795988
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4346657, 2.4451976
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2515421, 3.2503262
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3976603, 2.3755846
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7572908, 2.7398324

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3048831, upper bound: 1.2993972
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3002500, upper bound: 1.2994055
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6136703, 2.6191976
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2475419, 2.2425380
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2804317, 2.2885971
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8210654, 1.8267276
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1888638, 3.1832328
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4398537, 2.4400101
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2460537, 3.2558107
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3956509, 2.3775940
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7520695, 2.7450533

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3021312, upper bound: 1.2998014
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2988531, upper bound: 1.2999068
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6701813, 2.6658781
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2680621, 2.2772684
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3453822, 2.3323121
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8525867, 1.8572137
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2009401, 3.2073016
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4538040, 2.4652400
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2874098, 3.2749200
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3852000, 2.3949804
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7671027, 2.7773280

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3077341, upper bound: 1.2965369
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2965397, upper bound: 1.2992530
time: 6.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6701813, 2.6658769
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2680612, 2.2772675
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3453817, 2.3323116
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8525882, 1.8572109
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2009401, 3.2073016
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4538050, 2.4652371
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2874079, 3.2749181
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3851986, 2.3949823
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7671008, 2.7773275

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6209

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2987821, upper bound: 1.2977138
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3018484, upper bound: 1.2977094
time: 6.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6244559, 2.6110125
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3006182, 2.2951717
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3645182, 2.3774319
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8359809, 1.8499866
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2315426, 3.2264566
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4463658, 2.4599447
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2699537, 3.2603545
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4431896, 2.4360378
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7518215, 2.7455940

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3038916, upper bound: 1.3003911
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3067474, upper bound: 1.2975299
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6244540, 2.6110110
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3006172, 2.2951703
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3645182, 2.3774309
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8359818, 1.8499837
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2315426, 3.2264566
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4463677, 2.4599419
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2699518, 3.2603526
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4431877, 2.4360397
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7518210, 2.7455935

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3012381, upper bound: 1.3003893
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3021087, upper bound: 1.2995167
time: 5.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6169229, 2.6202364
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2931433, 2.2965250
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3732491, 2.3661785
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8415961, 1.8438871
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2245617, 3.2250681
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4515548, 2.4549165
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2635393, 3.2652216
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4380722, 2.4333615
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7420797, 2.7439842

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2986745, upper bound: 1.2985581
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2985714, upper bound: 1.3018223
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6149440, 2.6222153
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2919102, 2.2977581
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3737555, 2.3656721
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8416967, 1.8437865
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2228861, 3.2267447
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4517169, 2.4547558
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2638454, 3.2649145
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4364939, 2.4349394
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7397680, 2.7462959

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2996208, upper bound: 1.2987883
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3027748, upper bound: 1.2986982
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6178141, 2.6134436
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2506928, 2.2342658
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2665815, 2.2936664
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8220043, 1.8240361
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1850510, 3.1660643
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4429626, 2.4367785
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2452030, 3.2535501
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3887382, 2.3792148
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7555809, 2.7390695

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2942270, upper bound: 1.3017092
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2970469, upper bound: 1.3017109
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6180458, 2.6132114
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2410722, 2.2438860
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2823215, 2.2779264
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8254766, 1.8205638
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1772375, 3.1738777
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4399185, 2.4398227
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2535830, 3.2451696
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3738141, 2.3941388
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7443495, 2.7503014

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3015643, upper bound: 1.2973009
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3046433, upper bound: 1.2972968
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6217561, 2.6137931
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2941022, 2.2904458
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3631678, 2.3674803
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8432846, 1.8404453
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2117548, 3.2168899
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4547267, 2.4516258
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2640295, 3.2616177
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4334278, 2.4327137
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7445283, 2.7390656

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2958064, upper bound: 1.3065376
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2966575, upper bound: 1.3056763
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6197772, 2.6157722
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2928691, 2.2916789
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3636737, 2.3669744
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8433847, 1.8403447
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2100792, 3.2185664
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4548869, 2.4514651
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2643356, 3.2613106
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4318500, 2.4342916
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7422171, 2.7413774

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2985588, upper bound: 1.3000430
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2985437, upper bound: 1.3046572
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6128488, 2.6243134
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2927265, 2.2969427
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3759184, 2.3635120
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8496928, 1.8357902
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2231083, 3.2265205
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4599433, 2.4465284
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2594290, 3.2693338
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4329305, 2.4385035
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7410746, 2.7449899

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2972516, upper bound: 1.3008598
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962566, upper bound: 1.3054902
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6108699, 2.6262922
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2914934, 2.2981758
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3764243, 2.3630061
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8497934, 1.8356895
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2214317, 3.2281971
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4601035, 2.4463677
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2597361, 3.2690268
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4313526, 2.4400814
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7387629, 2.7473016

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2991366, upper bound: 1.2989722
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962566, upper bound: 1.3036096
time: 7.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6095991, 2.6232700
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2471256, 2.2429528
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2831025, 2.2859278
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8291626, 1.8186312
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1874123, 3.1846881
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4482412, 2.4316211
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2419453, 3.2599230
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3905087, 2.3827364
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7510643, 2.7460604

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2943278, upper bound: 1.3044295
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2942247, upper bound: 1.3076995
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6098318, 2.6230376
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2375050, 2.2525730
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2988424, 2.2701879
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8326349, 1.8151588
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1795988, 3.1925015
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4451971, 2.4346652
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2503262, 3.2515426
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3755846, 2.3976605
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7398329, 2.7572918

Time for backsubstitution: 14.63 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.7229785919189453
rel_dist={7: [-1.3105157996816672, 1.3105151877312444]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825692, upper bound: 1.1847141
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847170, upper bound: 1.1825680
time: 6.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.19
Output dim: 7, lower bound: -1.1825692, upper bound: 1.1847141
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.19
Output dim: 7, lower bound: -1.1847170, upper bound: 1.1825680

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5279198, 2.5280936
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2073231, 2.2001081
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2774973, 2.2893019
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8233442, 1.8259487
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1183300, 3.1124706
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4197202, 2.4174366
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1746902, 3.1809750
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3416748, 2.3304815
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6874557, 2.6790318

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825680, upper bound: 1.1827271
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1805825, upper bound: 1.1847133
time: 5.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5280933, 2.5279193
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2001085, 2.2073231
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2893023, 2.2774973
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8259487, 1.8233445
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1124697, 3.1183310
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4174361, 2.4197197
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1809750, 3.1746893
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3304815, 2.3416746
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6790318, 2.6874561

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1834960, upper bound: 1.1802595
time: 7.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1824053, upper bound: 1.1813467
time: 6.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.06
Output dim: 7, lower bound: -1.1825680, upper bound: 1.1827271
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.06
Output dim: 7, lower bound: -1.1805825, upper bound: 1.1847133
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.06
Output dim: 7, lower bound: -1.1834960, upper bound: 1.1802595
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.06
Output dim: 7, lower bound: -1.1824053, upper bound: 1.1813467

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4786615, 2.4718111
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1809483, 2.1699605
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2180400, 2.2375278
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7893834, 1.7964172
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1011667, 3.0925813
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3975282, 2.3991361
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1458206, 3.1479917
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3387160, 2.3260159
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6651049, 2.6527653

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1813467, upper bound: 1.1804182
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1802583, upper bound: 1.1815108
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4716368, 2.4788356
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1771755, 2.1737337
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2257228, 2.2298450
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7938132, 1.7919877
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0984411, 3.0953069
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4014192, 2.3952451
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1417065, 3.1521058
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3372092, 2.3275230
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6611900, 2.6566806

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1791020, upper bound: 1.1834296
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791043, upper bound: 1.1811649
time: 7.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5280914, 2.5279193
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2001081, 2.2073245
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2893014, 2.2774973
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8259487, 1.8233442
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1124687, 3.1183300
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4174361, 2.4197192
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1809750, 3.1746902
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3304815, 2.3416743
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6790328, 2.6874561

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1834960, upper bound: 1.1768805
time: 9.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1801159, upper bound: 1.1802584
time: 7.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5280933, 2.5279176
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2001085, 2.2073231
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2893023, 2.2774963
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8259487, 1.8233445
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1124697, 3.1183310
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4174361, 2.4197192
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1809750, 3.1746893
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3304811, 2.3416746
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6790328, 2.6874561

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1790249, upper bound: 1.1779646
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1790231, upper bound: 1.1813448
time: 7.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 27.57
Output dim: 7, lower bound: -1.1813467, upper bound: 1.1804182
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 27.57
Output dim: 7, lower bound: -1.1802583, upper bound: 1.1815108
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.57
Output dim: 7, lower bound: -1.1791020, upper bound: 1.1834296
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 27.57
Output dim: 7, lower bound: -1.1791043, upper bound: 1.1811649
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.57
Output dim: 7, lower bound: -1.1834960, upper bound: 1.1768805
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 27.57
Output dim: 7, lower bound: -1.1801159, upper bound: 1.1802584
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 27.57
Output dim: 7, lower bound: -1.1790249, upper bound: 1.1779646
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 27.57
Output dim: 7, lower bound: -1.1790231, upper bound: 1.1813448

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4729843, 2.4786992
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1744170, 2.1700501
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2242041, 2.2287045
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7935057, 1.7917547
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0946760, 3.0902848
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4014030, 2.3953485
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1416845, 3.1523190
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3382707, 2.3274014
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6627789, 2.6565366

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791020, upper bound: 1.1800427
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1757225, upper bound: 1.1834297
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5280848, 2.5279131
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2001023, 2.2073193
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2892971, 2.2774930
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8259315, 1.8233280
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1124687, 3.1183300
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4174185, 2.4197030
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1809626, 3.1746788
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3304715, 2.3416631
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6790261, 2.6874504

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6209

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825595, upper bound: 1.1768733
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1834889, upper bound: 1.1759511
time: 6.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.18 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.18
Output dim: 7, lower bound: -1.1791020, upper bound: 1.1800427
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.18
Output dim: 7, lower bound: -1.1757225, upper bound: 1.1834297
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.18
Output dim: 7, lower bound: -1.1825595, upper bound: 1.1768733
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.18
Output dim: 7, lower bound: -1.1834889, upper bound: 1.1759511

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4729767, 2.4786906
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1744113, 2.1700435
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2242002, 2.2287011
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7934895, 1.7917371
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0946760, 3.0902843
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4013896, 2.3953338
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1416740, 3.1523075
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3382592, 2.3273914
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6627755, 2.6565328

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1745093, upper bound: 1.1811036
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1733943, upper bound: 1.1822177
time: 5.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5253344, 2.5282302
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1992025, 2.2074227
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2894263, 2.2764034
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8263083, 1.8201106
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1113539, 3.1184583
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4161510, 2.4198542
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1810932, 3.1736507
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3306575, 2.3400590
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6767478, 2.6877351

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825260, upper bound: 1.1746959
time: 10.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1793019, upper bound: 1.1747377
time: 6.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5280848, 2.5251622
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2001023, 2.2064195
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2882080, 2.2774930
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8227143, 1.8233280
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1124687, 3.1172147
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4174185, 2.4184361
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1799345, 3.1746788
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3288674, 2.3416631
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6790261, 2.6851721

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1834867, upper bound: 1.1759387
time: 8.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1801052, upper bound: 1.1759479
time: 5.72 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 28.75 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.75
Output dim: 7, lower bound: -1.1745093, upper bound: 1.1811036
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.75
Output dim: 7, lower bound: -1.1733943, upper bound: 1.1822177
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.75
Output dim: 7, lower bound: -1.1825260, upper bound: 1.1746959
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.75
Output dim: 7, lower bound: -1.1793019, upper bound: 1.1747377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.75
Output dim: 7, lower bound: -1.1834867, upper bound: 1.1759387
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.75
Output dim: 7, lower bound: -1.1801052, upper bound: 1.1759479

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5241823, 2.5275970
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1971903, 2.2037668
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2831521, 2.2729564
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8250570, 1.8194220
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1031103, 3.1034679
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4160604, 2.4198098
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1788654, 3.1724300
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3268776, 2.3379805
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6757784, 2.6859674

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1790235, upper bound: 1.1732144
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1813632, upper bound: 1.1732114
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5183449, 2.5123668
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2026811, 2.2086859
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2721510, 2.2634373
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7741461, 1.7808337
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1048155, 3.1084719
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3670797, 2.3743896
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1583290, 3.1499896
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3077621, 2.3167014
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6737456, 2.6791358

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1834855, upper bound: 1.1739489
time: 10.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1815014, upper bound: 1.1759371
time: 5.58 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 30.56 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 30.56
Output dim: 7, lower bound: -1.1790235, upper bound: 1.1732144
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 30.56
Output dim: 7, lower bound: -1.1813632, upper bound: 1.1732114
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 30.56
Output dim: 7, lower bound: -1.1834855, upper bound: 1.1739489
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 30.56
Output dim: 7, lower bound: -1.1815014, upper bound: 1.1759371

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4690981, 2.4560940
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1763067, 2.1785383
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2126932, 2.2116647
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7401738, 1.7512910
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0876532, 3.0885830
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3448811, 2.3560820
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1294603, 3.1170063
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3048072, 2.3122387
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6513944, 2.6528687

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1799446, upper bound: 1.1725137
time: 10.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1822080, upper bound: 1.1725107
time: 5.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 30.70 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 30.70
Output dim: 7, lower bound: -1.1799446, upper bound: 1.1725137
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 30.70
Output dim: 7, lower bound: -1.1822080, upper bound: 1.1725107
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.7229785919189453
rel_dist={7: [-1.1847181417998263, 1.1847155369154763]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2073.30 seconds
