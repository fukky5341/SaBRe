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
execution time: IAR + LP analysis = 15.13 + 33.50 = 48.63 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.37 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.7229785919189453
rel_dist={7: [-1.52484302938982, 1.5248425766176732]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.7229785919189453
rel_dist={7: [-1.1847181417998263, 1.1847155369154763]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.7229785919189453
rel_dist={7: [-0.9114016965347282, 0.9113982906257574]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.7229785919189453
rel_dist={7: [-1.052759738359363, 1.0527567565617817]}

## Binary Search Result
Binary search time: 206.87 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3344.50 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6256696, upper bound: 1.6265164
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265167, upper bound: 1.6256696
time: 4.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.18
Output dim: 7, lower bound: -1.6256696, upper bound: 1.6265164
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.18
Output dim: 7, lower bound: -1.6265167, upper bound: 1.6256696

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.1296191, 3.1367784
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.5179892, 2.5203304
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6464119, 2.6435690
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1467733, 2.1383862
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5412569, 3.5441589
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7778263, 2.7811356
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6945057, 3.6918011
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6916904, 2.6875138
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0549383, 3.0609193

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6256685, upper bound: 1.6217832
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6209214, upper bound: 1.6265158
time: 4.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.1323705, 3.1296196
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.5188890, 2.5179892
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6435690, 2.6446581
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1383867, 2.1416042
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5423737, 3.5412574
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7790928, 2.7778263
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6918011, 3.6928291
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6875138, 2.6891172
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0572166, 3.0549388

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6265156, upper bound: 1.6209212
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6217832, upper bound: 1.6256683
time: 4.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.77 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.77
Output dim: 7, lower bound: -1.6256685, upper bound: 1.6217832
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.77
Output dim: 7, lower bound: -1.6209214, upper bound: 1.6265158
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.77
Output dim: 7, lower bound: -1.6265156, upper bound: 1.6209212
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.77
Output dim: 7, lower bound: -1.6217832, upper bound: 1.6256683

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0897288, 3.0804958
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4966455, 2.4901819
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5869551, 2.6020384
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1128106, 2.1147599
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5277290, 3.5242710
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7556353, 2.7680235
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6711197, 3.6588168
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6907415, 2.6830480
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0378098, 3.0346541

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6256492, upper bound: 1.6118886
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6158031, upper bound: 1.6217650
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0733371, 3.0968866
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4878411, 2.4989862
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6048822, 2.5841117
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1231465, 2.1044238
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5213699, 3.5306306
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7647142, 2.7589450
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6615219, 3.6684151
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6872249, 2.6865647
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0286736, 3.0437903

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6209025, upper bound: 1.6166211
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6110559, upper bound: 1.6264975
time: 4.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924773, 3.0733373
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4975448, 2.4878411
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5841122, 2.6031280
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1044240, 2.1179776
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5288448, 3.5213699
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7569017, 2.7647138
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6684151, 3.6598468
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6865649, 2.6846516
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0400882, 3.0286736

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6264973, upper bound: 1.6110558
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6166208, upper bound: 1.6209025
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0760875, 3.0897281
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4887404, 2.4966450
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6020384, 2.5852013
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1147599, 2.1076417
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5224857, 3.5277290
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7659807, 2.7556348
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6588173, 3.6694450
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6830478, 2.6881683
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0309520, 3.0378098

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6217649, upper bound: 1.6158039
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6118886, upper bound: 1.6256492
time: 5.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 7, lower bound: -1.6256492, upper bound: 1.6118886
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 7, lower bound: -1.6158031, upper bound: 1.6217650
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 7, lower bound: -1.6209025, upper bound: 1.6166211
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 7, lower bound: -1.6110559, upper bound: 1.6264975
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 7, lower bound: -1.6264973, upper bound: 1.6110558
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 7, lower bound: -1.6166208, upper bound: 1.6209025
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 7, lower bound: -1.6217649, upper bound: 1.6158039
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 7, lower bound: -1.6118886, upper bound: 1.6256492

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0885768, 3.0805533
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4968257, 2.4865265
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5806818, 2.6023602
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1115599, 2.1148207
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5284758, 3.5092821
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7555447, 2.7680392
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6688938, 3.6589408
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6869612, 2.6832373
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0379028, 3.0328846

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6110560, upper bound: 1.6037802
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6175399, upper bound: 1.6118864
time: 7.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0897288, 3.0793445
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4929900, 2.4901819
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5869551, 2.5957642
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1128106, 2.1135092
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5127392, 3.5242710
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7556353, 2.7679329
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6711197, 3.6565909
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6907415, 2.6792674
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0360403, 3.0346541

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6029407, upper bound: 1.6136509
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6029407, upper bound: 1.6217626
time: 5.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0721850, 3.0969441
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4880214, 2.4953308
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5986071, 2.5844340
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1218958, 2.1044846
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5221167, 3.5156412
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7646236, 2.7589607
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6592960, 3.6685390
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6834445, 2.6867540
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0287666, 3.0420208

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6209008, upper bound: 1.6085390
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6127683, upper bound: 1.6166188
time: 5.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0733371, 3.0957351
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4841857, 2.4989862
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6048822, 2.5778375
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1231465, 2.1031733
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5063801, 3.5306306
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7647142, 2.7588544
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6615219, 3.6661892
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6872249, 2.6827841
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0269041, 3.0437903

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6110542, upper bound: 1.6184156
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6029388, upper bound: 1.6264952
time: 5.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0913262, 3.0733948
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4977245, 2.4841857
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5778379, 2.6034498
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1031733, 2.1180384
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5295916, 3.5063806
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7568111, 2.7647295
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6661892, 3.6599708
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6827841, 2.6848407
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0401812, 3.0269036

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6264954, upper bound: 1.6029386
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6184153, upper bound: 1.6110544
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0924773, 3.0721860
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4938889, 2.4878411
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5841122, 2.5968537
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1044240, 2.1167269
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5138559, 3.5213699
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7569017, 2.7646236
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6684151, 3.6576209
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6865649, 2.6808708
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0383186, 3.0286736

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6110561, upper bound: 1.6127679
time: 8.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085390, upper bound: 1.6209017
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0749354, 3.0897856
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4889212, 2.4929895
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5957642, 2.5855236
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1135092, 2.1077023
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5232325, 3.5127401
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7658901, 2.7556505
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6565914, 3.6695690
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6792679, 2.6883574
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0310450, 3.0360398

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6217630, upper bound: 1.6077025
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6037819, upper bound: 1.6158013
time: 9.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0760875, 3.0885768
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4850855, 2.4966450
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.6020384, 2.5789270
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.1147599, 2.1063910
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5074959, 3.5277290
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7659807, 2.7555447
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6588173, 3.6672192
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6830478, 2.6843874
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0291824, 3.0378098

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6118867, upper bound: 1.6175400
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6037801, upper bound: 1.6256472
time: 5.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6110560, upper bound: 1.6037802
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6175399, upper bound: 1.6118864
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6029407, upper bound: 1.6136509
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6029407, upper bound: 1.6217626
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6209008, upper bound: 1.6085390
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6127683, upper bound: 1.6166188
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6110542, upper bound: 1.6184156
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6029388, upper bound: 1.6264952
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6264954, upper bound: 1.6029386
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6184153, upper bound: 1.6110544
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6110561, upper bound: 1.6127679
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6085390, upper bound: 1.6209017
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6217630, upper bound: 1.6077025
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6037819, upper bound: 1.6158013
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6118867, upper bound: 1.6175400
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.82
Output dim: 7, lower bound: -1.6037801, upper bound: 1.6256472

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0829086, 3.0677543
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4998207, 2.4887934
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5646219, 2.5909734
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0629811, 2.0804117
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5222750, 3.5005379
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7051978, 2.7323718
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6513977, 3.6342492
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6709805, 2.6582592
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0336256, 3.0268459

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6207536, upper bound: 1.6037770
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6256449, upper bound: 1.5988958
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0757780, 3.0748825
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4990911, 2.4895215
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5692902, 2.5863008
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0771503, 2.0662425
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5197325, 3.5030804
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7198768, 2.7176924
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6442032, 3.6414399
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6619825, 2.6672573
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0318642, 3.0286055

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6126495, upper bound: 1.6118842
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6175373, upper bound: 1.6070138
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0840597, 3.0665455
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4959850, 2.4924488
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5708971, 2.5843773
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0642323, 2.0791001
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5065393, 3.5155268
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7052884, 2.7322659
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6536255, 3.6318994
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6747608, 2.6542892
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0317621, 3.0286145

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6109060, upper bound: 1.6136485
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6157988, upper bound: 1.6087463
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0769300, 3.0736732
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4952555, 2.4931774
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5755653, 2.5797043
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0784016, 2.0649309
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5039959, 3.5180693
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7199683, 2.7175860
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6464300, 3.6390901
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6657629, 2.6632872
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0300016, 3.0303741

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5980425, upper bound: 1.6217600
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6029381, upper bound: 1.6168636
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0665150, 3.0841451
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4910154, 2.4975972
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5825481, 2.5730429
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0733175, 2.0700755
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5159149, 3.5068974
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7142768, 2.7232938
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6417942, 3.6438475
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6674643, 2.6617758
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0244875, 3.0359821

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6160052, upper bound: 1.6085362
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6208982, upper bound: 1.6036567
time: 5.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0593872, 3.0912757
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4902878, 2.4983263
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5872211, 2.5683746
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0874867, 2.0559063
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5133734, 3.5094404
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7289557, 2.7086134
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6346035, 3.6510434
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6584663, 2.6707737
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0227280, 3.0377436

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5980425, upper bound: 1.6166183
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6127658, upper bound: 1.6117639
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0676661, 3.0829363
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4871798, 2.5012531
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5888233, 2.5664468
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0745687, 2.0687642
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5001783, 3.5218863
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7143664, 2.7231879
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6440220, 3.6414976
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6712446, 2.6578059
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0226250, 3.0377507

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6061567, upper bound: 1.6184131
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6110516, upper bound: 1.6135069
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0605383, 3.0900669
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4864521, 2.5019817
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5934963, 2.5617781
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0887375, 2.0545950
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.4976358, 3.5244293
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7290463, 2.7085075
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6368313, 3.6486936
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6622462, 2.6668038
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0208654, 3.0395117

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5980425, upper bound: 1.6264924
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6029362, upper bound: 1.6216155
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0856590, 3.0605960
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.5007210, 2.4864521
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5617781, 2.5920630
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0545945, 2.0836287
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5233908, 3.4976368
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7064643, 2.7290621
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6486931, 3.6352777
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6668038, 2.6598616
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0359030, 3.0208654

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6216157, upper bound: 1.6029362
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6264929, upper bound: 1.5980405
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0785275, 3.0677238
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4999914, 2.4871802
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5664463, 2.5873904
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0687637, 2.0694594
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5208473, 3.5001793
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7211452, 2.7143826
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6414986, 3.6424685
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6578059, 2.6688595
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0341425, 3.0226250

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6135069, upper bound: 1.6110513
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6184127, upper bound: 1.6061569
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0868092, 3.0593870
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4968853, 2.4901080
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5680532, 2.5854669
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0558457, 2.0823174
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5076542, 3.5126257
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7065549, 2.7289557
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6509209, 3.6329279
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6705837, 2.6558917
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0340414, 3.0226336

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6117641, upper bound: 1.6127657
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6166165, upper bound: 1.6078909
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0796795, 3.0665150
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4961557, 2.4908361
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5727215, 2.5807939
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0700150, 2.0681481
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5051107, 3.5151682
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7212348, 2.7142768
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6437254, 3.6401186
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6615858, 2.6648898
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0322800, 3.0243931

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.5980425, upper bound: 1.6208984
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6085365, upper bound: 1.6160054
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0692654, 3.0769868
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4919157, 2.4952564
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5797043, 2.5741324
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0649309, 2.0732925
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5170298, 3.5039959
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7155433, 2.7199841
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6390896, 3.6448760
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6632872, 2.6633782
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0267658, 3.0300016

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6168641, upper bound: 1.6076995
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6217605, upper bound: 1.6027996
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0621376, 3.0841174
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4911880, 2.4959850
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5843773, 2.5694637
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0791001, 2.0591235
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5144873, 3.5065393
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7302241, 2.7053037
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6318989, 3.6520720
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6542892, 2.6723762
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0250063, 3.0317626

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6087462, upper bound: 1.6157988
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6136485, upper bound: 1.6109071
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0704165, 3.0757778
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4880800, 2.4989119
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5859795, 2.5675359
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0661821, 2.0719810
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.5012941, 3.5189848
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7156339, 2.7198777
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6413174, 3.6425261
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6670675, 2.6594083
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0249033, 3.0317698

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6070141, upper bound: 1.6175371
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.6118842, upper bound: 1.6126493
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -3.0632887, 3.0829082
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.4873524, 2.4996409
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.5906525, 2.5628676
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -2.0803509, 2.0578120
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.4987516, 3.5215282
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.7303138, 2.7051978
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.6341267, 3.6497221
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.6580696, 2.6684062
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -3.0231438, 3.0335312

Time for backsubstitution: 14.39 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.7229785919189453
rel_dist={7: [-1.6265317094795417, 1.626531505248428]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3096463, upper bound: 1.3105074
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3096484, upper bound: 1.3096454
time: 7.19 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.39
Output dim: 7, lower bound: -1.3096463, upper bound: 1.3105074
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.39
Output dim: 7, lower bound: -1.3096484, upper bound: 1.3096454

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6773391, 2.6814294
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3221588, 2.3234963
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.4405727, 2.4389477
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.9201126, 1.9153204
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2539721, 3.2556305
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.5176535, 2.5195451
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.3185558, 3.3170109
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4662561, 2.4638696
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7756224, 2.7790399

Time for backsubstitution: 13.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3096450, upper bound: 1.3077912
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3069323, upper bound: 1.3105065
time: 5.05 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6800895, 2.6773386
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.3230577, 2.3221588
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.4389477, 2.4400373
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.9153204, 1.9185383
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2550888, 3.2539725
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.5189209, 2.5176535
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.3170109, 3.3180389
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4638696, 2.4654729
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7779012, 2.7756228

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3105050, upper bound: 1.3069299
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3077902, upper bound: 1.3096447
time: 5.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.57
Output dim: 7, lower bound: -1.3096450, upper bound: 1.3077912
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.57
Output dim: 7, lower bound: -1.3069323, upper bound: 1.3105065
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.57
Output dim: 7, lower bound: -1.3105050, upper bound: 1.3069299
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.57
Output dim: 7, lower bound: -1.3077902, upper bound: 1.3096447

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6304221, 2.6251469
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2970414, 2.2933483
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3811159, 2.3897347
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8861504, 1.8872643
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2377186, 3.2357430
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4954624, 2.5025420
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2910566, 3.2840271
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4638000, 2.4594038
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7545781, 2.7527747

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3008505, upper bound: 1.3017073
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3035633, upper bound: 1.3077844
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6210570, 2.6345129
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2920108, 2.2983794
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3913603, 2.3794913
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8920569, 1.8813579
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2340851, 3.2393770
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.5006504, 2.4973540
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2855721, 3.2895117
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4617906, 2.4614134
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7493572, 2.7579956

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3008505, upper bound: 1.3044227
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3008484, upper bound: 1.3104975
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6331725, 2.6210563
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2979417, 2.2920103
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3794909, 2.3908238
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8813577, 1.8904822
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2388344, 3.2340851
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4967299, 2.5006509
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2895117, 3.2850571
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4614134, 2.4610074
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7568564, 2.7493577

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3104982, upper bound: 1.3008480
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3044219, upper bound: 1.3069231
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6238055, 2.6304224
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2929111, 2.2970419
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3897352, 2.3805804
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8872643, 1.8845758
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2352009, 3.2377191
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.5019178, 2.4954629
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2840271, 3.2905416
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4594035, 2.4630170
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7516360, 2.7545781

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3077833, upper bound: 1.3035631
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3008505, upper bound: 1.3096372
time: 5.24 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.40
Output dim: 7, lower bound: -1.3008505, upper bound: 1.3017073
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.40
Output dim: 7, lower bound: -1.3035633, upper bound: 1.3077844
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.40
Output dim: 7, lower bound: -1.3008505, upper bound: 1.3044227
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.40
Output dim: 7, lower bound: -1.3008484, upper bound: 1.3104975
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.40
Output dim: 7, lower bound: -1.3104982, upper bound: 1.3008480
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.40
Output dim: 7, lower bound: -1.3044219, upper bound: 1.3069231
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.40
Output dim: 7, lower bound: -1.3077833, upper bound: 1.3035631
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.40
Output dim: 7, lower bound: -1.3008505, upper bound: 1.3096372

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6292701, 2.6246862
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2955775, 2.2896929
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3748417, 2.3872299
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8848996, 1.8867631
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2317209, 3.2207537
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4953728, 2.5025120
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2888308, 3.2831440
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4600201, 2.4578917
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7538733, 2.7510056

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3096363, upper bound: 1.2970761
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3050235, upper bound: 1.3017021
time: 5.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6299624, 2.6239953
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2933860, 2.2918849
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3786116, 2.3834605
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8856487, 1.8860137
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2227297, 3.2297459
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4954338, 2.5024514
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2901735, 3.2818012
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4622884, 2.4556231
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7528090, 2.7520700

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3035615, upper bound: 1.3031589
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2989451, upper bound: 1.3077809
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6199050, 2.6340523
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2905469, 2.2947240
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3850851, 2.3769860
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8908062, 1.8808570
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2280874, 3.2243872
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.5005608, 2.4973240
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2833462, 3.2886286
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4580102, 2.4599013
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7486525, 2.7562261

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3069215, upper bound: 1.2997952
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3044195
time: 5.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6205955, 2.6333616
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2883554, 2.2969160
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3888550, 2.3732171
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8915553, 1.8801074
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2190952, 3.2333798
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.5006218, 2.4972639
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2846889, 3.2872858
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4602785, 2.4576325
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7475882, 2.7572904

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3058784
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3104978
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6320205, 2.6205957
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2964773, 2.2883554
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3732176, 2.3883195
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8801069, 1.8899808
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2328377, 3.2190957
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4966393, 2.5006208
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2872858, 3.2841740
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4576330, 2.4594951
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7561512, 2.7475877

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962282, upper bound: 1.2962261
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3058787, upper bound: 1.3008459
time: 5.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6327119, 2.6199050
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2942858, 2.2905469
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3769865, 2.3845496
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8808565, 1.8892314
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2238455, 3.2280879
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4967003, 2.5005603
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2886286, 3.2828312
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4599013, 2.4572265
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7550869, 2.7486520

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3044200, upper bound: 1.3023057
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2997959, upper bound: 1.3069205
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6226544, 2.6299620
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2914467, 2.2933860
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3834600, 2.3780756
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8860135, 1.8840747
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2292032, 3.2227297
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.5018272, 2.4954329
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2818012, 3.2896585
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4556236, 2.4615045
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7509308, 2.7528086

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3077815, upper bound: 1.2989446
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3035612
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6233459, 2.6292710
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2892551, 2.2955780
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3872299, 2.3743062
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8867626, 1.8833251
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2202110, 3.2317219
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.5018883, 2.4953723
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2831440, 3.2883158
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4578919, 2.4592361
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7498665, 2.7538729

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3050231
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2970768, upper bound: 1.3096376
time: 5.26 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.3096363, upper bound: 1.2970761
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.3050235, upper bound: 1.3017021
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.3035615, upper bound: 1.3031589
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2989451, upper bound: 1.3077809
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.3069215, upper bound: 1.2997952
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3044195
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3058784
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3104978
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2962282, upper bound: 1.2962261
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.3058787, upper bound: 1.3008459
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.3044200, upper bound: 1.3023057
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2997959, upper bound: 1.3069205
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.3077815, upper bound: 1.2989446
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3035612
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2962282, upper bound: 1.3050231
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.06
Output dim: 7, lower bound: -1.2970768, upper bound: 1.3096376

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6205463, 2.6118875
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2982607, 2.2919598
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3587823, 2.3738403
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8363214, 1.8462815
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2244310, 3.2120099
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4450259, 2.4605536
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2682514, 3.2584519
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4401832, 2.4329135
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7488403, 2.7449665

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3067843, upper bound: 1.2970745
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3096349, upper bound: 1.2942161
time: 5.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6164722, 2.6159606
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2978439, 2.2923756
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3614502, 2.3711700
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8444180, 1.8381848
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2229776, 3.2134628
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4534144, 2.4521651
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2641392, 3.2625613
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4350414, 2.4380553
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7478342, 2.7459722

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3021676, upper bound: 1.3017011
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3050220, upper bound: 1.2988423
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6212378, 2.6111965
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2960691, 2.2941513
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3625517, 2.3700709
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8370709, 1.8455322
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2154388, 3.2210021
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4450850, 2.4604926
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2695942, 3.2571092
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4424515, 2.4306450
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7477760, 2.7460313

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3007085, upper bound: 1.3031570
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962268, upper bound: 1.3003001
time: 5.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6171627, 2.6152697
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2956524, 2.2945671
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3652191, 2.3674011
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8451676, 1.8374355
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2139854, 3.2224550
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4534736, 2.4521046
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2654819, 3.2612185
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4373097, 2.4357867
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7467699, 2.7470365

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2960850, upper bound: 1.3077792
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2933682, upper bound: 1.3049269
time: 6.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6111794, 2.6212535
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2932291, 2.2969904
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3690257, 2.3635941
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8422275, 1.8403752
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2207966, 3.2156439
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4502139, 2.4553657
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2627640, 3.2639365
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4381733, 2.4349232
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7436190, 2.7501874

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2933682, upper bound: 1.2997947
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3069200, upper bound: 1.2969352
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6071062, 2.6253281
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2928133, 2.2974072
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3716960, 2.3609266
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8503242, 1.8322787
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2193441, 3.2170973
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4586005, 2.4469771
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2586555, 3.2680488
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4330320, 2.4400647
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7426138, 2.7511940

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2994485, upper bound: 1.3044198
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3023054, upper bound: 1.3015606
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6118698, 2.6205626
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2910376, 2.2991819
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3727951, 2.3598251
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8429770, 1.8396258
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2118044, 3.2246361
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4502730, 2.4553056
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2641068, 3.2625937
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4404421, 2.4326544
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7425547, 2.7512517

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2933681, upper bound: 1.3058765
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3008450, upper bound: 1.3030180
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6077967, 2.6246371
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2906218, 2.2995987
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3754654, 2.3571572
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8510737, 1.8315291
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2103510, 3.2260895
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4586616, 2.4469166
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2599983, 3.2667060
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4353004, 2.4377961
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7415495, 2.7522583

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2933659, upper bound: 1.3104943
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2933682, upper bound: 1.3076410
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6232967, 2.6077967
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2991610, 2.2906218
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3571577, 2.3749299
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8315291, 1.8494985
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2255459, 3.2103519
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4462924, 2.4586620
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2667065, 3.2594810
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4377961, 2.4345160
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7511187, 2.7415495

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3076423, upper bound: 1.2962237
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3104949, upper bound: 1.2933647
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6192226, 2.6118698
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2987442, 2.2910376
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3598251, 2.3722596
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8396258, 1.8414018
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2240925, 3.2118044
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4546809, 2.4502740
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2625942, 3.2635894
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4326544, 2.4396577
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7501125, 2.7425547

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3030186, upper bound: 1.3008444
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962268, upper bound: 1.2979920
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6239872, 2.6071060
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2969685, 2.2928133
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3609266, 2.3711605
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8322783, 1.8487492
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2165537, 3.2193441
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4463534, 2.4586015
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2680492, 3.2581382
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4400644, 2.4322474
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7500544, 2.7426138

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3015613, upper bound: 1.3023047
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3044186, upper bound: 1.2994475
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6199131, 2.6111789
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2965527, 2.2932296
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3635945, 2.3684902
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8403749, 1.8406525
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2151003, 3.2207966
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4547420, 2.4502130
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2639370, 3.2622466
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4349232, 2.4373891
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7490482, 2.7436190

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2933682, upper bound: 1.3069191
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2962268, upper bound: 1.3040684
time: 6.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6139297, 2.6171629
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2941294, 2.2956524
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3674011, 2.3646836
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8374352, 1.8435922
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2219114, 3.2139859
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4514804, 2.4534745
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2612190, 3.2649655
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4357867, 2.4365253
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7458973, 2.7467699

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3049259, upper bound: 1.2989424
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3077800, upper bound: 1.2960845
time: 5.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6098557, 2.6212375
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2937136, 2.2960691
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3700714, 2.3620162
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8455319, 1.8354955
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2204590, 3.2154388
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4598689, 2.4450860
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2571087, 3.2690768
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4306450, 2.4416671
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7448921, 2.7477760

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2933682, upper bound: 1.3035603
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3031581, upper bound: 1.3007096
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.6146202, 2.6164720
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2919378, 2.2978444
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3711700, 2.3609142
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8381848, 1.8428428
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.2129192, 3.2229781
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4515414, 2.4534140
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.2625618, 3.2636228
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.4380550, 2.4342570
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.7448330, 2.7478342

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2988432, upper bound: 1.3050210
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3017018, upper bound: 1.3021670
time: 5.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3067843, upper bound: 1.2970745
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3096349, upper bound: 1.2942161
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3021676, upper bound: 1.3017011
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3050220, upper bound: 1.2988423
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3007085, upper bound: 1.3031570
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2962268, upper bound: 1.3003001
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2960850, upper bound: 1.3077792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2933682, upper bound: 1.3049269
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2933682, upper bound: 1.2997947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3069200, upper bound: 1.2969352
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2994485, upper bound: 1.3044198
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3023054, upper bound: 1.3015606
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2933681, upper bound: 1.3058765
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3008450, upper bound: 1.3030180
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2933659, upper bound: 1.3104943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2933682, upper bound: 1.3076410
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3076423, upper bound: 1.2962237
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3104949, upper bound: 1.2933647
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3030186, upper bound: 1.3008444
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2962268, upper bound: 1.2979920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3015613, upper bound: 1.3023047
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3044186, upper bound: 1.2994475
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2933682, upper bound: 1.3069191
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2962268, upper bound: 1.3040684
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3049259, upper bound: 1.2989424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3077800, upper bound: 1.2960845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2933682, upper bound: 1.3035603
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3031581, upper bound: 1.3007096
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.2988432, upper bound: 1.3050210
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.14
Output dim: 7, lower bound: -1.3017018, upper bound: 1.3021670
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.14
Output dim: 7, lower bound: -1.2970768, upper bound: 1.3096376
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.7229785919189453
rel_dist={7: [-1.3105157996816672, 1.3105151877312444]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837886, upper bound: 1.1847085
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837886, upper bound: 1.1837857
time: 5.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.72
Output dim: 7, lower bound: -1.1837886, upper bound: 1.1847085
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.72
Output dim: 7, lower bound: -1.1837886, upper bound: 1.1837857

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5265784, 2.5296462
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2568817, 2.2578850
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3719597, 2.3707409
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8445597, 1.8409653
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1582108, 3.1594543
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4309292, 2.4323483
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1932402, 3.1920815
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3911114, 2.3893216
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6825171, 2.6850801

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837854, upper bound: 1.1827212
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1817998, upper bound: 1.1847069
time: 8.17 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.5293288, 2.5265784
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2577815, 2.2568817
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3707409, 2.3718300
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8409653, 1.8441832
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1593266, 3.1582108
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4321966, 2.4309297
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1920815, 3.1931086
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3893213, 2.3909249
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6847959, 2.6825171

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6209
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6209

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847076, upper bound: 1.1817990
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827221, upper bound: 1.1837846
time: 5.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.68
Output dim: 7, lower bound: -1.1837854, upper bound: 1.1827212
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.68
Output dim: 7, lower bound: -1.1817998, upper bound: 1.1847069
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.68
Output dim: 7, lower bound: -1.1847076, upper bound: 1.1817990
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.68
Output dim: 7, lower bound: -1.1827221, upper bound: 1.1837846

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4773207, 2.4733639
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2305064, 2.2277374
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3125029, 2.3189669
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8105969, 1.8114324
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1410484, 3.1395669
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4087391, 2.4140477
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1643696, 3.1590967
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3881531, 2.3848557
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6601677, 2.6588149

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837795, upper bound: 1.1782354
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1793020, upper bound: 1.1827153
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4702959, 2.4803884
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2267337, 2.2315102
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3201857, 2.3112841
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8150268, 1.8070028
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1383228, 3.1422920
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4126301, 2.4101567
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1602554, 3.1632099
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3866458, 2.3863628
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6562519, 2.6627307

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1817940, upper bound: 1.1802234
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1773164, upper bound: 1.1847010
time: 7.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4800701, 2.4702959
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2314067, 2.2267342
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3112841, 2.3200564
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8070025, 1.8146503
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1421642, 3.1383233
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4100056, 2.4126296
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1632099, 3.1601267
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3863626, 2.3864594
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6624460, 2.6562519

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1773185, upper bound: 1.1773169
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1802242, upper bound: 1.1817934
time: 5.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4730453, 2.4773204
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2276340, 2.2305069
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3189669, 2.3123736
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8114324, 1.8102207
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1394386, 3.1410489
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4138966, 2.4087386
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1590967, 3.1642408
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3848557, 2.3879664
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6585307, 2.6601672

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827162, upper bound: 1.1793013
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1773185, upper bound: 1.1837790
time: 5.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.53
Output dim: 7, lower bound: -1.1837795, upper bound: 1.1782354
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.53
Output dim: 7, lower bound: -1.1793020, upper bound: 1.1827153
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.53
Output dim: 7, lower bound: -1.1817940, upper bound: 1.1802234
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.53
Output dim: 7, lower bound: -1.1773164, upper bound: 1.1847010
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.53
Output dim: 7, lower bound: -1.1773185, upper bound: 1.1773169
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.53
Output dim: 7, lower bound: -1.1802242, upper bound: 1.1817934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.53
Output dim: 7, lower bound: -1.1827162, upper bound: 1.1793013
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.53
Output dim: 7, lower bound: -1.1773185, upper bound: 1.1837790

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4761686, 2.4727306
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2284951, 2.2240820
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3062286, 2.3155198
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8093462, 1.8107440
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1328030, 3.1245775
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4086475, 2.4140034
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1621437, 3.1578779
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3843727, 2.3827765
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6591964, 2.6570458

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837773, upper bound: 1.1748427
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1803973, upper bound: 1.1782333
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4766865, 2.4722123
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2268519, 2.2257257
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3090553, 2.3126926
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8099084, 1.8101819
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1260595, 3.1313219
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4086933, 2.4139576
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1631508, 3.1568708
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3860741, 2.3810751
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6583982, 2.6578441

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1792998, upper bound: 1.1793265
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1759198, upper bound: 1.1827133
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4696627, 2.4792371
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2230783, 2.2294989
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3167381, 2.3050098
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8143377, 1.8057523
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1233339, 3.1340470
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4125843, 2.4100666
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1590366, 3.1609840
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3845668, 2.3825822
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6544828, 2.6617594

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1773143, upper bound: 1.1813187
time: 8.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1739276, upper bound: 1.1846989
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4718947, 2.4766872
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2256222, 2.2268515
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3126926, 2.3089261
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8101816, 1.8095319
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1311941, 3.1260595
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4138050, 2.4086938
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1568708, 3.1630220
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3810754, 2.3858869
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6575594, 2.6583982

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827142, upper bound: 1.1759191
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1793274, upper bound: 1.1792995
time: 5.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4724126, 2.4761691
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2239780, 2.2284956
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3155198, 2.3060994
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.8107438, 1.8089700
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1244497, 3.1328039
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.4138508, 2.4086485
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1578779, 3.1620150
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3827767, 2.3841856
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6567612, 2.6591964

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 539

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1782341, upper bound: 1.1803962
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1748434, upper bound: 1.1837762
time: 6.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1837773, upper bound: 1.1748427
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1803973, upper bound: 1.1782333
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1792998, upper bound: 1.1793265
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1759198, upper bound: 1.1827133
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1773143, upper bound: 1.1813187
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1739276, upper bound: 1.1846989
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1827142, upper bound: 1.1759191
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1793274, upper bound: 1.1792995
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1782341, upper bound: 1.1803962
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.94
Output dim: 7, lower bound: -1.1748434, upper bound: 1.1837762

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4664259, 2.4599316
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2310743, 2.2263479
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2901688, 2.3014627
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7607679, 1.7682383
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1251497, 3.1158338
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3583007, 2.3699470
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1405354, 3.1331859
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3632507, 2.3577983
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6539125, 2.6510067

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1816304, upper bound: 1.1748412
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837762, upper bound: 1.1727018
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4638882, 2.4624681
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2291174, 2.2283039
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2949967, 2.2966328
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7674022, 1.7616036
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1173153, 3.1236677
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3646388, 2.3636107
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1384583, 3.1352611
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3610954, 2.3599532
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6523590, 2.6525593

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1737728, upper bound: 1.1827124
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1759187, upper bound: 1.1805665
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4568634, 2.4694941
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2253447, 2.2320776
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3026814, 2.2889504
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7718320, 1.7571740
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1145897, 3.1263933
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3685279, 2.3597198
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1343460, 3.1393771
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3595886, 2.3614602
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6484437, 2.6564755

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1717827, upper bound: 1.1846978
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1739264, upper bound: 1.1825518
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4621506, 2.4638882
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2282009, 2.2291183
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2966332, 2.2948675
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7616034, 1.7670255
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1235380, 3.1173158
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3634601, 2.3646383
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1352615, 3.1383281
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3599529, 2.3609078
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6522746, 2.6523590

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1805673, upper bound: 1.1759179
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827131, upper bound: 1.1737721
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4596138, 2.4664261
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.2262449, 2.2310743
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.3014627, 2.2900395
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7682381, 1.7603910
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.1157045, 3.1251497
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3697963, 2.3583012
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1331863, 3.1404052
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3577986, 2.3630629
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6507220, 2.6539125

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 478
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 478

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1727029, upper bound: 1.1837756
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1748421, upper bound: 1.1816294
time: 6.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1816304, upper bound: 1.1748412
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1837762, upper bound: 1.1727018
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1737728, upper bound: 1.1827124
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1759187, upper bound: 1.1805665
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1717827, upper bound: 1.1846978
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1739264, upper bound: 1.1825518
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1805673, upper bound: 1.1759179
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1827131, upper bound: 1.1737721
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1727029, upper bound: 1.1837756
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.50
Output dim: 7, lower bound: -1.1748421, upper bound: 1.1816294

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4651923, 2.4585233
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1734028, 2.1758909
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2076402, 2.2071290
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7425342, 1.7474000
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0782938, 3.0748382
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3435402, 2.3574696
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1284008, 3.1147652
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3028064, 2.3085473
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6481476, 2.6536665

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825226, upper bound: 1.1726994
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1793376, upper bound: 1.1727023
time: 7.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4624801, 2.4612343
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1786613, 2.1706324
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2006631, 2.2141042
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7465644, 1.7433696
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0763206, 3.0768118
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3521605, 2.3488498
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1200390, 3.1231251
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3118443, 2.2995090
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6550188, 2.6467948

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1737728, upper bound: 1.1782713
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1717791, upper bound: 1.1814672
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4554553, 2.4682600
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1748877, 2.1744056
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2083473, 2.2064214
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7509942, 1.7389400
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0735950, 3.0795379
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3560514, 2.3449588
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1159248, 3.1272411
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3103371, 2.3010163
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6511030, 2.6507111

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1717791, upper bound: 1.1802592
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1717770, upper bound: 1.1834517
time: 5.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4556298, 2.4680858
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1676731, 2.1816206
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2201524, 2.1946168
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7535982, 1.7363358
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0677347, 3.0853977
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3537683, 2.3472419
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1222095, 3.1209564
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.2991443, 2.3122091
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6426792, 2.6591349

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1739263, upper bound: 1.1781136
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1739228, upper bound: 1.1813032
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4609156, 2.4624801
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1705289, 2.1786609
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2141042, 2.2005343
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7433696, 1.7461874
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0766830, 3.0763206
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3486967, 2.3521605
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1231251, 3.1199093
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.2995090, 2.3116579
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6465111, 2.6550188

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1814681, upper bound: 1.1737679
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1782720, upper bound: 1.1737719
time: 6.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4582052, 2.4651921
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1757874, 2.1734023
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2071290, 2.2075114
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7474003, 1.7421572
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0747099, 3.0782943
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3573170, 2.3435402
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1147661, 3.1282711
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3085470, 2.3026199
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6533823, 2.6481481

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1717826, upper bound: 1.1793367
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1726993, upper bound: 1.1825221
time: 9.52 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 29.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1825226, upper bound: 1.1726994
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1793376, upper bound: 1.1727023
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1737728, upper bound: 1.1782713
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1717791, upper bound: 1.1814672
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1717791, upper bound: 1.1802592
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1717770, upper bound: 1.1834517
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1739263, upper bound: 1.1781136
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1739228, upper bound: 1.1813032
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1814681, upper bound: 1.1737679
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1782720, upper bound: 1.1737719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1717826, upper bound: 1.1793367
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.66
Output dim: 7, lower bound: -1.1726993, upper bound: 1.1825221

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4651899, 2.4585230
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1734018, 2.1758924
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2076397, 2.2071290
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7425342, 1.7474000
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0782928, 3.0748358
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3435402, 2.3574705
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1283989, 3.1147652
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3028064, 2.3085468
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6481481, 2.6536665

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1790202, upper bound: 1.1711997
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1813598, upper bound: 1.1711974
time: 5.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4554553, 2.4682579
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1748877, 2.1744056
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2083473, 2.2064209
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7509942, 1.7389400
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0735922, 3.0795379
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3560514, 2.3449593
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1159239, 3.1272411
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3103366, 2.3010163
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6511030, 2.6507111

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1703392, upper bound: 1.1821730
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1703429, upper bound: 1.1799099
time: 8.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -17.5972595, -13.5857925, -17.5972595, -13.5857925, -2.4582052, 2.4651899
1: -10.2654305, -7.4666820, -10.2654305, -7.4666820, -2.1757874, 2.1734023
2: -6.4559197, -3.5972705, -6.4559197, -3.5972705, -2.2071290, 2.2075109
3: -2.4377689, 0.1256915, -2.4377689, 0.1256915, -1.7474003, 1.7421572
4: -6.9938774, -2.8966291, -6.9938774, -2.8966291, -3.0747070, 3.0782943
5: -8.9602108, -5.7368851, -8.9602108, -5.7368851, -2.3573170, 2.3435407
6: -19.4462585, -15.5525494, -19.4462585, -15.5525494, -3.1147652, 3.1282711
7: 4.2598271, 6.9828057, 4.2598271, 6.9828057, -2.7229786, 2.7229786
8: -7.1687803, -4.4007745, -7.1687803, -4.4007745, -2.3085465, 2.3026199
9: -7.2100549, -3.7771640, -7.2100549, -3.7771640, -2.6533823, 2.6481481

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1711976, upper bound: 1.1813587
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1712000, upper bound: 1.1790193
time: 5.72 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 25.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.84
Output dim: 7, lower bound: -1.1790202, upper bound: 1.1711997
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 25.84
Output dim: 7, lower bound: -1.1813598, upper bound: 1.1711974
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.84
Output dim: 7, lower bound: -1.1703392, upper bound: 1.1821730
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 25.84
Output dim: 7, lower bound: -1.1703429, upper bound: 1.1799099
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.84
Output dim: 7, lower bound: -1.1711976, upper bound: 1.1813587
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 25.84
Output dim: 7, lower bound: -1.1712000, upper bound: 1.1790193
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.7229785919189453
rel_dist={7: [-1.1847181417998263, 1.1847155369154763]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2340.00 seconds
