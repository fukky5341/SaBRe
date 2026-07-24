## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 176.11861014800002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625)
1: (-113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706)
2: (-160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044)
3: (-81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450)
4: (-173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692)

## BASE Result
execution time: IAR + LP analysis = 1.87 + 1.59 = 3.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -191.4338822, upper bound: 191.4338822


# Binary Search by BASE starts (time budget: 1196.54 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=279.9653625488281
rel_dist={0: [-191.43347747308115, 191.4334774730812]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=279.9653625488281
rel_dist={0: [-191.43286165766193, 191.43286165766187]}

## Binary search (step 3) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=279.9653625488281
rel_dist={0: [-191.43192722771326, 191.43192722771323]}

## Binary search (step 4) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=279.9653625488281
rel_dist={0: [-191.4314584819337, 191.43145848193365]}

## Binary search (step 5) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=279.9653625488281
rel_dist={0: [-191.4312218785597, 191.4312218785597]}

## Binary search (step 6) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=279.9653625488281
rel_dist={0: [-191.43110226199482, 191.43110226199485]}

## Binary search (step 7) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=279.9653625488281
rel_dist={0: [-191.43104245379186, 191.43104245379186]}

## Binary search (step 8) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=279.9653625488281
rel_dist={0: [-191.4310076995096, 191.43100769950962]}

## Binary search (step 9) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=279.9653625488281
rel_dist={0: [-191.43098889644278, 191.43098889644278]}

## Binary search (step 10) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=279.9653625488281
rel_dist={0: [-191.4309794949129, 191.4309794949129]}

## Binary search (step 11) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=279.9653625488281
rel_dist={0: [-191.43097479415502, 191.43097479415508]}

## Binary search (step 12) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=279.9653625488281
rel_dist={0: [-191.43097244379015, 191.4309724437902]}

## Binary search (step 13) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=279.9653625488281
rel_dist={0: [-191.43097126863566, 191.43097126863574]}

## Binary search (step 14) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=279.9653625488281
rel_dist={0: [-191.43097068111345, 191.43097068111354]}

## Binary search (step 15) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=279.9653625488281
rel_dist={0: [-191.4309703874591, 191.43097038745907]}

## Binary search (step 16) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=279.9653625488281
rel_dist={0: [-191.43097024083306, 191.43097024083306]}

## Binary search (step 17) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=279.9653625488281
rel_dist={0: [-191.4309702318314, 191.43097076613515]}

## Binary search (step 18) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=279.9653625488281
rel_dist={0: [-191.4309717958775, 191.43097291164423]}

## Binary Search Result
Binary search time: 66.43 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1130.11 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.3744268, upper bound: 191.3736199
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.3736199, upper bound: 191.3744268
time: 0.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -191.3744268, upper bound: 191.3736199
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -191.3736199, upper bound: 191.3744268

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.1427727, upper bound: 179.1494216
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -179.1427727, upper bound: 179.1494216
time: 0.60 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1512683, upper bound: 191.1433999
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1510053, upper bound: 191.1511107
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -179.1427727, upper bound: 179.1494216
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -179.1427727, upper bound: 179.1494216
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -191.1512683, upper bound: 191.1433999
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -191.1510053, upper bound: 191.1511107

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6749241, upper bound: 176.6749241
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6749241, upper bound: 176.6749241
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7495401, upper bound: 178.6880640
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6826978, upper bound: 178.7629051
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8488281, upper bound: 187.8488281
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8488281, upper bound: 187.8488281
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1433999, upper bound: 191.1511107
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1433999, upper bound: 191.1433999
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -176.6749241, upper bound: 176.6749241
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -176.6749241, upper bound: 176.6749241
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -178.7495401, upper bound: 178.6880640
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -178.6826978, upper bound: 178.7629051
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -187.8488281, upper bound: 187.8488281
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -187.8488281, upper bound: 187.8488281
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -191.1433999, upper bound: 191.1511107
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -191.1433999, upper bound: 191.1433999

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.7133347, upper bound: 178.6485985
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6661281, upper bound: 178.6445050
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3728896
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3915755
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8488281, upper bound: 187.8488281
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8488281, upper bound: 187.8488281
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6390552, upper bound: 187.6390552
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6390552, upper bound: 187.6390552
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0515356, upper bound: 191.0557724
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0515356, upper bound: 191.0519671
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1433999, upper bound: 191.1433999
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1510053, upper bound: 191.1433999
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -178.7133347, upper bound: 178.6485985
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -178.6661281, upper bound: 178.6445050
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3728896
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3915755
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -187.8488281, upper bound: 187.8488281
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -187.8488281, upper bound: 187.8488281
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -187.6390552, upper bound: 187.6390552
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -187.6390552, upper bound: 187.6390552
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -191.0515356, upper bound: 191.0557724
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -191.0515356, upper bound: 191.0519671
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -191.1433999, upper bound: 191.1433999
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -191.1510053, upper bound: 191.1433999

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6098273, upper bound: 178.6127793
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.6768004, upper bound: 178.6089035
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2845294, upper bound: 177.2234655
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2234655, upper bound: 177.2234655
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3728896
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3201875
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.6347977, upper bound: 170.6347977
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -170.6347977, upper bound: 170.6347977
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6390552, upper bound: 187.6390552
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6390552, upper bound: 187.6390552
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4047689, upper bound: 187.4047689
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4047689, upper bound: 187.4047689
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1509256, upper bound: 186.1509256
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1509256, upper bound: 186.1509256
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0381087, upper bound: 191.0384897
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0381087, upper bound: 191.0381087
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1427871, upper bound: 191.1427871
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1427871, upper bound: 191.1427871
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1413828, upper bound: 191.1413828
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1489559, upper bound: 191.1413828
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -178.6098273, upper bound: 178.6127793
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -178.6768004, upper bound: 178.6089035
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -177.2845294, upper bound: 177.2234655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -177.2234655, upper bound: 177.2234655
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3728896
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3201875
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -170.6347977, upper bound: 170.6347977
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -170.6347977, upper bound: 170.6347977
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -187.6390552, upper bound: 187.6390552
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -187.6390552, upper bound: 187.6390552
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -187.4047689, upper bound: 187.4047689
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -187.4047689, upper bound: 187.4047689
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -186.1509256, upper bound: 186.1509256
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -186.1509256, upper bound: 186.1509256
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -191.0381087, upper bound: 191.0384897
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -191.0381087, upper bound: 191.0381087
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -191.1427871, upper bound: 191.1427871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -191.1427871, upper bound: 191.1427871
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -191.1413828, upper bound: 191.1413828
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -191.1489559, upper bound: 191.1413828

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.2089661, upper bound: 177.1889523
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.2123545
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3293274, upper bound: 177.1889523
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.3582922, upper bound: 171.3582922
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.3582922, upper bound: 171.3582922
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3728896
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3201875
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.3247394, upper bound: 175.3247394
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.3247394, upper bound: 175.3247394
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1845159, upper bound: 187.1845159
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1845159, upper bound: 187.1845159
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3969957, upper bound: 187.3969957
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3969957, upper bound: 187.3969957
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4011026, upper bound: 187.4011026
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4011026, upper bound: 187.4011026
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1203789, upper bound: 186.1203789
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1203789, upper bound: 186.1203789
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1352238, upper bound: 185.1352238
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1352238, upper bound: 185.1352238
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0244199, upper bound: 191.0247934
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0244199, upper bound: 191.0247471
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7375488, upper bound: 190.7375488
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7375488, upper bound: 190.7375488
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1297351, upper bound: 191.1297351
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1297351, upper bound: 191.1297351
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6366955, upper bound: 175.6366955
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.6366955, upper bound: 175.6366955
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1407773, upper bound: 191.1407773
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1407773, upper bound: 191.1407773
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1308088, upper bound: 191.1280646
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1280646, upper bound: 191.1280646
time: 0.56 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6283719
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -177.2089661, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -177.1889523, upper bound: 177.2123545
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -177.3293274, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -171.3582922, upper bound: 171.3582922
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -171.3582922, upper bound: 171.3582922
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3728896
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3201875
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -175.3247394, upper bound: 175.3247394
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -175.3247394, upper bound: 175.3247394
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.1845159, upper bound: 187.1845159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.1845159, upper bound: 187.1845159
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.3969957, upper bound: 187.3969957
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.3969957, upper bound: 187.3969957
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.4011026, upper bound: 187.4011026
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.4011026, upper bound: 187.4011026
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -186.1203789, upper bound: 186.1203789
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -186.1203789, upper bound: 186.1203789
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -185.1352238, upper bound: 185.1352238
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -185.1352238, upper bound: 185.1352238
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -191.0244199, upper bound: 191.0247934
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -191.0244199, upper bound: 191.0247471
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -190.7375488, upper bound: 190.7375488
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -190.7375488, upper bound: 190.7375488
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -191.1297351, upper bound: 191.1297351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -191.1297351, upper bound: 191.1297351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -175.6366955, upper bound: 175.6366955
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -175.6366955, upper bound: 175.6366955
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -191.1407773, upper bound: 191.1407773
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -191.1407773, upper bound: 191.1407773
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -191.1308088, upper bound: 191.1280646
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -191.1280646, upper bound: 191.1280646

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.35 seconds

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6228293
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6228293, upper bound: 176.6283719
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 3.35 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5993324, upper bound: 176.5993324
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5993324, upper bound: 176.5993324
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 25

Time for candidate selection: 3.35 seconds

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5967088, upper bound: 176.5927243
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5927243, upper bound: 176.5967088
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 3.61 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5993324, upper bound: 176.5993324
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5993324, upper bound: 176.5993324
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 3.45 seconds

### Candidate
type: RSZ, layer: 3, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5967088, upper bound: 176.5927243
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5927243, upper bound: 176.5967088
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 3.36 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7559880, upper bound: 175.7559880
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.7559880, upper bound: 175.7559880
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 26

Time for candidate selection: 3.42 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4230835, upper bound: 176.4229387
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4229387, upper bound: 176.4230835
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.36 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4230835, upper bound: 176.4229387
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4229387, upper bound: 176.4230835
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 3.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6257471
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6257471, upper bound: 176.6283719
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 21

Time for candidate selection: 3.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5926315, upper bound: 176.5926315
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5926315, upper bound: 176.5926315
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 18

Time for candidate selection: 3.40 seconds

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5926315, upper bound: 176.5926315
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5926315, upper bound: 176.5926315
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 3.62 seconds

### Candidate
type: RSZ, layer: 3, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6228293
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6228293, upper bound: 176.6283719
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 3.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6228293
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6228293, upper bound: 176.6283719
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 3.43 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6228293
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6228293, upper bound: 176.6283719
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 18

Time for candidate selection: 3.46 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6257471
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6257471, upper bound: 176.6283719
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 3.46 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4230835, upper bound: 176.4229387
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4229387, upper bound: 176.4230835
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2712117, upper bound: 175.9279816
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2712117, upper bound: 175.9279816
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.2123545
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.2844947, upper bound: 171.2844947
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -171.2844947, upper bound: 171.2844947
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9279816, upper bound: 175.9279816
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9279816, upper bound: 175.9279816
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0086779, upper bound: 176.0086779
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0086779, upper bound: 176.0938498
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3201875
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3201875
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.57 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6228293
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6228293, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5993324, upper bound: 176.5993324
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5993324, upper bound: 176.5993324
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5967088, upper bound: 176.5927243
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5927243, upper bound: 176.5967088
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5993324, upper bound: 176.5993324
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5993324, upper bound: 176.5993324
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5967088, upper bound: 176.5927243
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5927243, upper bound: 176.5967088
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.79
Output dim: 0, lower bound: -175.7559880, upper bound: 175.7559880
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.79
Output dim: 0, lower bound: -175.7559880, upper bound: 175.7559880
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.4230835, upper bound: 176.4229387
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.4229387, upper bound: 176.4230835
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.4230835, upper bound: 176.4229387
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.4229387, upper bound: 176.4230835
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6257471
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6257471, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5926315, upper bound: 176.5926315
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5926315, upper bound: 176.5926315
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5926315, upper bound: 176.5926315
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.5926315, upper bound: 176.5926315
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6228293
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6228293, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6228293
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6228293, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6228293
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6228293, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6283719, upper bound: 176.6257471
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.6257471, upper bound: 176.6283719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.4230835, upper bound: 176.4229387
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.4229387, upper bound: 176.4230835
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.2712117, upper bound: 175.9279816
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.2712117, upper bound: 175.9279816
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -177.1889523, upper bound: 177.2123545
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.79
Output dim: 0, lower bound: -171.2844947, upper bound: 171.2844947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.79
Output dim: 0, lower bound: -171.2844947, upper bound: 171.2844947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -177.1889523, upper bound: 177.1889523
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.79
Output dim: 0, lower bound: -175.9279816, upper bound: 175.9279816
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.79
Output dim: 0, lower bound: -175.9279816, upper bound: 175.9279816
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.0086779, upper bound: 176.0086779
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.79
Output dim: 0, lower bound: -176.0086779, upper bound: 176.0938498
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3201875
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -177.3201875, upper bound: 177.3201875
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.3932687, upper bound: 187.3932687
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.6240239, upper bound: 187.6240239
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.79
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -186.1203789, upper bound: 186.1203789
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -186.1203789, upper bound: 186.1203789
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -185.1352238, upper bound: 185.1352238
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -185.1352238, upper bound: 185.1352238
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -191.0244199, upper bound: 191.0247934
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -191.0244199, upper bound: 191.0247471
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -190.7375488, upper bound: 190.7375488
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -190.7375488, upper bound: 190.7375488
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -191.1297351, upper bound: 191.1297351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -191.1297351, upper bound: 191.1297351
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -191.1407773, upper bound: 191.1407773
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -191.1407773, upper bound: 191.1407773
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -191.1308088, upper bound: 191.1280646
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 0, lower bound: -191.1280646, upper bound: 191.1280646
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.4221294, upper bound: 191.4229799
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.4229799, upper bound: 191.4221294
time: 0.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -191.4221294, upper bound: 191.4229799
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -191.4229799, upper bound: 191.4221294

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.3572181, upper bound: 191.3579123
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.3572889, upper bound: 191.3571734
time: 0.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8868423, upper bound: 187.8868423
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8868423, upper bound: 187.8868423
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -191.3572181, upper bound: 191.3579123
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -191.3572889, upper bound: 191.3571734
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -187.8868423, upper bound: 187.8868423
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -187.8868423, upper bound: 187.8868423

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.3572181, upper bound: 191.3579123
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.3571734, upper bound: 191.3571734
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.2284236, upper bound: 191.2284236
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.2300188, upper bound: 191.2284236
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8868423, upper bound: 187.8868423
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8868423, upper bound: 187.8868423
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4429912, upper bound: 187.4429912
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4429912, upper bound: 187.4429912
time: 0.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -191.3572181, upper bound: 191.3579123
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -191.3571734, upper bound: 191.3571734
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -191.2284236, upper bound: 191.2284236
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -191.2300188, upper bound: 191.2284236
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -187.8868423, upper bound: 187.8868423
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -187.8868423, upper bound: 187.8868423
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -187.4429912, upper bound: 187.4429912
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 0, lower bound: -187.4429912, upper bound: 187.4429912

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.2284236, upper bound: 191.2304910
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.2285139, upper bound: 191.2284236
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1990362, upper bound: 186.1990362
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1990362, upper bound: 186.1990362
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.2284236, upper bound: 191.2284236
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.2284236, upper bound: 191.2284236
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0391861, upper bound: 191.0389489
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8778876, upper bound: 187.8778876
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8778876, upper bound: 187.8778876
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8794612, upper bound: 187.8794612
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8794612, upper bound: 187.8794612
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4131650, upper bound: 187.4131650
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4131650, upper bound: 187.4131650
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4429912, upper bound: 187.4429912
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4429912, upper bound: 187.4429912
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -191.2284236, upper bound: 191.2304910
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -191.2285139, upper bound: 191.2284236
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -186.1990362, upper bound: 186.1990362
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -186.1990362, upper bound: 186.1990362
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -191.2284236, upper bound: 191.2284236
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -191.2284236, upper bound: 191.2284236
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -191.0391861, upper bound: 191.0389489
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -187.8778876, upper bound: 187.8778876
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -187.8778876, upper bound: 187.8778876
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -187.8794612, upper bound: 187.8794612
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -187.8794612, upper bound: 187.8794612
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -187.4131650, upper bound: 187.4131650
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -187.4131650, upper bound: 187.4131650
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -187.4429912, upper bound: 187.4429912
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -187.4429912, upper bound: 187.4429912

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9182257, upper bound: 190.9200173
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9182257, upper bound: 190.9191712
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0389914
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9043249, upper bound: 185.9043249
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9043249, upper bound: 185.9043249
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1827770, upper bound: 186.1827770
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1827770, upper bound: 186.1827770
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.2283606, upper bound: 191.2283606
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.2283606, upper bound: 191.2283606
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7385340, upper bound: 185.7385340
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7385340, upper bound: 185.7385340
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8705065, upper bound: 187.8705065
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.8705065, upper bound: 187.8705065
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6599132, upper bound: 187.6599132
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6599132, upper bound: 187.6599132
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4379514, upper bound: 187.4379514
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4379514, upper bound: 187.4379514
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6625595, upper bound: 187.6625595
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6625595, upper bound: 187.6625595
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4047341, upper bound: 187.4047341
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4047341, upper bound: 187.4047341
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2149138, upper bound: 187.2149138
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2149138, upper bound: 187.2149138
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4345699, upper bound: 187.4345699
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4345699, upper bound: 187.4345699
time: 0.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -190.9182257, upper bound: 190.9200173
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -190.9182257, upper bound: 190.9191712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0389914
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -185.9043249, upper bound: 185.9043249
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -185.9043249, upper bound: 185.9043249
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -186.1827770, upper bound: 186.1827770
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -186.1827770, upper bound: 186.1827770
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -191.2283606, upper bound: 191.2283606
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -191.2283606, upper bound: 191.2283606
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -185.7385340, upper bound: 185.7385340
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -185.7385340, upper bound: 185.7385340
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.8705065, upper bound: 187.8705065
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.8705065, upper bound: 187.8705065
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.6599132, upper bound: 187.6599132
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.6599132, upper bound: 187.6599132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.4379514, upper bound: 187.4379514
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.4379514, upper bound: 187.4379514
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.6625595, upper bound: 187.6625595
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.6625595, upper bound: 187.6625595
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.4047341, upper bound: 187.4047341
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.4047341, upper bound: 187.4047341
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.2149138, upper bound: 187.2149138
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.2149138, upper bound: 187.2149138
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.4345699, upper bound: 187.4345699
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -187.4345699, upper bound: 187.4345699

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7372949
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7375904
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8946312, upper bound: 190.8955273
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8946312, upper bound: 190.8956949
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9053631, upper bound: 190.9053631
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9053631, upper bound: 190.9053631
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8566858, upper bound: 185.8566858
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8566858, upper bound: 185.8566858
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8840736, upper bound: 185.8840736
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8840736, upper bound: 185.8840736
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1179725, upper bound: 186.1179725
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1179725, upper bound: 186.1179725
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0237014, upper bound: 186.0237014
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0237014, upper bound: 186.0237014
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9182219, upper bound: 190.9182219
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9182219, upper bound: 190.9182219
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.5597583, upper bound: 189.5597583
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.5597583, upper bound: 189.5597583
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7357394
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7357394
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7385340, upper bound: 185.7385340
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7385340, upper bound: 185.7385340
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0697573, upper bound: 184.0697573
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0697573, upper bound: 184.0697573
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4295301, upper bound: 187.4295301
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4295301, upper bound: 187.4295301
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4295301, upper bound: 187.4295301
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4295301, upper bound: 187.4295301
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6329580, upper bound: 187.6329580
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.6329580, upper bound: 187.6329580
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4047341, upper bound: 187.4047341
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.4047341, upper bound: 187.4047341
time: 0.60 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7372949
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7375904
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.8946312, upper bound: 190.8955273
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.8946312, upper bound: 190.8956949
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.9053631, upper bound: 190.9053631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.9053631, upper bound: 190.9053631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -185.8566858, upper bound: 185.8566858
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -185.8566858, upper bound: 185.8566858
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -185.8840736, upper bound: 185.8840736
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -185.8840736, upper bound: 185.8840736
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -186.1179725, upper bound: 186.1179725
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -186.1179725, upper bound: 186.1179725
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -186.0237014, upper bound: 186.0237014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -186.0237014, upper bound: 186.0237014
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.9182219, upper bound: 190.9182219
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.9182219, upper bound: 190.9182219
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -189.5597583, upper bound: 189.5597583
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -189.5597583, upper bound: 189.5597583
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7357394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7357394
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -191.0388309, upper bound: 191.0388309
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -185.7385340, upper bound: 185.7385340
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -185.7385340, upper bound: 185.7385340
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -184.0697573, upper bound: 184.0697573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -184.0697573, upper bound: 184.0697573
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4295301, upper bound: 187.4295301
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4295301, upper bound: 187.4295301
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4295301, upper bound: 187.4295301
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4295301, upper bound: 187.4295301
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.6303433, upper bound: 187.6303433
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.6329580, upper bound: 187.6329580
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.6329580, upper bound: 187.6329580
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4081251, upper bound: 187.4081251
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2098740, upper bound: 187.2098740
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.2064212, upper bound: 187.2064212
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4047341, upper bound: 187.4047341
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -187.4047341, upper bound: 187.4047341

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.6686285, upper bound: 185.6686285
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.6686285, upper bound: 185.6686285
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7204961, upper bound: 190.7223806
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7204961, upper bound: 190.7225202
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8942373, upper bound: 190.8942373
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8942373, upper bound: 190.8942373
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8946312, upper bound: 190.8956949
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.8946312, upper bound: 190.8946988
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0362785, upper bound: 191.0362785
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0362785, upper bound: 191.0362785
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6055245, upper bound: 190.6055245
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.6055245, upper bound: 190.6055245
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9043297, upper bound: 190.9043297
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.9043297, upper bound: 190.9043297
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8402365, upper bound: 185.8402365
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8402365, upper bound: 185.8402365
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7898522, upper bound: 185.7898522
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7898522, upper bound: 185.7898522
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9592944, upper bound: 185.9592944
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9592944, upper bound: 185.9592944
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1362697, upper bound: 183.1362697
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1362697, upper bound: 183.1362697
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9533561, upper bound: 185.9533561
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9533561, upper bound: 185.9533561
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.1544212, upper bound: 189.1544212
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.1544212, upper bound: 189.1544212
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.1544212, upper bound: 189.1544212
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.1544212, upper bound: 189.1544212
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.5236379, upper bound: 189.5236379
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.5236379, upper bound: 189.5236379
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.5597583, upper bound: 189.5597583
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -189.5597583, upper bound: 189.5597583
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7179622, upper bound: 185.7179622
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7339298, upper bound: 190.7339298
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7339298, upper bound: 190.7339298
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7357394
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7357394, upper bound: 190.7357394
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7204961, upper bound: 190.7204961
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -190.7204961, upper bound: 190.7204961
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0368679, upper bound: 191.0368679
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0697573, upper bound: 184.0697573
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0697573, upper bound: 184.0697573
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0697573, upper bound: 184.0697573
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0697573, upper bound: 184.0697573
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0473381, upper bound: 184.0473381
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0473381, upper bound: 184.0473381
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8331083, upper bound: 182.8331083
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8331083, upper bound: 182.8331083
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0232146, upper bound: 184.0232146
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.0232146, upper bound: 184.0232146
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.7800601, upper bound: 181.7800601
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.7800601, upper bound: 181.7800601
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.2013814, upper bound: 187.2013814
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1794760, upper bound: 187.1794760
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1709873, upper bound: 187.1709873
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.1760271, upper bound: 187.1760271
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.3996943, upper bound: 187.3996943
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.10 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=279.9653625488281
rel_dist={0: [-191.43347747308115, 191.4334774730812]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8008666, upper bound: 186.8008666
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.8008666, upper bound: 186.8008666
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 0, lower bound: -186.8008666, upper bound: 186.8008666
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 0, lower bound: -186.8008666, upper bound: 186.8008666

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.7550678, upper bound: 186.7542107
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.7542107, upper bound: 186.7550678
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.7658410, upper bound: 186.7666972
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.7666972, upper bound: 186.7658410
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 0, lower bound: -186.7550678, upper bound: 186.7542107
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 0, lower bound: -186.7542107, upper bound: 186.7550678
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 0, lower bound: -186.7658410, upper bound: 186.7666972
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 0, lower bound: -186.7666972, upper bound: 186.7658410

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.6642981, upper bound: 186.6640811
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.6650955, upper bound: 186.6640811
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.6640811, upper bound: 186.6650955
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.6640811, upper bound: 186.6642981
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1835560
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1843526
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3115250, upper bound: 186.3108140
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.3108140, upper bound: 186.3108140
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -186.6642981, upper bound: 186.6640811
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -186.6650955, upper bound: 186.6640811
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -186.6640811, upper bound: 186.6650955
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -186.6640811, upper bound: 186.6642981
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1835560
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1843526
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -186.3115250, upper bound: 186.3108140
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -186.3108140, upper bound: 186.3108140

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.6144272, upper bound: 186.6144272
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.6152736, upper bound: 186.6144272
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1856903, upper bound: 186.1856903
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1856903, upper bound: 186.1858440
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1835560
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1835560
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1835560
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1843526
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.0708093, upper bound: 172.0708093
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -172.0708093, upper bound: 172.0708093
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2757550, upper bound: 186.2757550
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.2757550, upper bound: 186.2757550
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.6144272, upper bound: 186.6144272
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.6152736, upper bound: 186.6144272
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.1856903, upper bound: 186.1856903
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.1856903, upper bound: 186.1858440
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1835560
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1835560
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1835560
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.1835560, upper bound: 186.1843526
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -172.0708093, upper bound: 172.0708093
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -172.0708093, upper bound: 172.0708093
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.2757550, upper bound: 186.2757550
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -186.2757550, upper bound: 186.2757550

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.3376225, upper bound: 182.2710726
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.3376225, upper bound: 182.2710726
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.5805337, upper bound: 186.5805337
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.5805337, upper bound: 186.5805337
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9925557, upper bound: 185.9925557
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9925557, upper bound: 185.9925557
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.9208040, upper bound: 182.9208040
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.9208040, upper bound: 182.9208040
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1344270, upper bound: 185.1344270
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.1344270, upper bound: 185.1344270
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1198625, upper bound: 186.1200907
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1198625, upper bound: 186.1198892
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1199367, upper bound: 186.1199367
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1199367, upper bound: 186.1199367
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1815490, upper bound: 186.1815490
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1815490, upper bound: 186.1822329
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0920939, upper bound: 183.0920939
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0920939, upper bound: 183.0920939
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1563591, upper bound: 186.1563591
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.1563591, upper bound: 186.1563591
time: 0.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.0655692, upper bound: 186.0655692
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -182.3376225, upper bound: 182.2710726
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -182.3376225, upper bound: 182.2710726
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.5805337, upper bound: 186.5805337
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.5805337, upper bound: 186.5805337
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -185.9925557, upper bound: 185.9925557
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -185.9925557, upper bound: 185.9925557
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -182.9208040, upper bound: 182.9208040
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -182.9208040, upper bound: 182.9208040
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -185.1344270, upper bound: 185.1344270
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -185.1344270, upper bound: 185.1344270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.1198625, upper bound: 186.1200907
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.1198625, upper bound: 186.1198892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.1199367, upper bound: 186.1199367
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.1199367, upper bound: 186.1199367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.1815490, upper bound: 186.1815490
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.1815490, upper bound: 186.1822329
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -183.0920939, upper bound: 183.0920939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -183.0920939, upper bound: 183.0920939
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.1563591, upper bound: 186.1563591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 0, lower bound: -186.1563591, upper bound: 186.1563591

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4498051, upper bound: 178.4498051
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4498051, upper bound: 178.4498051
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.2254057, upper bound: 182.1577936
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.1577936, upper bound: 182.1577936
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.7783678, upper bound: 182.7783678
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.7783678, upper bound: 182.7783678
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9925557, upper bound: 185.9925557
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9925557, upper bound: 185.9925557
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8670034, upper bound: 182.8670034
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8670034, upper bound: 182.8670034
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8960716, upper bound: 182.8960716
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8960716, upper bound: 182.8960716
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8688451, upper bound: 181.8688451
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8688451, upper bound: 181.8688451
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.0629580, upper bound: 185.0629580
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.0629580, upper bound: 185.0629580
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0893400, upper bound: 186.0895719
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0893400, upper bound: 186.0893400
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9960511, upper bound: 185.9960511
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9960511, upper bound: 185.9960511
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0763965, upper bound: 183.0763965
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0763965, upper bound: 183.0763965
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8921059, upper bound: 178.8921059
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.8921059, upper bound: 178.8921059
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0763965, upper bound: 183.0763965
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0763965, upper bound: 183.0763965
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0288176, upper bound: 183.0288176
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0288176, upper bound: 183.0288176
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.3884731, upper bound: 184.3884731
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.3884731, upper bound: 184.3884731
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.3884731, upper bound: 184.3884731
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -184.3884731, upper bound: 184.3884731
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0920939, upper bound: 183.0920939
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0920939, upper bound: 183.0920939
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.9759882, upper bound: 182.9759882
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.9759882, upper bound: 182.9759882
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0307132, upper bound: 186.0307132
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0307132, upper bound: 186.0307132
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0926117, upper bound: 186.0926117
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -186.0926117, upper bound: 186.0926117
time: 0.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9758147, upper bound: 185.9758147
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -178.4498051, upper bound: 178.4498051
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -178.4498051, upper bound: 178.4498051
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.2254057, upper bound: 182.1577936
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.1577936, upper bound: 182.1577936
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.7783678, upper bound: 182.7783678
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.7783678, upper bound: 182.7783678
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9925557, upper bound: 185.9925557
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9925557, upper bound: 185.9925557
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.8670034, upper bound: 182.8670034
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.8670034, upper bound: 182.8670034
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.8960716, upper bound: 182.8960716
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.8960716, upper bound: 182.8960716
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -181.8688451, upper bound: 181.8688451
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -181.8688451, upper bound: 181.8688451
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.0629580, upper bound: 185.0629580
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.0629580, upper bound: 185.0629580
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -186.0893400, upper bound: 186.0895719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -186.0893400, upper bound: 186.0893400
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9960511, upper bound: 185.9960511
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -185.9960511, upper bound: 185.9960511
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.0763965, upper bound: 183.0763965
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.0763965, upper bound: 183.0763965
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -178.8921059, upper bound: 178.8921059
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -178.8921059, upper bound: 178.8921059
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.1440145, upper bound: 183.1440145
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.0763965, upper bound: 183.0763965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.0763965, upper bound: 183.0763965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.0288176, upper bound: 183.0288176
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.0288176, upper bound: 183.0288176
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -184.3884731, upper bound: 184.3884731
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -184.3884731, upper bound: 184.3884731
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -184.3884731, upper bound: 184.3884731
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -184.3884731, upper bound: 184.3884731
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.0920939, upper bound: 183.0920939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -183.0920939, upper bound: 183.0920939
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.9759882, upper bound: 182.9759882
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -182.9759882, upper bound: 182.9759882
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -186.0307132, upper bound: 186.0307132
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -186.0307132, upper bound: 186.0307132
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -186.0926117, upper bound: 186.0926117
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -186.0926117, upper bound: 186.0926117

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9063857, upper bound: 185.9063857
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9063857, upper bound: 185.9063857
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.9077248, upper bound: 181.9077248
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7265363, upper bound: 185.7265363
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.7265363, upper bound: 185.7265363
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9499653, upper bound: 185.9499653
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9499653, upper bound: 185.9499653
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2601968, upper bound: 178.2601968
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2601968, upper bound: 178.2601968
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2601968, upper bound: 178.2601968
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2601968, upper bound: 178.2601968
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4135272, upper bound: 178.4135272
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4135272, upper bound: 178.4135272
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8446804, upper bound: 181.8446804
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8446804, upper bound: 181.8446804
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2145769, upper bound: 178.2145769
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2145769, upper bound: 178.2145769
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2145769, upper bound: 178.2145769
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.2145769, upper bound: 178.2145769
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8760209, upper bound: 185.8760209
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8760209, upper bound: 185.8760209
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.0539967, upper bound: 182.0539967
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.0539967, upper bound: 182.0539967
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9023996, upper bound: 185.9023996
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9023996, upper bound: 185.9023996
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.1106327, upper bound: 182.1106327
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.1106327, upper bound: 182.1106327
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.9661754, upper bound: 185.9661754
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8892728, upper bound: 185.8892728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.8892728, upper bound: 185.8892728
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.7480622, upper bound: 181.7480622
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.7480622, upper bound: 181.7480622
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.6009201, upper bound: 181.6009201
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.6009201, upper bound: 181.6009201
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8547123, upper bound: 181.8547123
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8547123, upper bound: 181.8547123
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.6009201, upper bound: 181.6009201
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.6009201, upper bound: 181.6009201
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.7974376, upper bound: 182.7974376
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.7974376, upper bound: 182.7974376
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.7486966, upper bound: 182.7486966
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.7486966, upper bound: 182.7486966
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4049567, upper bound: 178.4049567
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -178.4049567, upper bound: 178.4049567
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8423093, upper bound: 182.8423093
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -182.8423093, upper bound: 182.8423093
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8688451, upper bound: 181.8688451
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8688451, upper bound: 181.8688451
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8181250, upper bound: 181.8181250
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -181.8181250, upper bound: 181.8181250
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9898115, upper bound: 177.9898115
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9898115, upper bound: 177.9898115
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.0170999, upper bound: 185.0170999
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.0170999, upper bound: 185.0170999
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.0686748, upper bound: 185.0686748
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.0686748, upper bound: 185.0686748
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.6368862, upper bound: 185.6368862
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.6368862, upper bound: 185.6368862
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.6525138, upper bound: 185.6525138
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.6525138, upper bound: 185.6525138
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.5869792, upper bound: 185.5869792
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -185.5869792, upper bound: 185.5886570
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0681533, upper bound: 183.0681533
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -183.0681533, upper bound: 183.0681533
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=279.9653625488281
rel_dist={0: [-191.43286165766193, 191.43286165766187]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1131.99 seconds
