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
execution time: IAR + LP analysis = 1.88 + 1.58 = 3.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -191.4338822, upper bound: 191.4338822


# Binary Search by BASE starts (time budget: 1196.55 seconds, max iter: 100)

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
Binary search time: 66.44 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1130.11 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.30 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0593939, upper bound: 191.0582352
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0592475, upper bound: 191.0582352
time: 0.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -191.0593939, upper bound: 191.0582352
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -191.0592475, upper bound: 191.0582352

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9703431, upper bound: 177.8637569
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9703431, upper bound: 177.8637569
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9373723, upper bound: 177.8637569
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9373723, upper bound: 177.8637569
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9373723
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9373723
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9703431
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9703431
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -177.9703431, upper bound: 177.8637569
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -177.9703431, upper bound: 177.8637569
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -177.9373723, upper bound: 177.8637569
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -177.9373723, upper bound: 177.8637569
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9373723
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9373723
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9703431
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9703431

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.8980727, upper bound: 176.6145531
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.8980727, upper bound: 176.6145531
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.8980727, upper bound: 176.6145531
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.8980727, upper bound: 176.6145531
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.57 seconds

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
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.8980727
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.8980727
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.8980727
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.8980727
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.8980727, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.8980727, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.8980727, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.8980727, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6145531, upper bound: 176.8980727
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6145531, upper bound: 176.8980727
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6145531, upper bound: 176.8980727
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -176.6145531, upper bound: 176.8980727

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6706050, upper bound: 176.3780963
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4421249, upper bound: 176.3780963
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6706050, upper bound: 176.3780963
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4291844, upper bound: 176.3780963
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6706050, upper bound: 176.3780963
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4421249, upper bound: 176.3780963
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6706050, upper bound: 176.3780963
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4291844, upper bound: 176.3780963
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4291844
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.6706050
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4421249
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.6706050
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4291844
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.6706050
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4421249
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.6706050
time: 0.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.6706050, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.4421249, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.6706050, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.4291844, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.6706050, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.4421249, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.6706050, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.4291844, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4291844
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.6706050
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4421249
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.6706050
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4291844
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.6706050
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4421249
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -176.3780963, upper bound: 176.6706050

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0787205, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0556026, upper bound: 175.9713576
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0787205, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0556026, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0556026
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0787205
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
time: 0.53 seconds

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0556026
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0787205
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
time: 0.54 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.0787205, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.0556026, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.0787205, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.0556026, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0556026
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0787205
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0556026
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0787205
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.62 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 0, lower bound: -176.3133010, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3133010
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.72
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
time: 0.63 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.71
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
Binary search (step 0): status=Status.VERIFIED, low=0.2500000, high=0.5000000, mid=0.2500000, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 1) starts
Candidate diff: 0.3750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.26 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0593939, upper bound: 191.0582352
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0592475, upper bound: 191.0582352
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -191.0593939, upper bound: 191.0582352
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -191.0592475, upper bound: 191.0582352

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
time: 0.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4440788, upper bound: 176.3780963
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4296832, upper bound: 176.3780963
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4440788, upper bound: 176.3780963
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4296832, upper bound: 176.3780963
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4296832
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4440788
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4296832
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4440788
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.4440788, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.4296832, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.4440788, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.4296832, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4296832
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4440788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4296832
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4440788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.66 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
time: 0.58 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
Binary search (step 1): status=Status.VERIFIED, low=0.3750000, high=0.5000000, mid=0.3750000, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 2) starts
Candidate diff: 0.4375000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0582352
time: 0.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0592475, upper bound: 191.0582352
time: 0.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0582352
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -191.0592475, upper bound: 191.0582352

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4440788, upper bound: 176.3780963
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4296832, upper bound: 176.3780963
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4440788, upper bound: 176.3780963
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.4296832, upper bound: 176.3780963
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4296832
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4440788
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4296832
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4440788
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
time: 0.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.4440788, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.4296832, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.4440788, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.7059803, upper bound: 176.3780963
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.4296832, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3822470, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3822470
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.3780963
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4296832
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4440788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4296832
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.4440788
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -176.3780963, upper bound: 176.7059803

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.53 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
time: 0.62 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.26
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
Binary search (step 2): status=Status.VERIFIED, low=0.4375000, high=0.5000000, mid=0.4375000, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 3) starts
Candidate diff: 0.4687500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919
time: 0.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0593939, upper bound: 191.0582352
time: 0.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0582352
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -191.0593939, upper bound: 191.0582352
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0582352

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
time: 0.63 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9388839
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -177.8637569, upper bound: 177.9840300

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.9376564, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6203471, upper bound: 176.6145531
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6145531, upper bound: 176.6203471
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.26
Output dim: 0, lower bound: -176.6145531, upper bound: 176.9376564

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5913844, upper bound: 176.2050121
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5864639, upper bound: 176.2050121
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5913844, upper bound: 176.2050121
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.5864639, upper bound: 176.2050121
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2229260, upper bound: 176.2050121
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2229260, upper bound: 176.2050121
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2229260, upper bound: 176.2050121
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2229260, upper bound: 176.2050121
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2229260
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2229260
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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2229260
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2229260
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.5864639
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.5913844
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.5864639
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.2050121, upper bound: 176.5913844
time: 0.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.5913844, upper bound: 176.2050121
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.5864639, upper bound: 176.2050121
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.5913844, upper bound: 176.2050121
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.5864639, upper bound: 176.2050121
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2229260, upper bound: 176.2050121
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2229260, upper bound: 176.2050121
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2229260, upper bound: 176.2050121
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2229260, upper bound: 176.2050121
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2229260
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2229260
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2229260
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2229260
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.5864639
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.5913844
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.5864639
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.2050121
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.26
Output dim: 0, lower bound: -176.2050121, upper bound: 176.5913844

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.57 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -176.0934929, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -176.0637350, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9875132, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9875132
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0637350
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 176.0934929
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
time: 0.66 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 0, lower bound: -176.3627861, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 0, lower bound: -176.3578226, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3578226
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 176.3627861
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.99
Output dim: 0, lower bound: -175.9713576, upper bound: 175.9713576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
time: 0.62 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.9276658, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.9224846, upper bound: 174.6024702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9224846
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.6024702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.96
Output dim: 0, lower bound: -174.6024702, upper bound: 174.9276658
Binary search (step 3): status=Status.VERIFIED, low=0.4687500, high=0.5000000, mid=0.4687500, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947481]}

## Binary search (step 4) starts
Candidate diff: 0.4843750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 0, lower bound: -191.1353919, upper bound: 191.1351532
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 0, lower bound: -191.1351532, upper bound: 191.1353919

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0593939, upper bound: 191.0582352
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -191.0592475, upper bound: 191.0582352
time: 0.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0592475
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -191.0593939, upper bound: 191.0582352
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -191.0582352, upper bound: 191.0593939
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 0, lower bound: -191.0592475, upper bound: 191.0582352

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9840300, upper bound: 177.8637569
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -177.9388839, upper bound: 177.8637569
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625
1: -113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706
2: -160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044
3: -81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450
4: -173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692

Time for backsubstitution: 1.88 seconds
Binary search (step 4): status=Status.UNKNOWN, low=0.4687500, high=0.4843750, mid=0.4843750, abs_max=279.9653625488281
rel_dist={0: [-191.43388219474815, 191.4338821947482]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.46875
execution time: 1131.54 seconds
